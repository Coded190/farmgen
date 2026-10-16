"""One web photograph per asset, chosen with the scene seed.

The search and the download are injected in tests. A miss returns no file so
the caller can fall back to Flux.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
from pathlib import Path

MIN_BYTES = 2048
MIN_SIDE = 64
PHOTO_TRIES = 5


def ordered_urls(urls: list[str], seed: int, asset_id: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """Rotate the result list by the seed so a rerun starts at the same URL."""
    skipped = set(exclude)
    choices = [url for url in urls if url not in skipped]
    if not choices:
        return []
    start = int.from_bytes(hashlib.sha256(f"{seed}|{asset_id}".encode()).digest()[:4], "big") % len(choices)
    return choices[start:] + choices[:start]


def fetch_photo(
    prompt: str,
    seed: int,
    asset_id: str,
    dest: Path,
    exclude: tuple[str, ...] = (),
    search=None,
    download=None,
) -> tuple[Path | None, str]:
    """Save one usable photo. Return ``(None, "")`` when every URL misses."""
    search = bing_image_urls if search is None else search
    download = download_image if download is None else download
    try:
        urls = list(search(prompt))
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"error:{exc}"
    del seed, asset_id
    for url in _ranked_urls(urls, exclude):
        try:
            if download(url, dest):
                return dest, url
        except (OSError, RuntimeError, ValueError):
            continue
    return None, ""


def take_accepted_photo(
    prompt: str,
    seed: int,
    asset_id: str,
    dest: Path,
    judge=None,
    exclude: tuple[str, ...] = (),
    search=None,
    download=None,
    attempts: int = PHOTO_TRIES,
    seen: list[str] | None = None,
    gate_dir: Path | None = None,
) -> tuple[Path | None, str]:
    """Save the first photo ``judge`` accepts. A reject tries the next URL.

    ``judge(path, attempt)`` returns a bool or an object with ``accept``.
    ``(None, "")`` means every candidate was rejected or missing, so the
    caller can fall back to Flux.
    """
    folder = gate_dir if gate_dir is not None else dest.parent / "photo_gate"
    skipped = list(exclude)
    for attempt in range(1, attempts + 1):
        candidate = folder / f"{asset_id}_attempt_{attempt}.png"
        path, url = fetch_photo(
            prompt,
            seed,
            asset_id,
            candidate,
            exclude=tuple(skipped),
            search=search,
            download=download,
        )
        if path is None:
            return None, ""
        skipped.append(url)
        if seen is not None and url not in seen:
            seen.append(url)
        if _accepted(judge, path, attempt):
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != dest.resolve():
                shutil.copy2(path, dest)
            return dest, url
    return None, ""


def _ranked_urls(urls: list[str], exclude: tuple[str, ...] = ()) -> list[str]:
    """Bing's order. The first URL is rank 1. Already tried URLs are skipped."""
    skipped = set(exclude)
    ranked: list[str] = []
    for url in urls:
        if url in skipped or url in ranked:
            continue
        ranked.append(url)
    return ranked


def _accepted(judge, path: Path, attempt: int) -> bool:
    if judge is None:
        return True
    decision = judge(path, attempt)
    if isinstance(decision, bool):
        return decision
    return bool(getattr(decision, "accept", False))


def download_image(url: str, dest: Path, get=None) -> bool:
    """Write a PNG when the response is a large enough picture."""
    if get is None:
        import requests

        response = requests.get(url, timeout=30, headers={"User-Agent": "farmgen/1.0"})
        response.raise_for_status()
        raw = response.content
    else:
        raw = get(url)
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < MIN_BYTES:
        return False
    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if image.width < MIN_SIDE or image.height < MIN_SIDE:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="PNG")
    return True


def duckduckgo_image_urls(query: str, limit: int = 8) -> list[str]:
    """Return image URLs from DuckDuckGo. A bot challenge is an error, not an empty list."""
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "farmgen/1.0"
    home = session.post("https://duckduckgo.com/", data={"q": query}, timeout=20)
    home.raise_for_status()
    match = re.search(r"vqd=([\d-]+)", home.text) or re.search(r'vqd="([^"]+)"', home.text)
    if match is None:
        return []
    listing = session.get(
        "https://duckduckgo.com/i.js",
        params={"q": query, "vqd": match.group(1), "o": "json", "l": "us-en", "p": "1", "s": "0"},
        timeout=20,
    )
    if listing.status_code != 200 or "challenge" in listing.text[:800]:
        raise RuntimeError(f"duckduckgo_challenge status={listing.status_code}")
    listing.raise_for_status()
    payload = listing.json()
    urls: list[str] = []
    for item in payload.get("results", []):
        url = item.get("image") if isinstance(item, dict) else None
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def urls_from_commons(payload: dict, limit: int = 8) -> list[str]:
    """Image URLs from a Wikimedia Commons generator=search response."""
    pages = (payload.get("query") or {}).get("pages") or {}
    urls: list[str] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [None])[0] if isinstance(page, dict) else None
        if not isinstance(info, dict):
            continue
        if str(info.get("mime", "")) not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        url = info.get("thumburl") or info.get("url")
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


_QUERY_SKIP = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "and",
    "near",
    "for",
    "used",
    "to",
    "from",
    "around",
    "forming",
    "made",
    "featuring",
    "including",
    "ring",
}


_OBJECT_NOUNS = (
    "barn",
    "silo",
    "tractor",
    "fence",
    "bush",
    "tree",
    "plant",
    "grass",
    "crate",
    "trough",
    "hay",
    "wheat",
    "post",
)
_MODIFIERS = {"wooden", "metal", "green", "red", "round", "rounded", "farm"}


def search_query(prompt: str) -> str:
    """Keep a few object words. A full sentence returns no Commons pages."""
    clause = prompt.split(",")[0]
    words = [word for word in clause.split() if word.lower() not in _QUERY_SKIP]
    return " ".join(words[:4])


def _prompt_words(prompt: str) -> list[str]:
    """Every content word, including material words after a comma."""
    words: list[str] = []
    for raw in prompt.lower().replace(",", " ").split():
        word = raw.strip(".").strip()
        if word and word not in _QUERY_SKIP:
            words.append(word)
    return words


def _object_noun(word: str) -> str:
    if word in _OBJECT_NOUNS:
        return word
    if word.endswith("s") and word[:-1] in _OBJECT_NOUNS:
        return word[:-1]
    return ""


def head_noun(prompt: str) -> str:
    """The object word a photo has to depict. Adjectives are not the query."""
    words = _prompt_words(prompt)
    for word in words:
        noun = _object_noun(word)
        if noun:
            return noun
    return words[-1] if words else ""


def object_keyword(prompt: str) -> str:
    """One object, with a material from anywhere in the prompt.

    A tree stays a tree. A fence keeps "wooden" even when that word follows
    a comma. The search does not ask for a second object or for a ring.
    """
    words = _prompt_words(prompt)
    noun = head_noun(prompt)
    if not noun:
        return ""
    materials = [word for word in words if word in _MODIFIERS and word != "farm"]
    if noun == "tree":
        return "deciduous tree"
    if noun == "fence":
        material = "wooden" if "wooden" in words else (materials[0] if materials else "")
        return f"{material} fence segment".strip()
    if noun == "bush":
        material = "green" if "green" in words else (materials[0] if materials else "")
        return f"{material} bush".strip()
    if noun == "tractor":
        return "farm tractor" if "farm" in words else (f"{materials[-1]} tractor" if materials else "tractor")
    if noun in words:
        before = [word for word in words[: words.index(noun)] if word in _MODIFIERS]
        if before:
            return f"{before[-1]} {noun}"
    if materials:
        return f"{materials[0]} {noun}"
    return noun


def search_attempts(prompt: str) -> list[str]:
    """The one object keyword. Leading adjectives are not a separate query."""
    keyword = object_keyword(prompt)
    return [keyword] if keyword else []


def search_phrases(prompt: str) -> list[str]:
    """One foreground object on a background TRELLIS can remove."""
    keyword = object_keyword(prompt)
    if not keyword:
        return []
    return [f"one {keyword}, entire object visible, isolated, plain background"]


def urls_from_bing(page: str, limit: int = 8) -> list[str]:
    """Image URLs from a Bing image-search result page."""
    import html
    import json

    urls: list[str] = []
    for raw in re.findall(r'\sm="([^"]*)"', page):
        try:
            payload = json.loads(html.unescape(raw))
        except json.JSONDecodeError:
            continue
        url = payload.get("murl") if isinstance(payload, dict) else None
        if isinstance(url, str) and url.startswith("http") and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def bing_image_urls(query: str, limit: int = 8) -> list[str]:
    """Photograph URLs from Bing image search.

    Bing's image endpoint is a plain HTTP GET. It does not present the
    DuckDuckGo bot challenge. Each query is the object keyword plus
    "unobstructed and clear view".
    """
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    found: list[str] = []
    for text in search_phrases(query):
        response = requests.get(
            "https://www.bing.com/images/async",
            params={"q": text, "first": 1, "count": max(limit, 10)},
            headers=headers,
            timeout=20,
        )
        body = response.text
        head = body[:1500].lower()
        if response.status_code != 200 or "captcha" in head or "verify you are" in head:
            raise RuntimeError(f"bing_challenge status={response.status_code}")
        for url in urls_from_bing(body, limit):
            if url not in found:
                found.append(url)
        if len(found) >= limit:
            break
    return found[:limit]


def urls_from_openverse(payload: dict, limit: int = 8) -> list[str]:
    """Image URLs from an Openverse ``/v1/images/`` response."""
    urls: list[str] = []
    for item in payload.get("results") or []:
        url = item.get("url") if isinstance(item, dict) else None
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def urls_matching_noun(results, noun: str, limit: int = 8) -> list[str]:
    """Keep photographs whose title or tags name the object."""
    needle = noun.lower()
    urls: list[str] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        if needle and needle not in _result_text(item):
            continue
        url = item.get("url")
        if isinstance(url, str) and url.startswith("http") and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _result_text(item: dict) -> str:
    parts = [str(item.get("title") or "")]
    tags = item.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                parts.append(str(tag.get("name") or ""))
            elif isinstance(tag, str):
                parts.append(tag)
    return " ".join(parts).lower()


def openverse_image_urls(query: str, limit: int = 8) -> list[str]:
    """Return photograph URLs whose title or tags name the object.

    Each attempt keeps the noun. A hit list that never says "silo" is skipped,
    which is how a basket titled "tall cylindrical" stays out of the gate.
    """
    import requests

    noun = head_noun(query)
    headers = {"User-Agent": "farmgen/1.0 (local scene research)"}
    found: list[str] = []
    for text in search_attempts(query):
        response = requests.get(
            "https://api.openverse.org/v1/images/",
            params={"q": text, "page_size": max(limit, 20), "category": "photograph"},
            timeout=20,
            headers=headers,
        )
        response.raise_for_status()
        for url in urls_matching_noun((response.json().get("results") or []), noun, limit):
            if url not in found:
                found.append(url)
        if len(found) >= limit:
            break
    return found[:limit]


def wikimedia_image_urls(query: str, limit: int = 8) -> list[str]:
    """Return photograph URLs from Wikimedia Commons."""
    import requests

    import time

    headers = {"User-Agent": "farmgen/1.0 (local scene research)"}
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {search_query(query)}",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": 1024,
    }
    response = None
    for attempt in range(3):
        response = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=20, headers=headers)
        if response.status_code != 429:
            break
        time.sleep(2 * (attempt + 1))
    response.raise_for_status()
    return urls_from_commons(response.json(), limit)


def crop_box(source: Path, dest: Path, box: tuple[float, float, float, float] | None) -> Path:
    """Cut one plant out of a field photo. A missing or huge box becomes a center crop."""
    from PIL import Image

    image = Image.open(source).convert("RGB")
    width, height = image.size
    left, top, right, bottom = _pixel_box(width, height, box)
    cropped = image.crop((left, top, right, bottom))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dest)
    return dest


def _pixel_box(
    width: int,
    height: int,
    box: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    if box is None or len(box) != 4:
        return _center_box(width, height)
    u0, v0, u1, v1 = (float(value) for value in box)
    if u1 < u0:
        u0, u1 = u1, u0
    if v1 < v0:
        v0, v1 = v1, v0
    area = max(0.0, u1 - u0) * max(0.0, v1 - v0)
    if area < 0.01 or area > 0.5:
        return _center_box(width, height)
    left = int(round(max(0.0, min(1.0, u0)) * width))
    right = int(round(max(0.0, min(1.0, u1)) * width))
    top = int(round(max(0.0, min(1.0, v0)) * height))
    bottom = int(round(max(0.0, min(1.0, v1)) * height))
    if right - left < 8 or bottom - top < 8:
        return _center_box(width, height)
    return left, top, right, bottom


def _center_box(width: int, height: int) -> tuple[int, int, int, int]:
    left = int(round(width * 0.35))
    right = int(round(width * 0.65))
    top = int(round(height * 0.35))
    bottom = int(round(height * 0.65))
    if right - left < 8:
        left, right = 0, width
    if bottom - top < 8:
        top, bottom = 0, height
    return left, top, right, bottom
