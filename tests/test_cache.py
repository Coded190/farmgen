from farmgen.cache import cache_hit, cache_key, cache_path


def test_cache_key_changes_with_prompt_and_size(tmp_path, monkeypatch):
    monkeypatch.setattr("farmgen.cache.asset_cache_dir", lambda: tmp_path)
    barn = cache_key("a red wooden barn", (12.0, 8.0, 7.0))
    other_prompt = cache_key("a blue wooden barn", (12.0, 8.0, 7.0))
    other_size = cache_key("a red wooden barn", (10.0, 8.0, 7.0))
    assert barn != other_prompt
    assert barn != other_size
    assert cache_key("  a red wooden barn  ", (12, 8, 7)) == barn
    assert cache_key("a red wooden barn", (12.0, 8.0, 7.0), image_sha256="abc") != barn
    assert not cache_hit(barn)
    cache_path(barn).write_bytes(b"mesh")
    assert cache_hit(barn)
    assert cache_path(barn).is_file()
