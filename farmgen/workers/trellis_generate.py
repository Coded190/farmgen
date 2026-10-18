"""Run one TRELLIS.2 image-to-3D inside the TRELLIS virtualenv.

FarmGen launches this file with ``third_party/TRELLIS.2/.venv/bin/python``.
It does not import the FarmGen package. The parent scales the mesh after this
process exits, so the GPU is free before the CPU scale step.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
# Weights are already on disk. An unauthenticated Hub check prints a warning
# once per asset and does not download anything.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Free bytes on the visible GPU, from torch.cuda.mem_get_info. This is the
# same 4096 MiB headroom the scheduler uses before it starts a job. Host RAM
# is not part of the check.
FILL_FREE_MIN_MIB = 4096


def fill_device(free_mib: int, need_mib: int = FILL_FREE_MIN_MIB) -> str:
    """Return ``gpu`` when free GPU memory is at least ``need_mib``, else ``cpu``."""
    return "gpu" if int(free_mib) >= int(need_mib) else "cpu"


def _gpu_mib(torch_module) -> tuple[int, int, int]:
    free, total = torch_module.cuda.mem_get_info()
    allocated = torch_module.cuda.memory_allocated()
    return int(free // 1048576), int(total // 1048576), int(allocated // 1048576)


def install_fill_guard(cumesh_module, torch_module) -> dict:
    """Try CuMesh hole fill when the GPU has free memory. On a miss, leave the mesh open.

    The parent process fan-fills pinholes only when this report says ``cpu``.
    ``gpu_free_mib``, ``gpu_total_mib``, and ``gpu_allocated_mib`` are CUDA
    figures for the visible card. On ``reason=cuda_oom`` they are the figures
    at the out-of-memory error, before the cache is emptied.
    """
    report = {
        "fill": "gpu",
        "reason": "ok",
        "gpu_free_mib": None,
        "gpu_total_mib": None,
        "gpu_allocated_mib": None,
        "need_mib": FILL_FREE_MIN_MIB,
        "memory": "gpu",
    }
    original = cumesh_module.CuMesh.fill_holes
    oom_type = getattr(torch_module.cuda, "OutOfMemoryError", torch_module.OutOfMemoryError)

    def _remember(reason: str, free: int, total: int, allocated: int) -> None:
        report["fill"] = "cpu"
        report["reason"] = reason
        report["gpu_free_mib"] = free
        report["gpu_total_mib"] = total
        report["gpu_allocated_mib"] = allocated
        print(
            "FARMGEN stage=asset fill_holes=cpu"
            f" reason={reason} memory=gpu gpu_free_mib={free} gpu_total_mib={total}"
            f" gpu_allocated_mib={allocated} need_mib={FILL_FREE_MIN_MIB}",
            flush=True,
        )

    def _guarded(self, max_hole_perimeter=3e-2):
        free, total, allocated = _gpu_mib(torch_module)
        if fill_device(free, FILL_FREE_MIN_MIB) == "cpu":
            _remember("low_free", free, total, allocated)
            return None
        try:
            filled = original(self, max_hole_perimeter)
        except oom_type:
            free, total, allocated = _gpu_mib(torch_module)
            _remember("cuda_oom", free, total, allocated)
            torch_module.cuda.empty_cache()
            return None
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            free, total, allocated = _gpu_mib(torch_module)
            _remember("cuda_oom", free, total, allocated)
            torch_module.cuda.empty_cache()
            return None
        if report["fill"] == "gpu" and report["gpu_free_mib"] is None:
            report["gpu_free_mib"] = free
            report["gpu_total_mib"] = total
            report["gpu_allocated_mib"] = allocated
            print(
                "FARMGEN stage=asset fill_holes=gpu memory=gpu"
                f" gpu_free_mib={free} gpu_total_mib={total} gpu_allocated_mib={allocated}"
                f" need_mib={FILL_FREE_MIN_MIB}",
                flush=True,
            )
        return filled

    cumesh_module.CuMesh.fill_holes = _guarded
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resolution", default="512")
    args = parser.parse_args()
    if args.resolution != "512":
        raise SystemExit(f"FarmGen only runs pipeline_type 512, got {args.resolution}")

    import importlib

    import torch
    from PIL import Image
    import trellis2.modules.image_feature_extractor as image_features
    # rembg/__init__.py star-imports the class, which hides the submodule.
    birefnet_module = importlib.import_module("trellis2.pipelines.rembg.BiRefNet")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
    import cumesh

    fill_report = install_fill_guard(cumesh, torch)

    # facebook/dinov3-vitl16-pretrain-lvd1689m is a gated repo. This public
    # copy is the same DINOv3ViTModel (hidden size 1024) and does not need a token.
    gated = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    public = "404-Gen/dinov3-vitl16-pretrain-lvd1689m"
    original_init = image_features.DinoV3FeatureExtractor.__init__

    def _public_dino(self, model_name, image_size=512):
        if model_name == gated:
            print(f"FARMGEN stage=asset dino={public} reason=facebook_dinov3_gated", flush=True)
            model_name = public
        original_init(self, model_name, image_size=image_size)

    image_features.DinoV3FeatureExtractor.__init__ = _public_dino

    def _extract_features(self, image):
        import torch.nn.functional as functional

        # transformers 5 moved the encoder under DINOv3ViTModel.model.
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        layers = getattr(self.model, "layer", None)
        if layers is None:
            layers = self.model.model.layer
        for layer_module in layers:
            hidden_states = layer_module(hidden_states, position_embeddings=position_embeddings)
        return functional.layer_norm(hidden_states, hidden_states.shape[-1:])

    image_features.DinoV3FeatureExtractor.extract_features = _extract_features

    # briaai/RMBG-2.0 is gated. ZhengPeng7/BiRefNet is the public model this
    # class was written for, and it returns the same RGBA image.
    gated_rmbg = "briaai/RMBG-2.0"
    public_rmbg = "ZhengPeng7/BiRefNet"
    original_rmbg = birefnet_module.BiRefNet.__init__

    def _public_rmbg(self, model_name: str = public_rmbg):
        if model_name == gated_rmbg:
            print(f"FARMGEN stage=asset rembg={public_rmbg} reason=bria_rmbg_gated", flush=True)
            model_name = public_rmbg
        original_rmbg(self, model_name=model_name)
        # The public checkpoint loads in fp16 while the preprocess tensor is fp32.
        self.model.float()

    birefnet_module.BiRefNet.__init__ = _public_rmbg

    started = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    image = Image.open(args.image)
    mesh = pipeline.run(image, seed=args.seed, pipeline_type="512")[0]
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=2048,
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(args.output)
    marker = {key: value for key, value in fill_report.items() if key != "logged"}
    Path(str(args.output) + ".fill.json").write_text(json.dumps(marker), encoding="utf-8")
    elapsed = time.perf_counter() - started
    del pipeline
    torch.cuda.empty_cache()
    print(json.dumps({"output": args.output, "seconds": elapsed, "torch": torch.__version__}))


if __name__ == "__main__":
    main()
