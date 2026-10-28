#!/usr/bin/env python3
"""Exercise actual CUDA kernels, patched Qwen/Utonia and post-processing imports."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def probe(vendor):
    import torch
    import torch_cluster
    import torch_scatter
    import flash_attn
    import spconv.pytorch as spconv

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the extension interpreter")
    device = "cuda"
    values = torch.tensor([1., 2., 3., 4.], device=device)
    indices = torch.tensor([0, 0, 1, 1], device=device)
    result = torch_scatter.scatter_add(values, indices)
    torch.testing.assert_close(result, torch.tensor([3., 7.], device=device))
    points = torch.rand((32, 3), device=device)
    selected = torch_cluster.fps(points, ratio=0.25, random_start=False)
    if selected.numel() != 8:
        raise RuntimeError("torch-cluster CUDA FPS returned unexpected shape")
    qkv = torch.randn((16, 3, 2, 32), device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([0, 8, 16], device=device, dtype=torch.int32)
    attention = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, lengths, 8, dropout_p=0.0)
    if not attention.isfinite().all():
        raise RuntimeError("FlashAttention CUDA produced nonfinite output")
    sparse_indices = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]], device=device, dtype=torch.int32)
    sparse = spconv.SparseConvTensor(torch.randn((3, 4), device=device), sparse_indices, [8, 8, 8], 1)
    layer = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False).cuda()
    features = layer(sparse).features
    if features.shape != (3, 8) or not features.isfinite().all():
        raise RuntimeError("spconv CUDA convolution failed")
    torch.cuda.synchronize()
    for module in ["igl", "pymeshlab", "igraph", "ray", "tensordict", "diffusers", "transformers"]:
        importlib.import_module(module)
    sys.path.insert(0, str(vendor / "recipe/part_seg/tools/inference"))
    inference = importlib.import_module("test_mask_pred")
    importlib.import_module("infer_grounding")
    importlib.import_module("infer_promptable")
    importlib.import_module("postprocess")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.utonia.model import SerializedAttention
    if not hasattr(Qwen3VLForConditionalGeneration, "forward") or not callable(inference.load_model):
        raise RuntimeError("Custom Qwen3-VL inference ABI is missing")
    # SerializedAttention invokes the same varlen FlashAttention kernel exercised above.
    if "flash_attn" not in SerializedAttention.forward.__code__.co_names:
        raise RuntimeError("Unexpected Utonia attention implementation")
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                native_cuda_kernels="PASS", upstream_imports="PASS", inference="NOT RUN")


def configure_environment(root):
    """Reuse saved settings only when no explicit setup/runtime environment exists."""
    from native_runtime import native_environment
    if os.environ.get("PARTLLM_NATIVE_ENVIRONMENT") != "1":
        for state_path in [root / ".native-build.json", root / ".partllm-setup.json"]:
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                os.environ.update(native_environment(state.get("native", state)))
                break
    os.environ["PARTLLM_UTONIA_ALLOW_DOWNLOAD"] = "0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    configure_environment(root)
    report = probe(args.vendor.resolve())
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result)
