#!/usr/bin/env python3
"""Modly installer; only standard-library imports before creating the private venv."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Modly executes this file with python -c/runpy, often from an unrelated cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from partllm_config import STATE_NAME, UPSTREAM_REVISION, WEIGHTS_REVISION, resolve_models_dir
from native_runtime import (CUMM_REVISION, CUMM_PATCH_ID, NATIVE_POLICY_REVISION, native_environment,
                            patch_cumm_blackwell, prepare_native_paths, verify_cumm_headers, preserve_old_build)
from native_runtime import source_build_environment

FLASH_REVISION = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
SPCONV_REVISION = "263d6b47425ef843c82f997b12d8b714013d216c"


def run(command, *, env=None, cwd=None, capture=False, timeout=None):
    # Do not print JSON payloads, environment variables, or authenticated URLs.
    print("[PartLLM]", " ".join(str(x) for x in command), flush=True)
    return subprocess.run([str(x) for x in command], check=True, env=env, cwd=cwd,
                          text=True, stdout=subprocess.PIPE if capture else None, timeout=timeout)


def parse_args(argv):
    if len(argv) == 2:
        payload = json.loads(argv[1])
        if not isinstance(payload, dict):
            raise ValueError("Modly setup argument must be a JSON object")
    elif len(argv) >= 4:
        payload = dict(python_exe=argv[1], ext_dir=argv[2], gpu_sm=int(argv[3]),
                       cuda_version=argv[4] if len(argv) >= 5 else 0)
    else:
        raise ValueError("Expected one Modly JSON setup argument or python_exe ext_dir gpu_sm [cuda_version]")
    if not payload.get("python_exe") or not payload.get("ext_dir"):
        raise ValueError("Setup requires python_exe and ext_dir")
    return payload


def normalize_cuda(value):
    text = str(value).lower().removeprefix("cu").strip()
    if "." in text:
        major, minor = text.split(".", 1)
        return int(major), int(minor)
    number = int(text or 0)
    return number // 10, number % 10


def inspect_python(python):
    code = "import json,sys,sysconfig,platform;print(json.dumps(dict(version=list(sys.version_info[:2]),soabi=sysconfig.get_config_var('SOABI'),machine=platform.machine(),system=platform.system())))"
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def venv_python(root):
    return root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def torch_versions(info):
    # cu128 has no torch2.8 CUDA wheel for ARM64; PyPI's ARM2.8 wheel is CPU-only.
    if info["machine"].lower() in {"aarch64", "arm64"}:
        return "2.9.1", "0.24.1"
    return "2.8.0", "0.23.0"


def pyg_source_command(pip, package):
    # Pip's source-wheel cache does not encode Torch/CUDA ABI or GPU targets.
    return pip + ["--no-build-isolation", "--no-deps", "--no-cache-dir", "--force-reinstall",
                  "--no-binary=" + package.split("==")[0], package]


def pyg_source_environment(environment):
    # Modly's launcher strips --no-cache-dir; the child-only pip env survives it.
    return dict(environment, FORCE_CUDA="1", PIP_NO_CACHE_DIR="1")


def reusable_native_build(python, root, signature, environment):
    """A matching signature is only a candidate: damaged binaries must heal."""
    state = root / ".native-build.json"
    if not state.is_file():
        return False
    try:
        previous = json.loads(state.read_text(encoding="utf-8"))
        if previous != signature:
            return False
        run([python, root / "scripts/native_probe.py", "--vendor", root / "vendor/PartLLM",
             "--output", root / "native-diagnostic.json"], cwd=root / "vendor/PartLLM",
            env=environment, timeout=180)
    except (OSError, ValueError, subprocess.SubprocessError):
        from uuid import uuid4
        state.rename(root / f".native-build.failed-{uuid4().hex[:8]}.json")
        print("[PartLLM] Saved failed native signature; Repair will rebuild required native packages", flush=True)
        return False
    print("[PartLLM] Reusing pinned native build after real CUDA/import health check", flush=True)
    return True


def ensure_venv(root, host_python):
    expected = inspect_python(host_python)
    if expected["version"] not in [[3, 11], [3, 12]]:
        raise RuntimeError("PartLLM requires CPython 3.11 or 3.12")
    if expected["system"] != "Linux":
        raise RuntimeError("PartLLM currently requires Linux CUDA native kernels. Windows, macOS and ROCm are unqualified.")
    target = venv_python(root)
    if target.is_file():
        try:
            actual = inspect_python(target)
        except (OSError, subprocess.CalledProcessError, ValueError):
            actual = None
        if actual != expected:
            # Preserve incompatible environments for recovery rather than deleting user files.
            from uuid import uuid4
            backup = root / f"venv.incompatible-{uuid4().hex[:8]}"
            (root / "venv").rename(backup)
            print(f"[PartLLM] Preserved incompatible venv as {backup.name}", flush=True)
    if not target.is_file():
        run([host_python, "-m", "venv", str(root / "venv")])
    run([target, "-m", "pip", "install", "pip==25.3", "setuptools==80.9.0", "wheel==0.45.1"])
    return target, expected


def checkout_source(destination, repo, revision, *, submodules=False):
    marker = destination / ".modly-source-revision"
    if marker.is_file() and marker.read_text().strip() == revision:
        return
    if destination.exists():
        raise RuntimeError(f"Existing source at {destination} differs from the pinned revision; preserve it and move it aside before Repair")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partllm-source-", dir=destination.parent))
    try:
        run(["git", "init", str(staging)])
        run(["git", "-C", staging, "remote", "add", "origin", repo])
        run(["git", "-C", staging, "fetch", "--depth", "1", "origin", revision])
        run(["git", "-C", staging, "checkout", "--detach", "FETCH_HEAD"])
        actual = subprocess.check_output(["git", "-C", str(staging), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError("Runtime source revision mismatch")
        if submodules:
            run(["git", "-C", staging, "submodule", "update", "--init", "--recursive", "--depth", "1"])
        (staging / ".modly-source-revision").write_text(revision + "\n", encoding="utf-8")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def select_toolkit(payload):
    """The driver-derived cuda_version is diagnostic, never a substitute for nvcc."""
    candidates = [payload.get("cuda_home"), os.environ.get("CUDA_HOME"), "/usr/local/cuda-12.8"]
    found = shutil.which("nvcc")
    if found:
        candidates.append(str(Path(found).resolve().parent.parent))
    for value in candidates:
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        nvcc = root / "bin/nvcc"
        if not nvcc.is_file():
            continue
        output = subprocess.check_output([str(nvcc), "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match and tuple(map(int, match.groups())) == (12, 8):
            return root
    raise RuntimeError("CUDA toolkit 12.8 (nvcc, headers, compiler) is required for native builds. Driver CUDA capability alone is insufficient. Set CUDA_HOME to the installed 12.8 toolkit.")


def install_native(python, root, info, payload):
    toolkit = select_toolkit(payload)
    sm = int(payload.get("gpu_sm") or 0)
    if not sm:
        sm = int(subprocess.check_output([str(python), "-c", "import torch;a,b=torch.cuda.get_device_capability();print(a*10+b)"], text=True).strip())
    if sm < 80:
        raise RuntimeError("Upstream FlashAttention requires an Ampere-or-newer NVIDIA GPU")
    # PyG/cumm receive SM120+PTX; FA2's pinned build emits SM120 cubins only.
    target = "12.0+PTX" if sm >= 120 else f"{sm // 10}.{sm % 10}+PTX"
    signature = dict(cuda_home=str(toolkit), architecture_target=target, gpu_sm=sm,
                     interpreter=info, torch=torch_versions(info)[0], flash=FLASH_REVISION,
                     cumm=CUMM_REVISION, spconv=SPCONV_REVISION,
                     native_policy_revision=NATIVE_POLICY_REVISION,
                     cumm_native_patch=CUMM_PATCH_ID,
                     flash_code_target="sm120-cubin" if sm >= 120 else f"sm{sm}-cubin",
                     max_jobs=int(payload.get("max_jobs", 2)))
    if not 1 <= signature["max_jobs"] <= 64:
        raise ValueError("max_jobs must be between 1 and 64")
    prepare_native_paths(root, signature)
    is_arm = info["machine"].lower() in {"aarch64", "arm64"}
    if is_arm:
        signature["cumm_include_path"] = str(root / ".build/cumm/include")
        signature["pyg_source_build_policy"] = "no-global-wheel-cache-v1"
    else:
        signature["cumm_native_patch"] = None
    environment = native_environment(signature)
    if reusable_native_build(python, root, signature, environment):
        return signature
    pip = [python, "-m", "pip", "install"]
    run(pip + ["ninja==1.13.0", "pccm==0.4.16", "ccimport==0.4.4", "fire==0.7.1"], env=environment)
    if is_arm:
        for package, repo, revision in [
            ("cumm", "https://github.com/FindDefinition/cumm.git", CUMM_REVISION),
            ("spconv", "https://github.com/traveller59/spconv.git", SPCONV_REVISION),
        ]:
            source = root / ".build" / package
            checkout_source(source, repo, revision)
            if package == "cumm":
                patch_cumm_blackwell(source)
                verify_cumm_headers(source)
            preserve_old_build(source, signature)
            run(pip + ["--no-build-isolation", "--no-deps", str(source)],
                env=source_build_environment(source, environment))
            if package == "cumm":
                run([python, "-c", "from cumm.core_cc import tensorview_bind; from cumm import tensorview; print('Installed cumm tensorview binding ABI verified')"],
                    env=environment)
        for package in ["torch-scatter==2.1.2", "torch-cluster==1.6.3"]:
            run(pyg_source_command(pip, package), env=pyg_source_environment(environment))
    elif info["machine"].lower() in {"x86_64", "amd64"}:
        run(pip + ["spconv-cu126==2.3.8", "cumm-cu126==0.7.11"], env=environment)
        run(pip + ["--only-binary=:all:", "--no-index", "--find-links", "https://data.pyg.org/whl/torch-2.8.0+cu128.html",
                   "torch-scatter==2.1.2+pt28cu128", "torch-cluster==1.6.3+pt28cu128"], env=environment)
    else:
        raise RuntimeError("Only Linux x86_64 and ARM64 CUDA installation lanes are implemented")
    flash_source = root / ".build/flash-attention"
    checkout_source(flash_source, "https://github.com/Dao-AILab/flash-attention.git", FLASH_REVISION, submodules=True)
    run(pip + ["--no-build-isolation", "--no-deps", str(flash_source)], env=environment)
    return signature


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    payload = parse_args(argv or sys.argv)
    root = Path(payload["ext_dir"]).expanduser().resolve()
    if root != Path(__file__).resolve().parent:
        raise RuntimeError("ext_dir must point to the setup.py installation being executed")
    accelerator = str(payload.get("accelerator", payload.get("torch_flavor", "cuda"))).lower()
    if accelerator != "cuda" or payload.get("torch_flavor", "cuda") != "cuda":
        raise RuntimeError("PartLLM upstream requires NVIDIA CUDA; CPU/MPS/ROCm are unsupported")
    # Consume the host's normalized signals, but intentionally select one audited CUDA wheel stack.
    normalize_cuda(payload.get("cuda_version", 128))
    if payload.get("gfx_target"):
        raise RuntimeError("ROCm gfx_target is incompatible with this CUDA-only runtime")
    index = payload.get("torch_index_url") or "https://download.pytorch.org/whl/cu128"
    if index.rstrip("/") != "https://download.pytorch.org/whl/cu128":
        raise RuntimeError("PartLLM's pinned torch2.8 lane requires the official cu128 PyTorch index")
    models = resolve_models_dir(payload, root)
    state_file = root / STATE_NAME
    if state_file.exists():
        old = json.loads(state_file.read_text(encoding="utf-8"))
        old["ready"] = False
        atomic_json(state_file, old)
    python, info = ensure_venv(root, Path(payload["python_exe"]).expanduser().resolve())
    torch_version, vision_version = torch_versions(info)
    run([python, "-m", "pip", "install", "--index-url", index,
         f"torch=={torch_version}", f"torchvision=={vision_version}"])
    run([python, "-m", "pip", "install", "-r", root / "requirements-runtime.txt"])
    # Exact known ARM/cp311/cp312 Torch2.9.1 metadata defect; never suppress pip check.
    if info["machine"].lower() in {"aarch64", "arm64"} and info["version"] in [[3, 11], [3, 12]]:
        run([python, root / "cusparselt_metadata.py", "--lane", "cu128"])
    vendor = root / "vendor/PartLLM"
    checkout_source(vendor, "https://github.com/czvvd/PartLLM.git", UPSTREAM_REVISION)
    run([python, vendor / "scripts/patch_transformers.py"], cwd=vendor)
    run([python, vendor / "scripts/patch_transformers.py", "--check"], cwd=vendor)
    weights = models / "partllm"
    # Explicit exception for this PROCESS: the host supplies no model_sources UI.
    run([python, root / "weights.py", weights])
    native = install_native(python, root, info, payload)
    report_path = root / "native-diagnostic.json"
    run([python, root / "scripts/native_probe.py", "--vendor", vendor, "--output", report_path],
        cwd=vendor, env=native_environment(native))
    run([python, "-m", "pip", "check"])
    atomic_json(root / ".native-build.json", native)
    atomic_json(state_file, dict(schema_version=1, ready=True, extension_dir=str(root), models_dir=str(models),
                                weights_dir=str(weights), upstream_revision=UPSTREAM_REVISION,
                                weights_revision=WEIGHTS_REVISION, interpreter=info, native=native))
    print("[PartLLM] Setup complete: checkpoint verified under configured models_dir; native CUDA kernels/imports passed. Model inference is a separate qualification step.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[PartLLM] SETUP FAILED: {error}\nRepair after correcting the reported dependency/toolkit issue. Existing weights and source are preserved.", file=sys.stderr, flush=True)
        raise SystemExit(1)
