"""Audited CUDA environment and a narrowly scoped, provenance-checked cumm patch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CUMM_REVISION = "b8d83b6511b70746d04485d0d2ec00896b735430"
CUMM_ORIGINAL_SHA256 = "351102fb388061cb57849bc39628167bcb5f376080161c2e3f302dcb3e2ff2e0"
CUMM_ARCH_ANCHOR = "        '8.6', '8.9', '9.0'\n"
CUMM_ARCH_PATCH = "        '8.6', '8.9', '9.0', '12.0'\n"
CUMM_TOOLKIT_ANCHOR = '            linux_cuda_root = Path("/usr/local/cuda")\n'
CUMM_TOOLKIT_PATCH = '            linux_cuda_root = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")\n'
CUMM_PATCH_ID = "cumm-0.7.11-sm120-cuda-home-v2"
NATIVE_POLICY_REVISION = 4


def source_build_environment(source, environment):
    """Prevent an installed previous package from shadowing its pinned source."""
    result = dict(environment)
    inherited = result.get("PYTHONPATH", "")
    result["PYTHONPATH"] = str(source) + (os.pathsep + inherited if inherited else "")
    return result


def prepare_native_paths(root, native):
    """Create a private ccimport job limiter; never replace the system ninja."""
    root = Path(root)
    directory = root / ".build/native-bin"
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = directory / "ninja"
    wrapper.write_text('#!/bin/sh\nexec "$PARTLLM_NINJA_EXE" -j "$MAX_JOBS" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    native.update(ninja_directory=str(directory), ninja_executable=str(root / "venv/bin/ninja"))


def verify_cumm_headers(source):
    includes = Path(source) / "include"
    for sentinel in ["tensorview/gemm/core/params.h", "tensorview/core/all.h"]:
        if not (includes / sentinel).is_file():
            raise RuntimeError(f"Pinned cumm source is missing {sentinel}")
    return includes


def preserve_old_build(source, signature):
    """Do not reuse objects produced by an earlier toolkit/native policy."""
    from uuid import uuid4
    source = Path(source)
    marker = source / ".modly-build-policy.json"
    previous = json.loads(marker.read_text()) if marker.is_file() else None
    build = source / "build"
    # A PyG-only cache-policy change cannot change cumm/spconv object code.
    object_signature = lambda value: {key: item for key, item in value.items() if key != "pyg_source_build_policy"} if value else None
    if object_signature(previous) != object_signature(signature) and build.exists():
        destination = source / f"build.before-policy-{uuid4().hex[:8]}"
        build.rename(destination)
        print(f"[PartLLM] Preserved previous native objects at {destination}", flush=True)
    marker.write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")


def native_environment(native, base=None):
    """Use the same audited toolkit for builds, native probes and inference/NVRTC."""
    environment = dict(os.environ if base is None else base)
    toolkit = Path(native["cuda_home"])
    target = native["architecture_target"]
    sm = int(native["gpu_sm"])
    environment.update(CUDA_HOME=str(toolkit),
                       PARTLLM_NATIVE_ENVIRONMENT="1",
                       PATH=str(toolkit / "bin") + os.pathsep + environment.get("PATH", ""),
                       TORCH_CUDA_ARCH_LIST=target, CUMM_CUDA_ARCH_LIST=target,
                       CUMM_DISABLE_JIT="1", SPCONV_DISABLE_JIT="1",
                       FLASH_ATTN_CUDA_ARCHS="120" if sm >= 120 else str(sm),
                       MAX_JOBS=str(native.get("max_jobs", 2)), NVCC_THREADS="1",
                       FLASH_ATTENTION_FORCE_BUILD="TRUE")
    if native.get("cumm_include_path"):
        environment["CUMM_INCLUDE_PATH"] = native["cumm_include_path"]
    if native.get("ninja_directory"):
        environment["PATH"] = native["ninja_directory"] + os.pathsep + environment["PATH"]
        environment["PARTLLM_NINJA_EXE"] = native["ninja_executable"]
    libraries = [toolkit / "lib64", toolkit / "lib",
                 toolkit / "targets/aarch64-linux/lib", toolkit / "targets/x86_64-linux/lib"]
    values = [str(path) for path in libraries if path.is_dir()]
    if environment.get("LD_LIBRARY_PATH"):
        values.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(values)
    return environment


def patch_cumm_blackwell(source):
    """Add SM120 and honor the selected CUDA_HOME in pinned cumm0.7.11.

    The existing multi-digit arch parser already emits compute_120/SM120 and
    +PTX flags correctly. The Linux fallback's include/lib64 use the same toolkit.
    No kernel algorithm or generated architecture is changed.
    This patch does not assert that those kernels execute correctly on GB10.
    """
    source = Path(source)
    revision = source / ".modly-source-revision"
    if not revision.is_file() or revision.read_text().strip() != CUMM_REVISION:
        raise RuntimeError("Refusing cumm arch patch: pinned source revision is missing or different")
    common = source / "cumm/common.py"
    original = common.read_bytes()
    text = original.decode("utf-8")
    restored = text.replace(CUMM_ARCH_PATCH, CUMM_ARCH_ANCHOR, 1).replace(CUMM_TOOLKIT_PATCH, CUMM_TOOLKIT_ANCHOR, 1)
    if hashlib.sha256(restored.encode()).hexdigest() != CUMM_ORIGINAL_SHA256:
        raise RuntimeError("Refusing cumm arch patch: source checksum differs or unrelated source modifications detected")
    if restored.count(CUMM_ARCH_ANCHOR) != 1 or restored.count(CUMM_TOOLKIT_ANCHOR) != 1:
        raise RuntimeError("Refusing cumm native patch: expected source anchors changed")
    old_patched = restored.replace(CUMM_ARCH_ANCHOR, CUMM_ARCH_PATCH, 1).encode()
    old_marker = source / ".modly-cumm-sm120-patch.json"
    old_provenance = dict(patch="cumm-0.7.11-sm120-whitelist-v1", source_revision=CUMM_REVISION,
                          original_sha256=CUMM_ORIGINAL_SHA256, patched_sha256=hashlib.sha256(old_patched).hexdigest())
    if old_marker.is_file() and json.loads(old_marker.read_text()) != old_provenance:
        raise RuntimeError("Refusing cumm native patch: existing patch provenance differs")
    patched = old_patched.decode().replace(CUMM_TOOLKIT_ANCHOR, CUMM_TOOLKIT_PATCH, 1).encode()
    provenance = dict(patch=CUMM_PATCH_ID, source_revision=CUMM_REVISION,
                      original_sha256=CUMM_ORIGINAL_SHA256,
                      patched_sha256=hashlib.sha256(patched).hexdigest())
    marker = source / ".modly-cumm-native-patch.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) != provenance:
        raise RuntimeError("Refusing cumm arch patch: existing patch provenance differs")
    if original != patched:
        temporary = common.with_suffix(".py.modly-patch-tmp")
        temporary.write_bytes(patched)
        temporary.replace(common)
    marker.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance
