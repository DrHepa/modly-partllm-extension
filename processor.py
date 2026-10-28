#!/usr/bin/env python3
"""Modly PROCESS NDJSON entry. Heavy upstream execution stays in a child process."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from partllm_config import COMMON_PARAMS, load_state, normalize_params
from native_runtime import native_environment

ROOT = Path(__file__).resolve().parent


def emit(kind, **fields):
    if kind == "progress" and "message" in fields:
        fields["label"] = fields.pop("message")
    print(json.dumps(dict(type=kind, **fields), ensure_ascii=False), flush=True)


def build_command(node, params, python, vendor, weights, staged, output, config):
    entry = "test_mask_pred.py" if node == "full-shape" else "infer_grounding.py"
    command = [str(python), str(vendor / "recipe/part_seg/tools/inference" / entry),
               "--config", str(config), "--model_path", str(weights), "--config_path", str(weights),
               "--output_dir", str(output), "--max_samples", "1", "--save_mesh", "true", "--resume", "false"]
    for spec in COMMON_PARAMS:
        key = spec["id"]
        command.extend([f"--{key}", str(params[key])])
    if node == "full-shape":
        if params["prompt_style"] == "legacy" and not params["num_parts"]:
            raise ValueError("Legacy prompt style requires num_parts greater than zero")
        command += ["--glb_dir", str(staged.parent), "--sem_mode", params["sem_mode"],
                    "--prompt_set", params["prompt_set"], "--prompt_style", params["prompt_style"]]
        if params["num_parts"]:
            command += ["--num_parts", str(params["num_parts"])]
    else:
        names = list(dict.fromkeys(x.strip() for x in re.split(r"[;\n]", params["part_names"]) if x.strip()))
        if not names or len(names) > 128 or any(len(x) > 200 for x in names):
            raise ValueError("Provide 1–128 part names separated by semicolons, each at most 200 characters")
        command += ["--mesh_path", str(staged), "--part_names", *names]
    return command


def select_primary(exports, preferred):
    for selection in [preferred, f"nosem_{preferred}"]:
        for result in exports:
            if result["selection"] == selection:
                return result
    return exports[0]


def run(payload):
    if not isinstance(payload, dict):
        raise ValueError("Process payload must be an object")
    inputs = payload.get("input") or {}
    if not isinstance(inputs, dict):
        raise ValueError("input must be an object")
    node = inputs.get("nodeId") or payload.get("nodeId")
    params = normalize_params(node, payload.get("params") or {})
    if node == "text-guided":
        text = inputs.get("text")
        texts = inputs.get("texts")
        if not text and isinstance(texts, dict):
            text = texts.get("input-1")
        if text:
            if not isinstance(text, str):
                raise ValueError("Connected part names must be text")
            params["part_names"] = text
    source = Path(inputs.get("filePath") or "").expanduser()
    if not source.is_absolute() or not source.is_file():
        raise ValueError("An existing absolute mesh input path is required")
    workspace = Path(payload.get("workspaceDir") or "")
    temporary = Path(payload.get("tempDir") or "")
    if not workspace.is_absolute() or not temporary.is_absolute():
        raise ValueError("Modly must supply absolute workspaceDir and tempDir")
    state = load_state(ROOT)
    weights = Path(state["weights_dir"])
    from weights import validate_weights
    validate_weights(weights)
    vendor = ROOT / "vendor/PartLLM"
    if not (vendor / "scripts/infer.sh").is_file():
        raise RuntimeError("PartLLM runtime source is missing. Run Repair.")
    python = ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise RuntimeError("PartLLM extension venv is missing. Run Repair.")
    result_dir = workspace / "Workflows" / f"partllm_{uuid4().hex}"
    result_dir.mkdir(parents=True)
    temporary.mkdir(parents=True, exist_ok=True)
    emit("progress", percent=3, message="Preparing transform-aware mesh input")
    from geometry import flatten_mesh, export_partition
    with tempfile.TemporaryDirectory(prefix="partllm_", dir=temporary) as run_temp:
        staged = Path(run_temp) / "input" / "input.glb"
        canonical, transform = flatten_mesh(source, staged)
        config = Path(run_temp) / "inference.yaml"
        config.write_text("{}\n", encoding="utf-8")
        native = result_dir / "native"
        command = build_command(node, params, python, vendor, weights, staged, native, config)
        environment = native_environment(state["native"])
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                           PARTLLM_UTONIA_ALLOW_DOWNLOAD="0", PYTHONUNBUFFERED="1", USE_VLLM="0",
                           HF_HUB_DISABLE_TELEMETRY="1")
        emit("progress", percent=10, message="Running PartLLM; see upstream.log for native diagnostics")
        log_path = result_dir / "upstream.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=vendor, env=environment, stdout=log, stderr=subprocess.STDOUT)
        errors = list(native.rglob("error.txt")) if native.exists() else []
        if completed.returncode or errors:
            raise RuntimeError(f"PartLLM inference failed (exit {completed.returncode}); inspect {log_path}")
        candidates = sorted(native.rglob("face_labels.npy")) if native.exists() else []
        if not candidates:
            raise RuntimeError(f"PartLLM returned no segmentation; inspect {log_path}")
        emit("progress", percent=90, message="Exporting segmented submeshes in original coordinates")
        exports = []
        for labels in candidates:
            relative = labels.parent.relative_to(native)
            filename = "_".join(relative.parts) + ".glb"
            destination = result_dir / filename
            partition = export_partition(canonical, labels.parent, destination, transform)
            exports.append(dict(file=str(destination), selection=relative.parts[-1], **partition))
        preferred = params.get("primary_output", "general")
        primary = select_primary(exports, preferred)
        report = dict(node=node, source=str(source.resolve()), parameters=params, source_transform=transform,
                      exports=exports, primary_output=primary["file"])
        (result_dir / "partllm.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("progress", percent=100, message="PartLLM segmented mesh ready")
    return dict(filePath=primary["file"])


def main():
    try:
        payload = json.loads(sys.stdin.readline())
        result = run(payload)
    except Exception as error:
        emit("error", message=str(error))
        return 1
    emit("done", **{"result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
