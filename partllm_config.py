"""Lightweight shared contract: importing this module never imports CUDA packages."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

UPSTREAM_REVISION = "524f4f68d2501bd9177e9f36f4f2ce4be090ab1b"
WEIGHTS_REVISION = "0c546a88ca52cced296200e173c8961b8f2ca0ba"
WEIGHTS_REPO = "Czvvd/PartLLM"
EXTENSION_ID = "modly-partllm-extension"
STATE_NAME = ".partllm-setup.json"


def option(name, default, choices, **kw):
    return dict(id=name, label=name.replace("_", " ").title(), type="select", default=default,
                options=[dict(value=x, label=x.title()) for x in choices], **kw)


def number(name, default, low, high, kind="int"):
    return dict(id=name, label=name.replace("_", " ").title(), type=kind,
                default=default, min=low, max=high, step=1 if kind == "int" else 0.001)


COMMON_PARAMS = [
    number("seed", 42, 0, 2147483647), number("sample_seed", 42, 0, 2147483647),
    number("max_new_tokens", 10240, 1, 32768),
    option("do_sample", "false", ["false", "true"]),
    number("temperature", 0.7, 0.001, 10.0, "float"),
    number("top_p", 0.8, 0.001, 1.0, "float"), number("top_k", 20, 0, 100000),
    option("auto_add_bg", "true", ["true", "false"]),
    number("max_generated_parts", 64, 0, 1024),
    option("save_per_part", "false", ["false", "true"]),
    option("postprocess", "true", ["true", "false"]),
    option("pp_drop_small", "true", ["true", "false"]),
    number("pp_rel_area_threshold", 0.001, 0.0, 1.0, "float"),
    option("graph_cut", "true", ["true", "false"]),
    option("apply_z_positive", "false", ["false", "true"]),
]
FULL_PARAMS = [
    option("sem_mode", "semantic", ["semantic", "nosem", "all"]),
    option("prompt_set", "all", ["all", "coarse", "general", "fine"]),
    number("num_parts", 0, 0, 1024),
    option("prompt_style", "current", ["current", "legacy"]),
    option("primary_output", "general", ["general", "coarse", "fine"]),
] + COMMON_PARAMS
TEXT_PARAMS = [dict(id="part_names", label="Part Names", type="string", default="seat; chair back; leg",
                    tooltip="Separate part names with semicolons or newlines. Connected text overrides this field.")] + COMMON_PARAMS
SCHEMAS = {"full-shape": FULL_PARAMS, "text-guided": TEXT_PARAMS}


def normalize_params(node_id, values):
    if node_id not in SCHEMAS:
        raise ValueError(f"Unsupported nodeId: {node_id}")
    if not isinstance(values, dict):
        raise ValueError("params must be an object")
    result = {}
    for spec in SCHEMAS[node_id]:
        key = spec["id"]
        value = values.get(key, spec["default"])
        if spec["type"] == "select":
            value = str(value).lower() if isinstance(value, bool) else str(value)
            if value not in [x["value"] for x in spec["options"]]:
                raise ValueError(f"Invalid {key}")
        elif spec["type"] in {"float", "int"}:
            if isinstance(value, bool):
                raise ValueError(f"Invalid {key}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {key}") from exc
            if not math.isfinite(numeric) or not spec["min"] <= numeric <= spec["max"]:
                raise ValueError(f"{key} is outside the allowed range")
            if spec["type"] == "int" and not numeric.is_integer():
                raise ValueError(f"{key} must be an integer")
            value = int(numeric) if spec["type"] == "int" else numeric
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be text")
        result[key] = value
    return result


def settings_candidates():
    if os.environ.get("MODLY_SETTINGS_PATH"):
        yield Path(os.environ["MODLY_SETTINGS_PATH"])
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    yield config / "Modly" / "settings.json"
    yield config / "modly" / "settings.json"
    if os.environ.get("APPDATA"):
        yield Path(os.environ["APPDATA"]) / "Modly" / "settings.json"
    yield Path.home() / "Library" / "Application Support" / "Modly" / "settings.json"


def resolve_models_dir(payload, extension_dir):
    """Do not guess models from the extension's parent or expose token-bearing settings."""
    explicit = payload.get("models_dir") or payload.get("modelsDir") or os.environ.get("MODELS_DIR") or os.environ.get("MODLY_MODELS_DIR")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise ValueError("models_dir must be absolute")
        return path.resolve()
    for candidate in settings_candidates():
        if not candidate.is_file():
            continue
        try:
            settings = json.loads(candidate.read_text(encoding="utf-8"))
            extensions = Path(settings["extensionsDir"]).expanduser().resolve()
            models = Path(settings["modelsDir"]).expanduser()
        except (KeyError, ValueError, TypeError):
            continue
        if extension_dir.resolve().parent == extensions and models.is_absolute():
            return models.resolve()
    raise ValueError("Cannot resolve this Modly installation's models_dir. Use Repair after configuring Models/Extensions paths, or pass models_dir explicitly.")


def load_state(extension_dir):
    path = extension_dir / STATE_NAME
    if not path.is_file():
        raise RuntimeError("PartLLM setup has not completed. Run extension Repair.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("extension_dir") != str(extension_dir.resolve()) or state.get("upstream_revision") != UPSTREAM_REVISION or state.get("weights_revision") != WEIGHTS_REVISION:
        raise RuntimeError("PartLLM setup belongs to a different installation/revision. Run Repair.")
    if state.get("ready") is not True:
        raise RuntimeError("PartLLM native setup is incomplete. Run Repair and inspect its native diagnostic.")
    return state
