"""Explicit setup-only checkpoint provisioning under Modly's configured models_dir."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from partllm_config import WEIGHTS_REPO, WEIGHTS_REVISION

REQUIRED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "preprocessor_config.json", "model.safetensors.index.json"]


def safe_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint contains an unsafe relative path")
    return path


def validate_weights(root, *, hashes=False):
    root = Path(root)
    for name in REQUIRED:
        path = safe_file(root, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"PartLLM weights missing {name}. Run extension Repair.")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise RuntimeError("PartLLM config model_type is not qwen3_vl")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = index.get("weight_map") or {}
    if not mapping or not any("utonia_model" in key for key in mapping):
        raise RuntimeError("PartLLM checkpoint is missing its integrated Utonia encoder")
    for name in set(mapping.values()):
        path = safe_file(root, name)
        if not path.is_file() or path.stat().st_size < 1024:
            raise RuntimeError(f"PartLLM checkpoint shard is incomplete: {name}. Run Repair.")
    inventory_path = root / ".modly-partllm-inventory.json"
    if not inventory_path.is_file():
        raise RuntimeError("PartLLM download inventory missing. Run Repair to verify this checkpoint.")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("revision") != WEIGHTS_REVISION:
        raise RuntimeError("PartLLM checkpoint revision differs from this extension. Run Repair.")
    for item in inventory["files"]:
        path = safe_file(root, item["name"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise RuntimeError(f"PartLLM checkpoint file size mismatch: {item['name']}. Run Repair.")
        if hashes and item.get("sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"PartLLM checkpoint digest mismatch: {item['name']}")
    return inventory


def download_weights(root):
    from huggingface_hub import HfApi, snapshot_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # This pinned checkpoint is public. Ignore unrelated expired global HF login
    # credentials rather than changing the user's account/configuration.
    info = HfApi(token=False).model_info(WEIGHTS_REPO, revision=WEIGHTS_REVISION, files_metadata=True)
    if info.sha != WEIGHTS_REVISION:
        raise RuntimeError("Hugging Face returned an unexpected checkpoint revision")
    files = []
    for sibling in info.siblings:
        if sibling.rfilename.startswith("."):
            continue
        safe_file(root, sibling.rfilename)
        lfs = sibling.lfs
        files.append(dict(name=sibling.rfilename, size=sibling.size,
                          sha256=getattr(lfs, "sha256", None) if lfs else None))
    if not files or any(item["size"] is None for item in files):
        raise RuntimeError("Cannot verify checkpoint file sizes from Hugging Face metadata")
    snapshot_download(WEIGHTS_REPO, revision=WEIGHTS_REVISION, local_dir=str(root), token=False,
                      allow_patterns=[x["name"] for x in files], max_workers=4)
    inventory = dict(repo=WEIGHTS_REPO, revision=WEIGHTS_REVISION, files=files)
    target = root / ".modly-partllm-inventory.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    temporary.replace(target)
    validate_weights(root, hashes=True)


if __name__ == "__main__":
    import sys
    download_weights(Path(sys.argv[1]))
