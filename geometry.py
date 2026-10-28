"""Transform-aware input flattening and geometry-preserving segmented GLB export."""
from __future__ import annotations

import json
import re
from pathlib import Path


def flatten_mesh(source: Path, destination: Path):
    import numpy as np
    import trimesh

    if source.suffix.lower() not in {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}:
        raise ValueError("Supported input formats: GLB, glTF, OBJ, PLY, STL, OFF. Convert FBX to GLB with Blender first.")
    loaded = trimesh.load(str(source), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        meshes = []
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            geometry = loaded.geometry[name]
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
                mesh = geometry.copy()
                mesh.apply_transform(transform)
                meshes.append(mesh)
    else:
        meshes = []
    if not meshes:
        raise ValueError("Input contains no triangle meshes")
    mesh = trimesh.util.concatenate(meshes)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError("Input geometry is empty, non-finite or degenerate")
    # Convert texture samples before concatenating into one canonical primitive.
    if mesh.visual.kind == "texture":
        mesh.visual = mesh.visual.to_color()
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(destination))
    # Reload the actual staged artifact; this is the source face order supplied upstream.
    canonical = trimesh.load(str(destination), force="mesh", process=False)
    bounds = np.asarray(canonical.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return canonical, {"scale": 1.0 / diagonal, "shift": (-(bounds[0] + bounds[1]) / (2 * diagonal)).tolist(),
                       "input_instances": len(meshes), "faces": len(canonical.faces), "vertices": len(canonical.vertices)}


def part_names(metadata):
    # Names precede each SEG token in PartLLM's response; names are documentary,
    # while numerical face labels remain the authoritative partition.
    names = []
    for prefix in metadata.get("response", "").split("<|SEG|>")[:-1]:
        tail = prefix.rsplit("\n", 1)[-1]
        try:
            structured = json.loads(tail)
        except (ValueError, TypeError):
            structured = None
        if isinstance(structured, dict) and isinstance(structured.get("label"), str):
            tail = structured["label"]
        tail = re.sub(r"<[^>]+>|\[[^\]]*\]|\([^)]*\)", "", tail).strip(" :;,\t")
        names.append(tail[-100:] or f"part_{len(names)}")
    return names


def export_partition(canonical, native_dir, destination, transform):
    import numpy as np
    import trimesh

    labels_path = native_dir / "face_labels.npy"
    if not labels_path.is_file() or not (native_dir / "pred_mesh.ply").is_file():
        raise RuntimeError("Upstream produced no face partition/mesh; inspect upstream.log and error.txt")
    labels = np.load(labels_path, allow_pickle=False)
    if labels.ndim != 1 or len(labels) != len(canonical.faces) or labels.dtype.kind not in "iu" or (labels < 0).any():
        raise RuntimeError("Upstream face labels are invalid or topology changed")
    predicted = trimesh.load(native_dir / "pred_mesh.ply", process=False)
    if not isinstance(predicted, trimesh.Trimesh) or len(predicted.faces) != len(canonical.faces):
        raise RuntimeError("Upstream mesh topology changed; refusing an unsafe label remap")
    expected = canonical.vertices * transform["scale"] + transform["shift"]
    if predicted.vertices.shape != expected.shape or not np.allclose(predicted.vertices, expected, atol=2e-5) or not np.array_equal(predicted.faces, canonical.faces):
        raise RuntimeError("Upstream mesh does not match the canonical face/vertex order")
    metadata = json.loads((native_dir / "metadata.json").read_text(encoding="utf-8"))
    names = part_names(metadata)
    scene, parts = trimesh.Scene(), []
    palette = [[230, 25, 75, 255], [60, 180, 75, 255], [67, 99, 216, 255], [245, 130, 49, 255], [145, 30, 180, 255], [66, 212, 244, 255]]
    for value in np.unique(labels):
        index = int(value)
        face_indices = np.flatnonzero(labels == index)
        label = names[index] if index < len(names) else "background" if index == len(names) else f"part_{index}"
        part = canonical.submesh([face_indices], append=True, repair=False)
        part.visual = trimesh.visual.ColorVisuals(mesh=part, vertex_colors=np.tile(palette[index % len(palette)], (len(part.vertices), 1)))
        part.metadata.update(partllm_class=index, partllm_name=label, source_face_count=len(face_indices))
        name = f"{index:03d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', label)[:70]}"
        scene.add_geometry(part, node_name=name, geom_name=name)
        parts.append(dict(class_id=index, name=label, faces=len(face_indices), node=name))
    scene.metadata.update(partllm_parts=parts, coordinate_space="original_input_world_space")
    scene.export(str(destination))
    if destination.stat().st_size < 32:
        raise RuntimeError("Segmented GLB export is empty")
    return dict(parts=parts, native_directory=str(native_dir), native_metadata=metadata)
