#!/usr/bin/env python3
"""Create an original synthetic chair fixture; no third-party assets are used."""
import argparse
from pathlib import Path


def create(destination):
    import trimesh
    scene = trimesh.Scene()
    components = [("seat", (1.2, .18, 1.2), (0, 1.0, 0)),
                  ("back", (1.2, 1.2, .15), (0, 1.65, -.52))]
    for x in [-.48, .48]:
        for z in [-.48, .48]:
            components.append((f"leg_{x}_{z}", (.14, .95, .14), (x, .475, z)))
    for name, extents, center in components:
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    print(create(parser.parse_args().output))
