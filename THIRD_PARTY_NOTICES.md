# Third-party notices and weight terms

The MIT license at the repository root applies **only to this Modly wrapper**.
It does not relicense PartLLM, checkpoints, compiled dependencies or bundled
notices. Setup downloads third-party runtime source into `vendor/` and `.build/`;
their original copyright headers, licenses and Git histories are retained.

| Component | Source / pin | Terms and boundary |
| --- | --- | --- |
| PartLLM code | [czvvd/PartLLM](https://github.com/czvvd/PartLLM), `524f4f68d2501bd9177e9f36f4f2ce4be090ab1b` | Apache-2.0, copied in `licenses/PartLLM-Apache-2.0.txt`. Includes Qwen/Hugging Face, ByteDance VERL and Meta/Utonia notices in their original files. |
| PartLLM checkpoint | [Czvvd/PartLLM](https://huggingface.co/Czvvd/PartLLM), `0c546a88ca52cced296200e173c8961b8f2ca0ba` | The inspected repository has no explicit license declaration/model card. **Weight terms remain unresolved.** Do not assume Apache/MIT, unrestricted redistribution or commercial permission. Request clarification from the publisher before checkpoint redistribution or commercial deployment. Publishing this MIT wrapper source does not grant checkpoint rights. |
| FlashAttention | Dao-AILab/flash-attention `060c9188beec3a8b62b33a3bfa6d5d2d44975fab` | Original BSD license and embedded CUTLASS notices are preserved by the setup source checkout. |
| cumm / spconv | FindDefinition/cumm `b8d83b6511b70746d04485d0d2ec00896b735430`; traveller59/spconv `263d6b47425ef843c82f997b12d8b714013d216c` | Original Apache-2.0 notices retained in source checkouts. |
| PyMeshLab / MeshLab | PyPI `pymeshlab==2025.7.post1` | GPL-family software, distributed by its upstream package. Installed in the private environment; not relicensed by this wrapper. Review its license and distribution obligations before redistributing a prebuilt environment. |
| libigl, igraph, PyTorch, Transformers, Diffusers, PyG and other Python packages | Exact top-level pins in `requirements-runtime.txt` and `setup.py` | Each installed distribution retains its own license metadata and notices. No bundled binary environment is shipped by this source repository. |

Modly is the host application by Lightning Pixel. The integration author is
DrHepa; this does not imply authorship or endorsement by the upstream projects.

On the ARM64 Blackwell source lane, setup locally modifies cumm0.7.11's accepted
architecture whitelist to include `12.0` and makes its Linux toolkit fallback
honor the selected `CUDA_HOME`; no algorithms are replaced. The original
Git revision remains intact, and `.modly-cumm-native-patch.json` records the exact
original/patched SHA-256 values. Original copyright/license notices are retained.

The original synthetic chair fixture generator in `scripts/` belongs to this
MIT wrapper and contains no downloaded third-party mesh asset.
