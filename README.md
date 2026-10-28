# PartLLM for Modly

Python **PROCESS** extension integrating [PartLLM](https://github.com/czvvd/PartLLM)
for mesh-to-mesh part segmentation. Integration by **DrHepa**. The native Hugging
Face inference backend is used; no training environment or vLLM worker is started.

> **Qualification:** real setup, CUDA kernels and model inference passed on
> Python3.11.9 (2026-10-28) and Python3.12.3 (2026-10-23), Linux ARM64/GB10,
> through both requested Modly process runners. The installed Python3.12 runtime
> was preserved unchanged; Python3.11 was qualified in isolation under `/tmp`.
> This is not an Electron UI E2E or cross-platform qualification.
> See [VALIDATION.md](VALIDATION.md) for exact evidence and limitations.

## Usage

| Node | Input | Output |
| --- | --- | --- |
| PartLLM Full-Shape Segmentation | Mesh | GLB containing separate named/class-indexed part submeshes |
| PartLLM Text-Guided Segmentation | Mesh + optional text | GLB partition for the requested part names, including background |

Connect one mesh. The second text input accepts semicolon- or newline-separated
names (`seat; chair back; leg`); connected text overrides **Part Names**. The
parameter is also available when no text node is connected.

GLB/glTF, OBJ, PLY, STL and OFF inputs are accepted. GLB is recommended. The adapter
flattens **all scene instances with their world transforms** into a single
canonical GLB before running upstream. This avoids upstream's first-instance-only
scene conversion. FBX is deliberately not advertised: upstream requires an extra
Blender/Assimp system integration for it. Convert FBX to GLB in Blender first.

## Parameters

- Full shape: **Semantic Mode** `semantic` (`nosem`/`all` available), **Prompt Set**
  `all` (`coarse`/`general`/`fine`), **Num Parts** `0` (disabled; positive overrides
  prompt set), **Prompt Style** `current` (`legacy` requires a positive part count).
- **Primary Output** `general` chooses the workflow artifact when several prompt
  results are generated. All results remain saved. If unavailable, the first
  matching nonsemantic granularity is used when only `nosem` outputs exist;
  otherwise the first generated result is returned (for example numeric-only).
- Generation: seed/sample seed `42`, max new tokens `10240`, sampling `false`,
  temperature `0.7`, top-p `0.8`, top-k `20`, auto-add background `true`, maximum
  generated parts `64` (`0` disables that limit).
- Post-processing: enabled `true`, drop small regions `true`, relative area
  threshold `0.001`, graph cut `true`; legacy Utonia Z-positive shift `false`.
- **Save Per Part** `false` controls upstream point-cloud PLY exports. The wrapper
  always saves actual segmented GLBs. These are different outputs.

All exposed controls use upstream-consumed manifest types. Numeric/finite bounds
are rechecked at runtime. Upstream's fixed internal post-processing heuristics
are unchanged. Batch selection/cache directories are not exposed: one Modly
execution processes one canonical mesh, avoiding stale cache/topology mapping.
Resume is disabled for unique runs. `save_mesh` is forced true because a reliable
mesh partition requires upstream's face labels and topology validation.

## Installation

When this repository is published, use **Extensions → Install from GitHub** with
the extension repository URL. Local-folder installation only links the source;
run **Repair** to provision it. The normal supported path is the UI, not manual
terminal package installation.

Setup consumes Modly's JSON setup payload and historical positional arguments,
creates exactly `<extension>/venv`, installs pinned dependencies, downloads the
complete pinned checkpoint and verifies native CUDA operations. Setup returns a
nonzero status if any required native dependency/probe fails.

### Weight location (PROCESS-specific provisioning)

Modly PROCESS extensions have **no model_sources weight UI or injected model_dir**.
For this explicitly requested adapter, **setup downloads weights** into:

```text
<configured models_dir>/partllm/
  config.json
  tokenizer.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  ...complete pinned tokenizer/processor assets...
  .modly-partllm-inventory.json
```

The checkpoint is approximately **9.84 GB**; additional space is required for
PyTorch, native build source/artifacts, pip cache and runtime outputs. Setup asks
Hugging Face for exact sizes/LFS SHA-256 values and verifies them after resumable
download. Runtime checks the recorded revision, inventory sizes and shard index.
Weights are never stored in the extension or a global HF cache. Utonia is already
inside the checkpoint; no separate encoder checkpoint is downloaded.

Models resolution order: explicit `models_dir`/`modelsDir` setup argument, `MODELS_DIR`
or `MODLY_MODELS_DIR`, then the real Modly `settings.json` **only when its
extensionsDir matches this installed extension's parent**. Set `MODLY_SETTINGS_PATH`
for a nonstandard settings location. There is no guessed sibling-model fallback.
Setup does not log settings/token values. Moving models/extensions requires Repair;
readiness is bound to the actual installation root and pinned revisions.
Public checkpoint requests explicitly disable implicit Hugging Face authentication;
an unrelated expired global login cannot block them. No account settings change.

Inference runs with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`PARTLLM_UTONIA_ALLOW_DOWNLOAD=0`. Missing weights fail with a Repair hint rather
than triggering a runtime download.

## Requirements and compatibility

- CPython **3.11 and 3.12** have explicit ABI checks and installation paths. An
  incompatible existing private environment is preserved under a renamed backup.
- Linux x86_64: PyTorch `2.8.0` + torchvision `0.23.0`; Linux ARM64:
  PyTorch `2.9.1` + torchvision `0.24.1`, official CUDA `cu128` index.
  The ARM64 override is required because the official cu128 index has no CUDA
  torch2.8 ARM64 wheel; PyPI's torch2.8 ARM64 build is not an equivalent CUDA lane.
  Native/kernel/model qualification remains separate from wheel availability.
  Both lanes use
  Transformers `4.57.3` plus **the complete upstream custom qwen3_vl folder**.
- CUDA **toolkit 12.8** with nvcc/headers and a working C++ compiler, Git, and
  standard build tools are required. Setup checks actual nvcc, not Modly's
  driver-derived `cuda_version`. Set `CUDA_HOME` or setup `cuda_home` if needed.
- Linux x86_64: pinned PyG torch2.8/cu128 wheels, spconv-cu126 `2.3.8` /
  cumm-cu126 `0.7.11`, source-built FlashAttention `2.8.3`.
- Linux ARM64: pinned source builds for torch-scatter, torch-cluster, cumm,
  spconv and FlashAttention. For GB10/SM121, PyG and cumm/spconv use CUDA12.8's
  **SM120 + compute120 PTX** target. FlashAttention2.8.3's source build with
  `FLASH_ATTN_CUDA_ARCHS=120` emits **SM120 cubins only**, not PTX. These are build
  targets, **not a claim of GB10 execution compatibility**; actual GPU probes
  remain mandatory. No unsupported SM121 nvcc target is requested.
- FlashAttention is required by both custom Qwen and Utonia. There is no pretend
  CPU/SDPA fallback. CPU, MPS, ROCm and Windows are currently unsupported lanes.

Native build defaults are `MAX_JOBS=2`, `NVCC_THREADS=1`. Initial source builds can
be lengthy. Repair reuses a matching pinned native signature only after a real
bounded CUDA/import health check. A failed signature is preserved and required
native packages are rebuilt, allowing Repair to heal missing/corrupt native
binaries. A failed setup does not set readiness, delete weights or
silently disable post-processing/native features. No setup script installs system
packages or modifies the host Modly environment.

ARM64 source wheels explicitly set `CUMM_DISABLE_JIT=1` and
`SPCONV_DISABLE_JIT=1`; otherwise ordinary builds omit their native extensions.
The pinned cumm0.7.11 predates Blackwell. Setup applies a documented local patch
to `cumm/common.py`: add `12.0` to the accepted architecture whitelist and make
its Linux toolkit fallback honor `CUDA_HOME` for both `include` and `lib64`. The
existing multi-digit parser and PTX-generation code remain unchanged. Setup checks
the exact original SHA-256, accepts only the identical previously patched file,
refuses unrelated modifications, and records both checksums plus the immutable
source revision in `.modly-cumm-native-patch.json` (earlier whitelist-only
provenance is retained when upgrading). This keeps spconv2.3.8's required
`cumm<0.8` contract without claiming that a build proves native correctness.

The selected toolkit path and per-library target policy are persisted in setup
state. Build, native probe and inference use the same CUDA_HOME, PATH,
LD_LIBRARY_PATH and architecture settings, including any spconv NVRTC execution.
`CUMM_INCLUDE_PATH` selects the pinned source's verified tensorview headers rather
than the unrelated `site-packages/include` installed by libigl. A private ninja
wrapper invokes the exact extension-venv binary with the requested job limit:
ccimport itself does not honor `MAX_JOBS`. Native-policy changes preserve old
build directories before rebuilding instead of reusing wrong-toolkit objects.
Each native source build receives its own source-first `PYTHONPATH`, preventing
an already installed package from corrupting the generated binding namespace.
ARM PyG builds force a fresh CUDA source installation: pip's ordinary wheel cache
does not encode the Torch/CUDA ABI. A child-only `PIP_NO_CACHE_DIR=1` preserves
this protection even though Modly removes the CLI `--no-cache-dir` flag. Other
packages retain the host's normal shared cache; no global cache is cleared.

The official ARM64 cuSPARSELt0.7.1 wheel has an internal `manylinux2014_sbsa`
tag that strict pip check rejects. The Linux/aarch64/**CPython3.11 or 3.12**/
torch2.9.1+cu128 lane includes a narrowly audited metadata repair: exact original
WHEEL/RECORD and every payload hash, venv containment, AArch64 ELF identity and
native library load are checked before atomically normalizing WHEEL and RECORD.
Unknown files/versions fail closed. The identical native wheel was independently
audited on Python3.11 before extending the ABI gate on 2026-10-28. No pip-check
errors are suppressed or native binaries changed.

For diagnostics/custom provisioning only (replace all placeholders with absolute
paths):

```bash
<host-python> <installed-extension>/setup.py '{"python_exe":"<host-python>","ext_dir":"<installed-extension>","models_dir":"<configured-models-dir>","gpu_sm":121,"cuda_version":128,"accelerator":"cuda","torch_flavor":"cuda","cuda_home":"/usr/local/cuda-12.8"}'
<installed-extension>/venv/bin/python <installed-extension>/scripts/native_probe.py --vendor <installed-extension>/vendor/PartLLM
```

## Outputs

Each run writes `<workspaceDir>/Workflows/partllm_<unique-id>/` containing all
segmented GLBs, `partllm.json`, `upstream.log` and the full native output tree
(metadata, logits, face labels and PLY exports). Intermediates use the supplied
`tempDir` and are cleaned. Original inputs are never overwritten.

GLB exports contain **actual separate submeshes**, not only vertex colors. Labels
are mapped only after exact face-order/topology and normalized-vertex checks.
Geometry is exported from the canonical input in its **original world coordinate
system**, not upstream's unit-diagonal normalized coordinates. Class-indexed part
names/face counts are saved in GLB extras and the sidecar report; generated name
parsing is best-effort, but numeric face labels are authoritative. This changes
materials to a segmentation palette and does not preserve PBR textures/rigs.

Upstream sometimes catches inference failures and exits with status zero. The
adapter checks `error.txt`, required face labels, metadata, native topology and
real output files; exit status alone cannot report success. The PROCESS stdout
contains NDJSON progress/logs and exactly one terminal `done` or `error`.
There is no Modly PROCESS cooperative-cancellation callback; the host owns process
termination. A terminated run may retain diagnostic/partial output folders.

## Limitations

### Interactive / refine: deferred, not hidden capabilities

Upstream interactive mode needs positive 3D surface clicks in **original mesh
coordinates**; refine also needs a same-topology per-face binary mask and positive
or negative correction points. Modly's ordinary PROCESS slots do not transport
mesh picking, click history or persistent face-mask state. No nonfunctional
interactive/refine nodes are advertised.

Possible future integrations:

1. A dedicated Modly mesh-viewer picking panel transports source coordinates,
   mesh identity/topology hash and click polarity; PROCESS consumes that structured
   artifact and emits the reusable binary mask alongside the mesh.
2. An explicit coordinate/mask-file workflow can offer headless one-shot clicks,
   but is less discoverable and must validate mask length/topology exactly.
3. A governed host UI/tool integration can manage iterative refinement sessions;
   it needs an explicit host API instead of inventing manifest fields.

The required upstream inference/import dependencies are installed for these paths,
but UI/runtime interaction remains future work. Optional vLLM acceleration is not
installed or exposed; it needs a separate versioned environment/bridge qualification
and is not required for native inference feature parity.

## Validation evidence

| Tier | Evidence (dates distinguish separate qualifications) |
| --- | --- |
| Contracts, geometry, guards, protocol and native-repair regressions | PASS: 77 tests under real Python3.11.9 and 3.12.3 on 2026-10-28, including both-ABI metadata guards and 4 exact-host-launcher checks |
| Instanced GLB flattening, world bounds, real submesh export, topology mismatch rejection | PASS (synthetic geometry tests) |
| Skill strict validator / actual host manifest validators | Clean-wrapper strict validator PASS (2026-10-28); host manifest validators PASS (2026-10-23); neither alone proves inference |
| Python3.11.9/Linux ARM64/GB10 full installation/native/inference | PASS (2026-10-28): own CP311 native builds, CUDA operations, imports, strict pip check, both actual launchers' Repair and both runners × both nodes (4/4 executions) |
| Linux x86_64 installation and inference | NOT RUN |
| Python3.12.3/Linux ARM64 runtime dependencies / strict pip check | PASS (2026-10-23); scoped cuSPARSELt repair applied and native library loaded |
| Complete checkpoint under configured models_dir | PASS: 14 files, 9,838,035,471 bytes; all published SHA-256 values verified |
| Python3.12.3/Linux ARM64/GB10 native setup and inference | PASS (2026-10-23): real CUDA operations/imports and both actual runners; all six full-shape prompt outputs exercised. No new CP312 inference run on 2026-10-28 |
| Python3.11 artifact geometry QC | PASS (2026-10-28): all 9 GLBs retain 72 source triangles exactly once, bounds delta0, native-class/submesh matches |
| Exact upstream/local setup launcher Repair | PASS from unrelated cwd with settings-based models_dir, healthy native reuse and strict pip check: CP312 (2026-10-23), isolated CP311 (2026-10-28) |
| Installed Python3.12 runtime preservation | PASS (2026-10-28): 26/26 protected identities unchanged, still Python3.12.3; no source/docs/state/venv synchronization |
| Electron UI Install-from-GitHub / viewer / restart | NOT RUN |
| Windows/macOS/AMD | UNSUPPORTED |

Independent fresh-context audits guided the native/header, installed-package
shadowing and actual host-launcher fixes. The original failed attempts are not
counted as successful validation; later clean native operations and model
generations establish the PASS entries above. Fine-grained model predictions can
still be imperfect: one synthetic-chair leg was classified as background in
`fine` mode, while geometry/topology remained intact. No wrapper remapping hides
upstream predictions. See [VALIDATION.md](VALIDATION.md).

The 2026-10-28 setup/helper/test change exists only in the source repository and
isolated `/tmp` stage; the working installed CP312 copy was intentionally not
updated. Earlier source/runtime parity is historical, not a current claim. The
provisioned stage's validator found 720 upstream placeholder errors in build/vendor
trees; this is not a clean-wrapper static PASS or a functional runtime failure.

Run the lightweight suite with a Python environment containing numpy/trimesh:

```bash
python -m unittest discover -s tests -v
python scripts/create_chair_fixture.py /absolute/path/chair.glb
```

The additional metadata-repair safety tests use pytest (developer test dependency,
not an inference requirement): `python -m pytest tests -q`.

Native probe checks CUDA scatter, FPS, varlen FlashAttention, sparse convolution,
and real full-shape/text-guided/interactive/postprocess imports. Its JSON report
explicitly says model inference is NOT RUN because that diagnostic alone cannot
prove generation. Separate real model generations are recorded in VALIDATION.md.

## Troubleshooting

- Missing checkpoint/inventory: run Repair; do not copy partial shards manually.
- Native build failure: verify CUDA12.8 nvcc, compiler and the exact Python ABI;
  inspect setup output. No failed native feature is silently skipped.
- Setup completed but inference failed: inspect that run's `upstream.log` and
  `native/**/error.txt`. Do not infer OOM from a generic CUDA launch error.
- Moved extension/models path: rerun Repair for that installation root.
- Large meshes: graph cut and logits consume additional host/GPU memory; lower
  requested granularity/part count or explicitly disable graph cut if needed.

## Credits

- Integration: DrHepa, MIT wrapper license.
- Upstream: czvvd/PartLLM at `524f4f68d2501bd9177e9f36f4f2ce4be090ab1b`, Apache-2.0 code.
- Checkpoint: Czvvd/PartLLM at `0c546a88ca52cced296200e173c8961b8f2ca0ba`, **terms unresolved**.
- Host: Modly by Lightning Pixel. Contracts audited against upstream
  `a471509b164b02bc573f4eafbe2dbf5149c1d6d1` and local `eddbbff` without editing either.

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), especially weight-use
uncertainty and PyMeshLab's GPL terms. Do not publish/redistribute a binary
environment or checkpoint under the wrapper MIT license.

## License

This wrapper is MIT licensed; original upstream/runtime/checkpoint terms are
separate. See the root `LICENSE`, `licenses/` and third-party notices.
