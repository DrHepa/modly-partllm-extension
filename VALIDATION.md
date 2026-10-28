# Validation record — updated 2026-10-28

## Isolated CPython3.11.9 qualification — 2026-10-28: PASS

Linux ARM64 / NVIDIA GB10 SM121 / CUDA toolkit12.8 / PyTorch2.9.1+cu128.
The isolated private venv uses SOABI `cpython-311-aarch64-linux-gnu`; its native
dependencies were compiled for CP311, not copied from the working CP312 runtime.
The source/checkpoint/host revisions are the same as the historical table below;
the local host's full revision is `eddbbff628fff627a40fd35a563c5261f8bd9fdc`.

- Actual setup completed with readiness true after pinned-source builds of cumm,
  spconv, torch-scatter, torch-cluster and FlashAttention. CUDA scatter, FPS,
  varlen FlashAttention and sparse convolution passed; full-shape/text-guided,
  interactive and post-processing imports passed. Strict `pip check` exited0.
  Torch's existing SM12.1/maxSM12.0 warning remained; these observed kernel
  successes do not qualify every GPU operation.
- Both exact upstream/local setup launchers completed real Repair from `/tmp`,
  resolving the existing models root from matching isolated settings. Native
  health checks passed before reuse; no CP312 Repair or rebuild was performed.
- Both actual `PythonProcessRunner` implementations ran both nodes: **4/4**
  executions, each with exactly one NDJSON `done`, child exit0 and a nonempty GLB
  under the isolated workspace's Workflows. Only `electron.app` was shimmed for
  the Node harness; model/kernel execution was real, not Electron GUI E2E.
- The original 72-triangle chair and connected text `seat; chair back; leg` were
  used with 512max_new_tokens and graph cut/post-processing enabled. Public
  full-shape `all/all` emitted six exports; local semantic/general emitted one;
  each text-guided run emitted one. Independent QC of all **9 GLBs** confirmed
  every world-space source triangle exactly once, bounds delta0 and native-class
  to GLB-submesh correspondence. Both text-guided GLBs had SHA256
  `6e554d56334e1ee84f03db65d2a66c50a70c31039061a26d3c4910e2b46ff96e`.
- Semantic quality is not guaranteed: fine output retained 12 background faces.
  The two full-shape parameterizations produced different general partitions
  within CP311; the cause was not isolated. No semantic-equivalence or cross-ABI
  determinism claim follows from geometric correctness.
- The pinned 14-file checkpoint was reused from configured models storage;
  qualification source, venv, builds, workspace and logs stayed in an isolated
  temporary directory. No second checkpoint download was needed.

### Metadata gate and preservation boundary

An independent audit confirmed the exact same cuSPARSELt0.7.1 six-file payload,
original WHEEL/RECORD, AArch64 ELF and native load under CP311. On 2026-10-28 only
the source repository and isolated stage's setup/helper/tests were updated to
allow CP311/CP312 for this exact metadata repair. Hashes, venv containment,
descriptor/symlink guards, RECORD, Torch/lane and ELF/load checks remain intact;
no native binary modification or CP312-extension reuse was introduced.

The working installed extension remained **Python3.12.3**: all **26/26** protected
source/config/readiness/native-state/interpreter identities and symlink bindings
matched before/after. No source, documentation, tests, state or venv was synced
there; no CP311 runtime was placed under the installed extensions directory.
The CP312 setup/inference PASS below remains the **2026-10-23** evidence, not a
new CP312 inference run. The 2026-10-23 20-file source/runtime parity statement is
historical: the current source intentionally differs from the installed copy.

### Current pure tests, static scope and evidence

The source suite passed **77 tests** under real Python3.11.9 and Python3.12.3,
including four tests using the actual upstream/local host launchers with all
`MODLY_TEST_*_HOST` variables set. Clean-wrapper strict skill validation passed
with zero errors/warnings; these checks do not replace the actual runtime proof.

The already provisioned `/tmp` stage's strict-validator invocation instead
reported **720 upstream placeholder errors**: 690 under `.build`, 30 under
`vendor`, zero warnings. This is not a clean-wrapper static PASS or a runtime
failure. No upstream code was stripped and no validator was weakened.

Local provenance includes setup/Repair logs, native diagnostics, dependency checks,
runner results, geometry QC, metadata-gate checks and installed-runtime preservation
snapshots. The final job snapshot contained no owned active setup/build/runner jobs.
These retained local records are not included as public release artifacts; the
qualification results and limitations are summarized here.

## Historical CPython3.12.3 qualification — 2026-10-23

## Confirmed environment

Ubuntu24.04.5LTS, Linux7.0.0-1019-nvidia, ARM64, NVIDIA GB10 (SM121),
driver580.178.04, CUDA toolkit/runtime12.8, CPython3.12.3,
PyTorch2.9.1+cu128, torchvision0.24.1. Native policy4 builds PyG/cumm/spconv
for SM120+compute120 PTX and FlashAttention2.8.3 for SM120 cubins.

| Authority | Immutable revision |
| --- | --- |
| PartLLM source | `524f4f68d2501bd9177e9f36f4f2ce4be090ab1b` |
| Czvvd/PartLLM checkpoint | `0c546a88ca52cced296200e173c8961b8f2ca0ba` |
| Modly upstream host | `a471509b164b02bc573f4eafbe2dbf5149c1d6d1` |
| Local Modly host | `eddbbff` (pre-existing dirty checkout preserved) |

## Provisioning and Repair: PASS on Python3.12

- Real managed-interpreter setup created the extension's private venv and installed
  the full native inference/post-processing dependency closure.
- All14 checkpoint files, **9,838,035,471bytes**, were downloaded under the actual
  configured `models_dir/partllm`; revision, sizes, shard index and published LFS
  SHA-256 values passed verification. No separate Utonia checkpoint was fetched.
- Native CUDA scatter, FPS, varlen FlashAttention and sparse convolution passed.
  Full-shape, grounding, interactive, post-processing, custom Qwen and Utonia
  imports passed. Strict `pip check`: no broken requirements.
- Both exact Modly setup-launcher implementations ran real idempotent Repair from
  an unrelated cwd. Models were resolved from actual matching settings without an
  explicit models-dir override. Native health checks passed before reuse; no
  unnecessary rebuild occurred. No global CUDA symlink/account settings changed.
- 55 regression tests passed, including four checks against the exact host
  launchers. Strict extension validation and both host manifest validators passed.

### Corrected installation failures, not counted as PASS

1. ARM64 has no official CUDA12.8 torch2.8 wheel: an explicit ARM2.9.1 lane was
   implemented; x86_64 retains2.8.
2. An expired implicit HF token blocked a public checkpoint: public requests now
   opt out of authentication without modifying account configuration.
3. Exact cuSPARSELt0.7.1 SBSA metadata defect: narrowly gated CPython3.12 repair
   validates payload hashes/ELF/native load and atomically fixes WHEEL/RECORD.
4. cumm selected libigl's unrelated headers and the global CUDA13 toolkit:
   explicit tensorview headers and SHA-verified CUDA_HOME fallback patch fixed it.
   Original cumm source SHA256 is `351102fb388061cb57849bc39628167bcb5f376080161c2e3f302dcb3e2ff2e0`;
   patched SHA256 is `82981c93f19c5a9d10220059d82a6b4c492dd33c226d3c06b525b3037afa9fd3`.
5. Installed cumm shadowed its source during Repair, generating a wrong binding
   namespace: source-first per-package build environments fixed it.
6. A globally cached torch-scatter source wheel caused SIGSEGV on import. Separate
   process/faulthandler isolation identified that module; rebuilding against the
   actual Torch/CUDA stack fixed it. PyG-only cache disabling now survives the
   host launcher's CLI rewriting. The global cache was not cleared.

Earlier build directories and failure logs were preserved. No kernel algorithms
or checkpoint tensors were replaced to bypass these failures.

## Real model inference through both host runners: PASS

The harness bundles each **actual** `PythonProcessRunner` with esbuild and uses
an `electron.app` shim for the Node-only harness. Python, CUDA kernels, checkpoint
loading, inference, post-processing and exported files are real: there are no
model/kernel mocks. This is host-runner integration, **not Electron UI E2E**.

Input: original synthetic six-component chair produced by
`scripts/create_chair_fixture.py`,48vertices/72triangles. Original assets were not
overwritten. Connected text was `seat; chair back; leg`. Smoke generations used
512max_new_tokens; post-processing and graph cut were enabled.

| Host | Node / exercise | Result |
| --- | --- | --- |
| Upstream | Full-shape,3 requested parts | PASS; Seat12faces, Backrest12, Base48 |
| Local | Text-guided, connected text | PASS; valid partition GLB |
| Local | Full-shape, all semantic/nonsemantic granularities | PASS; all6exports, primary semantic `general` |
| Upstream | Text-guided, final label parser | PASS; valid partition GLB |

Each execution returned one NDJSON `done`, exited0 and produced a nonempty GLB
under the active workspace's Workflows. Runtime was offline/local-only. Current
JSON-per-SEG labels are parsed as semantic names rather than leftover bbox JSON.
Nonsemantic primary selection respects the requested granularity as well.

### Qualified artifact integrity

| Artifact role | SHA256 |
| --- | --- |
| Full-shape semantic general GLB | `e4cdc47c89215ea6b5e12d356f451e102cdd4ac8f443277391c20428a5d7ca29` |
| Text-guided GLB | `6e554d56334e1ee84f03db65d2a66c50a70c31039061a26d3c4910e2b46ff96e` |

Sidecars and complete native outputs were retained with the local artifacts. Independent
QC checked all6full-shape exports against native labels, normalized PLY and source
geometry:72triangles preserved, no extra/lost faces, source bounds delta0. A fresh
Blender import and six opened views validated the general result. GLBs use vertex
COLOR_0 segmentation colors, not authored PBR materials. Fine predictions are not
perfect: one leg was background and the seat was split across Frame/Cushion. This
is an upstream prediction limitation, not wrapper geometry loss.

## Pending / not claimed

- **Python3.11 historical blocker resolved on 2026-10-28:** the earlier slow
  dependency downloads were completed and isolated setup/native/inference passed
  as recorded above; the 2026-10-23 incomplete status is no longer current.
- Linux x86_64 hardware inference, Windows/macOS/AMD, a clean published GitHub
  installation, Electron UI/viewer/restart/cancellation and interactive/refine UI
  integration are not qualified here.
- Checkpoint licensing remains unresolved; see THIRD_PARTY_NOTICES.md. No binary
  environment/checkpoint redistribution permission is implied by these tests.

## Local provenance

Historical setup/Repair logs, native diagnostics, runner results and output QC
were retained locally. They are not included as public release artifacts and
should not be treated as a durable public evidence bundle.
