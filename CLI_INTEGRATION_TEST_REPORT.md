# CLI / GGUF Integration Status

This report supersedes the earlier Task 9.3 report, which incorrectly said
`vgre chat` and `vgre pull` were missing. Both commands are implemented.

## Current behavior

- `vgre chat` defaults to the pinned SmolLM2 preset; `vgre chat smollm2` selects
  the same model explicitly. Downloads use a platform-specific cache or the
  `VGRE_MODEL_CACHE_DIR` override.
- `vgre chat <model.gguf>` reads GGUF metadata, configures the model, loads
  weights, finds a neighboring tokenizer sidecar, and starts an interactive
  session.
- `vgre pull smollm2` downloads the pinned SmolLM2 GGUF and tokenizer files.
- `vgre generate --model <model.gguf>` uses the same metadata-driven loader.
- GGUF metadata access and the model C API expose dimensions, RoPE base, and
  normalization epsilon. The transformer applies both values during inference.
- `--ternary` is wired through CLI → Python → C API → transformer decode and
  invokes the packed 2-bit GEMM; this mode is lossy and opt-in.
- The source CLI prefers the runtime installed by `vgre_sync.sh`. The sync
  installer also replaces the stale native library bundled in the venv package
  and constructs a small model to verify the required C API before reporting the
  Python model CLI ready.

## Reported errors

There were two different failures. The earlier `undefined symbol:
vgre_lm_set_rope_norm` came from loading a Python package library that did not
match the current Python bindings. The installer now copies the just-built
native libraries into the managed package, the source dispatcher prefers the
fresh install, and the install check constructs a model using the new API.

The latest pasted log is different: `model.gguf` could not be opened because
that file does not exist at the current working directory. `vgre_sync.sh`
installs the runtime, not model weights. The CLI now reports that clearly and
points to `vgre chat smollm2` (or `vgre pull smollm2`) for the supported preset.

## Verification

- Downloaded the pinned SmolLM2 GGUF and tokenizer into a temporary cache, then
  ran real Q4_K model loading, text generation, and interactive chat on CPU.
- Ran the same cached model through `--ternary`; packed-ternary generation and
  chat both completed through the real native backend.
- The Python regression tests verify command registration, help, missing-file
  behavior, metadata extraction, and generation with a valid GGUF fixture.
- The Linux CTest regression `VgreCliNativeLibrarySelection` passes.
- Focused CTests cover GGUF quantization, the metadata C API, metadata-driven
  Python generation, native-library selection, and ternary decode.
- CI includes a pretrained integration step that pulls the real model and runs
  generate and interactive chat.
- Repeated locally on 2026-10-02 with the downloaded 105 MB SmolLM2 Q4_K_M
  model: `vgre pull` reused and validated the cache, `vgre generate` emitted
  text, and piped input through `vgre chat` produced an `Assistant:` response.

Phi-3 is still rejected intentionally: the loader currently supports Llama and
Qwen2 tensor layouts, while Phi-3 GGUF uses additional architecture-specific
tensor/attention handling. The transformer model path is CPU inference; this
verification does not claim physical GPU model inference.

PyPI publication and Vercel Git integration are external release/deployment
settings; repository tests cannot verify credentials or configure those hosted
services.

## Prior GitHub CI audit (2026-10-02)

At the time, the latest available GitHub Actions run for `main` was [run 36388156153](https://github.com/etienne0114/vgre/actions/runs/36388156153), created on 2026-09-28. Its Linux CTest artifact showed the same LLVM command-line option registration crash in `PythonLmBindings`, `PythonNn`, `PyTorchIntegration`, and `TensorFlowIntegration`. The Windows build and tests passed; the macOS test step failed. GitHub redirected the macOS log download to its artifact storage host, which was unreachable from this environment, so that failure's exact test is not confirmed here.

The Linux cause was the Python loader preloading the sibling `libvgre_cudart` library into the same process as `libvgre`; both libraries contain the statically linked LLVM runtime. The loader now skips VGRE libraries during sibling preloading. Local CTest verification passed `PythonLmBindings`, `PythonNn`, `PythonGgufAutoConfig`, and `VgreCliNativeLibrarySelection`; the metadata C and C++ API tests and `TCPClusterSecurityHybrid` also pass.

At that point, the cross-platform Python integration set passed 104 tests locally. Its two memory-measurement tests skipped because `psutil` was not installed in the local virtual environment; CI installs that dependency. The workflow ran this set on Linux, macOS, and Windows, included the previously excluded security hybrid test, uploaded CTest logs on failure for each OS, and treated each OS job as required. The later hosted run and its fixes are recorded below.

The full native project build completed locally, and all 410 CTest tests passed
with `HOME` redirected to a temporary test directory. The weak chat test fixtures
that encoded all GGUF metadata as strings and accepted arbitrary CLI failures
were replaced with subprocess checks using the structurally valid typed-metadata
GGUF fixture and actual native inference. The CI workflow also runs generation
and interactive chat against the downloaded SmolLM2 model on each OS.

Removed the unreferenced root-level `test_model.gguf` (15-byte plain text, not a GGUF file) and `test_bench.json` (unreferenced benchmark numbers without provenance).

## Latest GitHub CI run (2026-10-02)

Run [37005920162](https://github.com/etienne0114/vgre/actions/runs/37005920162)
on commit `3cd525f5` exposed two issues. Linux, macOS, and Windows all passed
the real model pull and generation steps, then failed the chat smoke assertion:
the interactive output prefixes `Assistant:` with `User: `, while the workflow
required it at the start of the line. The Linux LLVM-free CTest artifact showed
one failing test, `TCPClusterSecurityHybrid`; its test expected token
auto-generation although production correctly requires a configured shared
token before enabling cluster security.

The smoke checks now search for the assistant label anywhere in the output on
all three operating systems. The security integration test now supplies an
isolated configured token and checks initialization results; production remains
fail-closed. The targeted `TCPClusterSecurityHybrid` CTest and the complete
local Linux CTest suite pass (410/410). Local socket tests were run with
localhost socket access. A new hosted run is needed to verify these changes on
the GitHub Linux, macOS, Windows, and LLVM-free runners.
