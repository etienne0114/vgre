# VGRE WebAssembly target (§3.3)

VGRE's compute kernels (`vgre_kernels.c`: vadd, vmul, saxpy, relu, reduce_sum,
sgemm) compiled to a freestanding **wasm32** module — sandboxed execution that
runs in any WebAssembly runtime (browser, Node, wasmtime) with no GPU and no
host syscalls.

## Build + run

No emscripten needed — clang's built-in WebAssembly backend is sufficient:

```bash
clang --target=wasm32 -O2 -nostdlib -Wl,--no-entry -Wl,--export-dynamic \
      -Wl,--export=__heap_base frameworks/wasm/vgre_kernels.c -o vgre_kernels.wasm
node frameworks/wasm/test_wasm.mjs    # compiles + runs + checks (CI: WasmKernels ctest)
```

## Scope

This is the WASM half of §3.3 (sandboxed execution). The RISC-V half needs a
`riscv64` cross-toolchain + QEMU, which aren't installed here; the kernels are
plain C and cross-compile unchanged once that toolchain is present.
