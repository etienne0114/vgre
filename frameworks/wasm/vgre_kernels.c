// VGRE compute kernels for the multi-architecture targets (docs/missingFeatures.md
// §3.3): the GPU-style compute kernels VGRE runs, as portable freestanding C that
// cross-compiles unchanged to WebAssembly (clang wasm32) and RISC-V (riscv64-gcc).
// On wasm, pointers are byte offsets into linear memory; the export_name attribute
// is wasm-only and elided on other targets.
typedef float f32;

#ifdef __wasm__
#define VGRE_EXPORT(name) __attribute__((export_name(name)))
#else
#define VGRE_EXPORT(name)
#endif

VGRE_EXPORT("vadd")
void vadd(const f32* a, const f32* b, f32* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

VGRE_EXPORT("vmul")
void vmul(const f32* a, const f32* b, f32* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

VGRE_EXPORT("saxpy")
void saxpy(f32 alpha, const f32* x, f32* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

VGRE_EXPORT("relu")
void relu(const f32* x, f32* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

VGRE_EXPORT("reduce_sum")
f32 reduce_sum(const f32* x, int n) {
    f32 s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
}

// Row-major C[MxN] = A[MxK] * B[KxN]
VGRE_EXPORT("sgemm")
void sgemm(const f32* A, const f32* B, f32* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            f32 s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}
