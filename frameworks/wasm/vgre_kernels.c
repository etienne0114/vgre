// VGRE compute kernels compiled to WebAssembly (docs/missingFeatures.md §3.3).
// Freestanding wasm32 (no libc / no syscalls) — the GPU-style compute kernels
// VGRE runs, here in a sandboxed WASM module loadable by any WASM runtime
// (browser, node, wasmtime). Pointers are byte offsets into wasm linear memory.
typedef float f32;

__attribute__((export_name("vadd")))
void vadd(const f32* a, const f32* b, f32* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

__attribute__((export_name("vmul")))
void vmul(const f32* a, const f32* b, f32* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

__attribute__((export_name("saxpy")))
void saxpy(f32 alpha, const f32* x, f32* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

__attribute__((export_name("relu")))
void relu(const f32* x, f32* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__attribute__((export_name("reduce_sum")))
f32 reduce_sum(const f32* x, int n) {
    f32 s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
}

// Row-major C[MxN] = A[MxK] * B[KxN]
__attribute__((export_name("sgemm")))
void sgemm(const f32* A, const f32* B, f32* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            f32 s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}
