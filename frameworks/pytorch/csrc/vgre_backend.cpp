// VGRE PyTorch backend (out-of-tree, PrivateUse1).  docs/missingFeatures.md §4.2.
//
// Registers VGRE as a PyTorch device ("vgre"): tensor storage is allocated by
// VGRE's emulated cudaMalloc (host-addressable device memory), host<->device
// copies go through cudaMemcpy, and matmul is routed through VGRE's emulated
// cuBLAS (cublasSgemm). All other ATen ops are covered by a CPU fallback that
// transfers operands to CPU, runs the op, and copies results back.
//
// VGRE's cuda*/cublas* symbols are resolved by dlsym from the libvgre_cudart
// handle rather than the linker's global table: a CUDA-enabled PyTorch build
// also exports cudaMalloc (real driver) and would otherwise interpose, so we
// must call VGRE's implementation explicitly.

#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <cstring>
#include <dlfcn.h>

namespace {

constexpr int kCudaMemcpyH2D = 1, kCudaMemcpyD2H = 2, kCudaMemcpyD2D = 3;
constexpr int kCublasOpN = 0;

// ── VGRE emulated runtime, resolved from libvgre_cudart (not the global table) ─
struct VgreRt {
    int (*cudaMalloc)(void**, size_t) = nullptr;
    int (*cudaFree)(void*) = nullptr;
    int (*cudaMemcpy)(void*, const void*, size_t, int) = nullptr;
    int (*cudaDeviceSynchronize)() = nullptr;
    int (*cudaGetDeviceCount)(int*) = nullptr;
    int (*cublasCreate)(void**) = nullptr;
    int (*cublasDestroy)(void*) = nullptr;
    int (*cublasSgemm)(void*, int, int, int, int, int, const float*, const float*, int,
                       const float*, int, const float*, float*, int) = nullptr;
};

const VgreRt& rt() {
    static VgreRt r = [] {
        VgreRt v;
        // The library is already loaded (linked as a dependency); get its handle.
        void* h = dlopen("libvgre_cudart.so", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libvgre_cudart.so", RTLD_NOW); // fallback: load it
        TORCH_CHECK(h, "VGRE backend: cannot dlopen libvgre_cudart.so: ", dlerror());
        auto sym = [&](const char* n) {
            void* p = dlsym(h, n);
            TORCH_CHECK(p, "VGRE backend: missing symbol ", n);
            return p;
        };
        v.cudaMalloc = reinterpret_cast<decltype(v.cudaMalloc)>(sym("cudaMalloc"));
        v.cudaFree = reinterpret_cast<decltype(v.cudaFree)>(sym("cudaFree"));
        v.cudaMemcpy = reinterpret_cast<decltype(v.cudaMemcpy)>(sym("cudaMemcpy"));
        v.cudaDeviceSynchronize = reinterpret_cast<decltype(v.cudaDeviceSynchronize)>(sym("cudaDeviceSynchronize"));
        v.cudaGetDeviceCount = reinterpret_cast<decltype(v.cudaGetDeviceCount)>(sym("cudaGetDeviceCount"));
        v.cublasCreate = reinterpret_cast<decltype(v.cublasCreate)>(sym("cublasCreate"));
        v.cublasDestroy = reinterpret_cast<decltype(v.cublasDestroy)>(sym("cublasDestroy"));
        v.cublasSgemm = reinterpret_cast<decltype(v.cublasSgemm)>(sym("cublasSgemm"));
        int n = 0;
        v.cudaGetDeviceCount(&n); // initialize the VGRE runtime
        return v;
    }();
    return r;
}

// ── Allocator backed by VGRE device memory ──────────────────────────────────
struct VGREAllocator final : public c10::Allocator {
    static void deleter(void* ptr) { if (ptr) rt().cudaFree(ptr); }

    c10::DataPtr allocate(size_t nbytes) override {
        void* p = nullptr;
        if (nbytes) {
            int rc = rt().cudaMalloc(&p, nbytes);
            TORCH_CHECK(rc == 0 && p, "VGRE cudaMalloc failed (rc=", rc, ") for ", nbytes, " bytes");
        }
        return {p, p, &deleter, c10::Device(c10::DeviceType::PrivateUse1, 0)};
    }
    c10::DeleterFnPtr raw_deleter() const override { return &deleter; }
    void copy_data(void* dest, const void* src, std::size_t count) const override {
        rt().cudaMemcpy(dest, src, count, kCudaMemcpyD2D);
    }
};

VGREAllocator g_alloc;

// ── Device guard so PyTorch can target the "vgre" device ────────────────────
struct VGREGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType kType = c10::DeviceType::PrivateUse1;
    c10::DeviceType type() const override { return kType; }
    c10::Device exchangeDevice(c10::Device) const override { return c10::Device(kType, 0); }
    c10::Device getDevice() const override { return c10::Device(kType, 0); }
    void setDevice(c10::Device) const override {}
    void uncheckedSetDevice(c10::Device) const noexcept override {}
    c10::Stream getStream(c10::Device d) const override { return c10::Stream(c10::Stream::DEFAULT, d); }
    c10::Stream exchangeStream(c10::Stream) const override {
        return c10::Stream(c10::Stream::DEFAULT, c10::Device(kType, 0));
    }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, VGREGuardImpl);

// ── Tensor factory ops ──────────────────────────────────────────────────────
at::Tensor vgre_empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                                    std::optional<at::Layout>, std::optional<at::Device>,
                                    std::optional<bool>, std::optional<at::MemoryFormat> mf) {
    constexpr c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
    return at::Tensor(at::detail::empty_generic(size, &g_alloc, ks, c10::dtype_or_default(dtype), mf));
}

at::Tensor vgre_empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                              std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                              std::optional<at::Device>, std::optional<bool>) {
    constexpr c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
    return at::Tensor(
        at::detail::empty_strided_generic(size, stride, &g_alloc, ks, c10::dtype_or_default(dtype)));
}

// ── Cross-device copy (CPU <-> VGRE) via cudaMemcpy ─────────────────────────
at::Tensor vgre_copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
    TORCH_CHECK(self.sizes() == dst.sizes(), "VGRE _copy_from: size mismatch");
    at::Tensor s = self.contiguous();
    TORCH_CHECK(dst.is_contiguous(), "VGRE _copy_from: non-contiguous destination unsupported");
    bool srcDev = self.device().type() == c10::DeviceType::PrivateUse1;
    bool dstDev = dst.device().type() == c10::DeviceType::PrivateUse1;
    int kind = (srcDev && dstDev) ? kCudaMemcpyD2D
               : (!srcDev && dstDev) ? kCudaMemcpyH2D
               : (srcDev && !dstDev) ? kCudaMemcpyD2H : 0;
    size_t nbytes = static_cast<size_t>(s.numel()) * s.element_size();
    if (nbytes) {
        if (kind == 0) std::memcpy(dst.data_ptr(), s.data_ptr(), nbytes);
        else rt().cudaMemcpy(dst.data_ptr(), s.data_ptr(), nbytes, kind);
    }
    return dst;
}

// ── Matmul via VGRE's emulated cuBLAS (genuine GPU-emulation compute) ───────
at::Tensor vgre_mm(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "VGRE mm currently supports float32");
    at::Tensor A = a.contiguous(), B = b.contiguous();
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "VGRE mm: shape mismatch");
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    at::Tensor C = at::Tensor(at::detail::empty_generic(
        {M, N}, &g_alloc, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), at::kFloat, std::nullopt));

    float alpha = 1.0f, beta = 0.0f;
    void* handle = nullptr;
    rt().cublasCreate(&handle);
    // Row-major C = A·B computed as column-major C^T = B^T·A^T.
    rt().cublasSgemm(handle, kCublasOpN, kCublasOpN, (int)N, (int)M, (int)K, &alpha,
                     static_cast<const float*>(B.data_ptr()), (int)N,
                     static_cast<const float*>(A.data_ptr()), (int)K, &beta,
                     static_cast<float*>(C.data_ptr()), (int)N);
    rt().cudaDeviceSynchronize();
    rt().cublasDestroy(handle);
    return C;
}

// Used by the CPU fallback to write results back into device output tensors.
at::Tensor vgre_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return vgre_copy_from(self, dst, /*non_blocking*/ false);
}

// ── Generic CPU fallback for every other op ─────────────────────────────────
void vgre_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    at::native::cpu_fallback(op, stack);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", &vgre_empty_memory_format);
    m.impl("empty_strided", &vgre_empty_strided);
    m.impl("_copy_from", &vgre_copy_from);
    m.impl("_copy_from_and_resize", &vgre_copy_from_and_resize);
    m.impl("mm", &vgre_mm);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&vgre_cpu_fallback>());
}

REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_alloc);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VGRE PyTorch backend (PrivateUse1): alloc/copy via emulated CUDA, mm via cuBLAS";
}
