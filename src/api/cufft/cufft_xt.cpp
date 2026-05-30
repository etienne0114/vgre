// cuFFT Xt multi-GPU / callback API — VGRE single-CPU emulation layer.
// All Xt calls are thin wrappers over the existing single-GPU FFT paths.

#include "cufft_internal.h"
#include <cstdlib>
#include <cstring>

using namespace vgre_cufft;

// ── cudaDataType enum values (mirrors cusparse_shim.h / CUDA headers) ─────────
enum vgre_xt_dtype {
    VGRE_XT_R_32F  = 0,   // float
    VGRE_XT_R_64F  = 1,   // double
    VGRE_XT_R_16F  = 2,   // half
    VGRE_XT_C_32F  = 4,   // complex float
    VGRE_XT_C_64F  = 5,   // complex double
};

// Compute the output buffer size for an Xt plan allocation.
// Uses the complex element size multiplied by the total logical element count
// and batch, safe for all transform directions.
static size_t xtPlanBufferBytes(const CufftPlan &p) {
    int64_t logElems = logicalElementCount(p);
    size_t elem = complexElementSize(p.type);
    return static_cast<size_t>(logElems) * static_cast<size_t>(p.batch) * elem;
}

extern "C" {

// ── cufftXtMalloc ─────────────────────────────────────────────────────────────
cufftResult_t cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor,
                             cufftXtSubFormat_t /*format*/)
{
    if (!descriptor) return CUFFT_INVALID_VALUE;

    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;

    size_t bytes = xtPlanBufferBytes(p);
    if (bytes == 0) return CUFFT_INVALID_VALUE;

    cudaLibXtDesc *desc = static_cast<cudaLibXtDesc*>(
                              std::malloc(sizeof(cudaLibXtDesc)));
    if (!desc) return CUFFT_ALLOC_FAILED;

    void *buf = std::malloc(bytes);
    if (!buf) { std::free(desc); return CUFFT_ALLOC_FAILED; }

    std::memset(desc, 0, sizeof(*desc));
    desc->version    = 0;
    desc->nGPUs      = 1;
    desc->GPUs[0]    = 0;
    desc->data[0]    = buf;
    desc->size[0]    = bytes;
    desc->subFormat  = nullptr;

    *descriptor = desc;
    return CUFFT_SUCCESS;
}

// ── cufftXtFree ───────────────────────────────────────────────────────────────
cufftResult_t cufftXtFree(cudaLibXtDesc *descriptor)
{
    if (!descriptor) return CUFFT_INVALID_VALUE;
    if (descriptor->data[0]) std::free(descriptor->data[0]);
    std::free(descriptor);
    return CUFFT_SUCCESS;
}

// ── cufftXtMemcpy ─────────────────────────────────────────────────────────────
cufftResult_t cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer,
                             cufftXtCopyType_t type)
{
    if (!dstPointer || !srcPointer) return CUFFT_INVALID_VALUE;

    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;

    size_t bytes = xtPlanBufferBytes(p);

    switch (type) {
    case CUFFT_COPY_HOST_TO_DEVICE: {
        // src = plain host buffer, dst = Xt descriptor
        cudaLibXtDesc *dst = static_cast<cudaLibXtDesc*>(dstPointer);
        if (!dst->data[0]) return CUFFT_INVALID_VALUE;
        std::memcpy(dst->data[0], srcPointer, bytes);
        break;
    }
    case CUFFT_COPY_DEVICE_TO_HOST: {
        // src = Xt descriptor, dst = plain host buffer
        cudaLibXtDesc *src = static_cast<cudaLibXtDesc*>(srcPointer);
        if (!src->data[0]) return CUFFT_INVALID_VALUE;
        std::memcpy(dstPointer, src->data[0], bytes);
        break;
    }
    case CUFFT_COPY_DEVICE_TO_DEVICE: {
        // both are Xt descriptors
        cudaLibXtDesc *dst = static_cast<cudaLibXtDesc*>(dstPointer);
        cudaLibXtDesc *src = static_cast<cudaLibXtDesc*>(srcPointer);
        if (!dst->data[0] || !src->data[0]) return CUFFT_INVALID_VALUE;
        std::memcpy(dst->data[0], src->data[0], bytes);
        break;
    }
    default:
        return CUFFT_INVALID_VALUE;
    }

    return CUFFT_SUCCESS;
}

// ── cufftXtExecDescriptor ─────────────────────────────────────────────────────
cufftResult_t cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc *input,
                                    cudaLibXtDesc *output, int direction)
{
    if (!input || !output) return CUFFT_INVALID_VALUE;

    void *idata = input->data[0];
    void *odata = output->data[0];
    if (!idata || !odata) return CUFFT_INVALID_VALUE;

    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;

    switch (p.type) {
    case CUFFT_C2C:
    case CUFFT_C16C:
        return cufftExecC2C(plan, idata, odata, direction);
    case CUFFT_Z2Z:
        return cufftExecZ2Z(plan, idata, odata, direction);
    case CUFFT_R2C:
    case CUFFT_R16C:
        return cufftExecR2C(plan, idata, odata);
    case CUFFT_C2R:
    case CUFFT_C16R:
        return cufftExecC2R(plan, idata, odata);
    case CUFFT_D2Z:
        return cufftExecD2Z(plan, idata, odata);
    case CUFFT_Z2D:
        return cufftExecZ2D(plan, idata, odata);
    default:
        return CUFFT_INVALID_TYPE;
    }
}

// ── cufftXtGetSizeMany ────────────────────────────────────────────────────────
cufftResult_t cufftXtGetSizeMany(cufftHandle plan, size_t *workSize,
                                 cufftXtSubFormat_t /*inputFormat*/,
                                 cufftXtSubFormat_t /*outputFormat*/,
                                 cufftXtSubFormat_t /*executionFormat*/,
                                 int /*executionType*/,
                                 int /*numGPUs*/, int * /*whichGPUs*/)
{
    if (!workSize) return CUFFT_INVALID_VALUE;
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    // CPU emulation needs no extra workspace
    *workSize = 0;
    return CUFFT_SUCCESS;
}

// ── cufftXtMakePlanMany ───────────────────────────────────────────────────────
// Map cudaDataType input/output pairs to cufftType_t, then fill the plan.
cufftResult_t cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n,
                                  long long int *inembed, long long int istride,
                                  long long int idist, int inputType,
                                  long long int *onembed, long long int ostride,
                                  long long int odist, int outputType,
                                  long long int batch, size_t *workSize,
                                  int /*executionType*/)
{
    if (!n || batch <= 0 || rank < 1 || rank > 3) return CUFFT_INVALID_VALUE;
    if (!workSize) return CUFFT_INVALID_VALUE;

    // Determine cufftType from inputType/outputType
    cufftType_t fftType;
    bool inReal   = (inputType  == VGRE_XT_R_32F);
    bool inDouble = (inputType  == VGRE_XT_R_64F);
    bool outReal  = (outputType == VGRE_XT_R_32F);
    bool outDouble= (outputType == VGRE_XT_R_64F);
    bool inCplxD  = (inputType  == VGRE_XT_C_64F);
    bool outCplxD = (outputType == VGRE_XT_C_64F);

    if (inDouble && outCplxD) {
        fftType = CUFFT_D2Z;
    } else if (inCplxD && outDouble) {
        fftType = CUFFT_Z2D;
    } else if (inCplxD && outCplxD) {
        fftType = CUFFT_Z2Z;
    } else if (inReal && !outReal) {
        fftType = CUFFT_R2C;
    } else if (!inReal && outReal) {
        fftType = CUFFT_C2R;
    } else {
        // Default: complex float
        fftType = CUFFT_C2C;
    }

    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(static_cast<uint64_t>(plan));
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;

    CufftPlan &p = it->second;
    p.rank    = rank;
    p.batch   = static_cast<int>(batch);
    p.type    = fftType;
    p.advanced = true;
    p.nx = static_cast<int>(n[0]);
    p.ny = (rank >= 2) ? static_cast<int>(n[1]) : 0;
    p.nz = (rank >= 3) ? static_cast<int>(n[2]) : 0;
    p.istride = static_cast<int>(istride);
    p.ostride = static_cast<int>(ostride);
    p.idist   = static_cast<int>(idist);
    p.odist   = static_cast<int>(odist);
    for (int i = 0; i < rank; ++i) {
        p.inembed[i] = inembed ? static_cast<int>(inembed[i]) : static_cast<int>(n[i]);
        p.onembed[i] = onembed ? static_cast<int>(onembed[i]) : static_cast<int>(n[i]);
    }
    setDefaultDistances(p);

    *workSize = 0;
    return CUFFT_SUCCESS;
}

// ── cufftXtSetCallback ────────────────────────────────────────────────────────
// VGRE CPU emulation: callbacks cannot be executed as GPU function pointers.
// Accept and silently ignore — return success so application code proceeds.
cufftResult_t cufftXtSetCallback(cufftHandle plan, void ** /*callbackRoutine*/,
                                  cufftXtCallbackType_t /*cbType*/,
                                  void ** /*callerInfo*/)
{
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    return CUFFT_SUCCESS;
}

// ── cufftXtClearCallback ──────────────────────────────────────────────────────
cufftResult_t cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType_t /*cbType*/)
{
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    return CUFFT_SUCCESS;
}

} // extern "C"
