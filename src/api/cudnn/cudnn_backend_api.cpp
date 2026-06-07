// cuDNN v8+ Backend API. Descriptor graphs are executed by routing each
// operation to the existing legacy cuDNN shim paths (conv, pool, activation,
// matmul, reductions, attention, etc.).
//
// For unimplemented operations, returns CUDNN_STATUS_NOT_SUPPORTED.

#include "cudnn_backend_internal.h"
#include "vgre/common/openmp_helper.h"
#include "vgre/common/logger.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <cstring>

// ── Global backend state ─────────────────────────────────────────────────────
// Definitions of the extern symbols declared in cudnn_backend_internal.h
std::unordered_map<uintptr_t, BackendNode> g_backendNodes;
uintptr_t g_nextBackendId = 1;

BackendNode* getNode(uintptr_t id) {
    auto it = g_backendNodes.find(id);
    return (it != g_backendNodes.end()) ? &it->second : nullptr;
}

const std::vector<uint64_t>* getAttrVec(const BackendNode* node, int attr) {
    if (!node) return nullptr;
    auto it = node->attrs.find(attr);
    return (it != node->attrs.end()) ? &it->second : nullptr;
}

uintptr_t getAttrUint64(const BackendNode* node, int attr, uintptr_t def) {
    auto* v = getAttrVec(node, attr);
    return (v && !v->empty()) ? static_cast<uintptr_t>((*v)[0]) : def;
}

float getAttrFloat(const BackendNode* node, int attr, float def) {
    auto* v = getAttrVec(node, attr);
    if (!v || v->empty()) return def;
    float f = 0.0f;
    memcpy(&f, &(*v)[0], sizeof(float));
    return f;
}

namespace {

static size_t backendAttributeElementSize(int attributeType) {
    switch (attributeType) {
    case 0: return sizeof(char);
    case 1: return sizeof(float);
    case 2: return sizeof(double);
    case 3: return sizeof(int32_t);
    case 4: return sizeof(int64_t);
    case 5: return sizeof(uint64_t);
    case 6: return sizeof(uint64_t); // descriptor handles are stored as ids
    default: return sizeof(uint64_t);
    }
}

static uint64_t readBackendAttributeElement(const uint8_t *base, size_t elemSize) {
    uint64_t out = 0;
    memcpy(&out, base, std::min(elemSize, sizeof(out)));
    return out;
}

static void writeBackendAttributeElement(uint8_t *base, size_t elemSize, uint64_t value) {
    memcpy(base, &value, std::min(elemSize, sizeof(value)));
}

static TensorDesc buildTensorDesc(uintptr_t nodeId) {
    TensorDesc td{};
    td.n = 1; td.c = 1; td.h = 1; td.w = 1;
    td.dtype = CUDNN_DATA_FLOAT;
    td.fmt = CUDNN_TENSOR_NCHW;
    const BackendNode* node = getNode(nodeId);
    if (!node) return td;
    auto* dims = getAttrVec(node, CUDNN_ATTR_TENSOR_DIMENSIONS);
    if (dims) {
        if (dims->size() > 0) td.n = static_cast<int>((*dims)[0]);
        if (dims->size() > 1) td.c = static_cast<int>((*dims)[1]);
        if (dims->size() > 2) td.h = static_cast<int>((*dims)[2]);
        if (dims->size() > 3) td.w = static_cast<int>((*dims)[3]);
    }
    auto* dt = getAttrVec(node, CUDNN_ATTR_TENSOR_DATA_TYPE);
    if (dt && !dt->empty()) td.dtype = static_cast<cudnnDataType_t>((*dt)[0]);
    return td;
}

static FilterDesc buildFilterDesc(uintptr_t nodeId) {
    FilterDesc fd{};
    fd.k = 1; fd.c = 1; fd.r = 1; fd.s = 1;
    fd.dtype = CUDNN_DATA_FLOAT;
    const BackendNode* node = getNode(nodeId);
    if (!node) return fd;
    auto* dims = getAttrVec(node, CUDNN_ATTR_TENSOR_DIMENSIONS);
    if (dims) {
        if (dims->size() > 0) fd.k = static_cast<int>((*dims)[0]);
        if (dims->size() > 1) fd.c = static_cast<int>((*dims)[1]);
        if (dims->size() > 2) fd.r = static_cast<int>((*dims)[2]);
        if (dims->size() > 3) fd.s = static_cast<int>((*dims)[3]);
    }
    auto* dt = getAttrVec(node, CUDNN_ATTR_TENSOR_DATA_TYPE);
    if (dt && !dt->empty()) fd.dtype = static_cast<cudnnDataType_t>((*dt)[0]);
    return fd;
}

static ConvDesc buildConvDesc(uintptr_t nodeId) {
    ConvDesc cd{};
    cd.pad_h = 0; cd.pad_w = 0; cd.str_h = 1; cd.str_w = 1; cd.dil_h = 1; cd.dil_w = 1;
    const BackendNode* node = getNode(nodeId);
    if (!node) return cd;
    auto* pre = getAttrVec(node, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS);
    if (pre) {
        if (pre->size() > 0) cd.pad_h = static_cast<int>((*pre)[0]);
        if (pre->size() > 1) cd.pad_w = static_cast<int>((*pre)[1]);
    }
    auto* str = getAttrVec(node, CUDNN_ATTR_CONVOLUTION_STRIDES);
    if (str) {
        if (str->size() > 0) cd.str_h = static_cast<int>((*str)[0]);
        if (str->size() > 1) cd.str_w = static_cast<int>((*str)[1]);
    }
    auto* dil = getAttrVec(node, CUDNN_ATTR_CONVOLUTION_DILATIONS);
    if (dil) {
        if (dil->size() > 0) cd.dil_h = static_cast<int>((*dil)[0]);
        if (dil->size() > 1) cd.dil_w = static_cast<int>((*dil)[1]);
    }
    return cd;
}

static std::unordered_map<uintptr_t, void*> parseVariantPack(void* variantPack) {
    std::unordered_map<uintptr_t, void*> ptrs;
    if (!variantPack) return ptrs;
    const BackendNode* vp = getNode(reinterpret_cast<uintptr_t>(variantPack));
    if (!vp) return ptrs;
    auto* ids = getAttrVec(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    auto* dps = getAttrVec(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    if (!ids || !dps) return ptrs;
    size_t n = std::min(ids->size(), dps->size());
    for (size_t i = 0; i < n; ++i) {
        ptrs[static_cast<uintptr_t>((*ids)[i])] = reinterpret_cast<void*>((*dps)[i]);
    }
    return ptrs;
}

static std::vector<uintptr_t> parseOperationSet(uintptr_t opSetId) {
    std::vector<uintptr_t> ops;
    const BackendNode* node = getNode(opSetId);
    if (!node) return ops;
    auto* v = getAttrVec(node, CUDNN_ATTR_OPERATIONSET_OPS);
    if (!v) return ops;
    for (auto val : *v) ops.push_back(static_cast<uintptr_t>(val));
    return ops;
}

static uintptr_t traversePlanToOpSet(uintptr_t planId) {
    const BackendNode* plan = getNode(planId);
    if (!plan) return 0;
    // If the plan itself is an operation set, return it directly
    if (plan->type == CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR) return planId;
    // Plan → Engine Config
    uintptr_t ecId = getAttrUint64(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    if (!ecId) return 0;
    const BackendNode* ec = getNode(ecId);
    if (!ec) return 0;
    // Engine Config → Engine
    uintptr_t engId = getAttrUint64(ec, CUDNN_ATTR_ENGINECFG_ENGINE);
    if (!engId) return 0;
    const BackendNode* eng = getNode(engId);
    if (!eng) return 0;
    // Engine → Operation Graph
    return getAttrUint64(eng, CUDNN_ATTR_ENGINE_OPERATION_GRAPH);
}

} // namespace

// ── Public helpers for cudnn_graph.cpp fusion ─────────────────────────────────

// Forward declaration needed by cudnn_executeFusedConvBN below.
// The full definition is in cudnn_convolution.cpp.
extern "C" cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnFilterDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    int, void*, size_t, const void*, cudnnTensorDescriptor_t, void*);

std::unordered_map<uintptr_t, void*> cudnn_parseVariantPack(void* variantPack) {
    return parseVariantPack(variantPack);
}

// Fused Conv+BatchNorm forward pass.
//
// INFERENCE  (phase=0, running mean/var provided):
//   Fold BN parameters into Conv weights using the algebraic identity:
//     γ*(Conv(X,W)-μ)/σ + β  =  Conv(X, W·γ/σ) + (β - γμ/σ)
//   Cost: O(K·C·R·S) weight scaling + single Conv + O(N·K·H·W) bias add.
//   Eliminates the intermediate Conv output tensor entirely.
//
// TRAINING  (phase=1, statistics computed from Conv output):
//   Two-pass algorithm: Conv to tmp buffer → per-channel mean/var → normalize.
//   The tmp buffer stays in L2 across both passes, amortising DRAM bandwidth
//   vs. materialising two separate kernel outputs.
cudnnStatus_t cudnn_executeFusedConvBN(cudnnHandle_t handle,
                                        uintptr_t convNodeId,
                                        uintptr_t bnNodeId,
                                        void* variantPack)
{
    auto dataPtrs = parseVariantPack(variantPack);

    const BackendNode* convNode = getNode(convNodeId);
    const BackendNode* bnNode   = getNode(bnNodeId);
    if (!convNode || !bnNode) return CUDNN_STATUS_INVALID_VALUE;

    // ── Conv attrs ─────────────────────────────────────────────────────────────
    uintptr_t xId    = getAttrUint64(convNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);
    uintptr_t wId    = getAttrUint64(convNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W);
    uintptr_t convYId = getAttrUint64(convNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y);
    uintptr_t cId    = getAttrUint64(convNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC);
    float alphaConv  = getAttrFloat(convNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0f);

    void* xPtr = dataPtrs.count(xId) ? dataPtrs.at(xId) : nullptr;
    void* wPtr = dataPtrs.count(wId) ? dataPtrs.at(wId) : nullptr;
    if (!xPtr || !wPtr) return CUDNN_STATUS_INVALID_VALUE;

    TensorDesc xDesc   = buildTensorDesc(xId);
    FilterDesc wDesc   = buildFilterDesc(wId);
    TensorDesc convYDesc = buildTensorDesc(convYId);
    ConvDesc   cDesc   = buildConvDesc(cId);

    // ── BN attrs ───────────────────────────────────────────────────────────────
    uintptr_t bnXId  = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, 0);
    if (!bnXId) bnXId = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
    uintptr_t bnYId  = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, 0);
    if (!bnYId) bnYId = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);

    uintptr_t scaleId  = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, 0);
    uintptr_t biasId   = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, 0);
    uintptr_t meanId   = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, 0);
    uintptr_t invVarId = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VAR_DESC, 0);
    uintptr_t epsId    = getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, 0);
    int phase = static_cast<int>(getAttrUint64(bnNode, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, 0));

    float epsilon = 1e-5f;
    if (epsId && dataPtrs.count(epsId) && dataPtrs.at(epsId))
        epsilon = *static_cast<const float*>(dataPtrs.at(epsId));

    void* bnYPtr = (bnYId && dataPtrs.count(bnYId)) ? dataPtrs.at(bnYId) : nullptr;
    if (!bnYPtr) return CUDNN_STATUS_INVALID_VALUE;

    TensorDesc bnYDesc = buildTensorDesc(bnYId);
    int K     = wDesc.k;
    int Cin   = wDesc.c, R = wDesc.r, S = wDesc.s;
    int N     = xDesc.n;
    int HWout = bnYDesc.h * bnYDesc.w;

    const float* scalePtr = (scaleId && dataPtrs.count(scaleId)) ?
                             static_cast<const float*>(dataPtrs.at(scaleId)) : nullptr;
    const float* biasPtr  = (biasId  && dataPtrs.count(biasId))  ?
                             static_cast<const float*>(dataPtrs.at(biasId))  : nullptr;

    std::vector<float> defScale, defBias;
    if (!scalePtr) { defScale.assign(K, 1.0f); scalePtr = defScale.data(); }
    if (!biasPtr)  { defBias.assign(K, 0.0f);  biasPtr  = defBias.data();  }

    if (phase == 0) {
        // ── INFERENCE: fold BN into Conv weights ──────────────────────────────
        // Algebraic identity:
        //   y[n,k,h,w] = γ[k]*(conv(X,W)[n,k,h,w] - μ[k])/sqrt(σ²[k]+ε) + β[k]
        //              = conv(X, W_f)[n,k,h,w] + b_f[k]
        // where W_f[k,c,r,s] = W[k,c,r,s] · γ[k]/sqrt(σ²[k]+ε)
        //       b_f[k]        = β[k] - γ[k]·μ[k]/sqrt(σ²[k]+ε)
        const float* runMean = (meanId   && dataPtrs.count(meanId))   ?
                                static_cast<const float*>(dataPtrs.at(meanId))   : nullptr;
        const float* runVar  = (invVarId && dataPtrs.count(invVarId)) ?
                                static_cast<const float*>(dataPtrs.at(invVarId)) : nullptr;

        if (!runMean || !runVar) {
            // No running stats: fall back to training path below
            phase = 1;
            goto training_path;
        }

        {
            const float* wOrig = static_cast<const float*>(wPtr);
            size_t kernelElems = static_cast<size_t>(Cin) * R * S;
            std::vector<float> wFused(static_cast<size_t>(K) * kernelElems);
            std::vector<float> bFused(K);

            for (int k = 0; k < K; ++k) {
                float sigma_k  = std::sqrt(runVar[k] + epsilon);
                float invSigma = scalePtr[k] / sigma_k;           // γ[k]/σ[k]
                size_t kOff    = static_cast<size_t>(k) * kernelElems;
                for (size_t i = 0; i < kernelElems; ++i)
                    wFused[kOff + i] = wOrig[kOff + i] * invSigma;
                bFused[k] = biasPtr[k] - invSigma * runMean[k];   // β[k] - γ[k]μ[k]/σ[k]
            }

            float alpha = alphaConv, beta = 0.0f;
            cudnnStatus_t s = cudnnConvolutionForward(
                handle, &alpha, &xDesc, xPtr,
                &wDesc, wFused.data(), &cDesc,
                0, nullptr, 0, &beta, &bnYDesc, bnYPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;

            // Add per-channel folded bias: y[n,k,h,w] += b_f[k]
            float* yf = static_cast<float*>(bnYPtr);
            for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) {
                float bk = bFused[k];
                int base = (n * K + k) * HWout;
                for (int hw = 0; hw < HWout; ++hw)
                    yf[base + hw] += bk;
            }

            VGRE_LOG_DEBUG("cuDNNBN", "CONV_NORM inference fold: K=" + std::to_string(K) +
                           " Cin=" + std::to_string(Cin) + " kernel=" + std::to_string(R) + "x" + std::to_string(S));
        }
    } else {
        // ── TRAINING: two-pass Conv → stats → normalize ───────────────────────
        training_path:
        size_t tmpElems = static_cast<size_t>(N) * K * HWout;
        std::vector<float> tmp(tmpElems);

        float alpha = alphaConv, beta = 0.0f;
        cudnnStatus_t s = cudnnConvolutionForward(
            handle, &alpha, &xDesc, xPtr,
            &wDesc, wPtr, &cDesc,
            0, nullptr, 0, &beta, &convYDesc, tmp.data());
        if (s != CUDNN_STATUS_SUCCESS) return s;

        // Per-channel mean: μ[k] = (1/N·HW) Σ(n,h,w) tmp[n,k,h,w]
        float inv_nhw = 1.0f / static_cast<float>(N * HWout);
        std::vector<float> mu(K, 0.0f), invStd(K, 0.0f);
        for (int k = 0; k < K; ++k) {
            double sum = 0.0;
            for (int n = 0; n < N; ++n)
            for (int hw = 0; hw < HWout; ++hw)
                sum += tmp[(n * K + k) * HWout + hw];
            mu[k] = static_cast<float>(sum * inv_nhw);
        }
        // Per-channel variance: σ²[k] = (1/N·HW) Σ(n,h,w) (tmp[n,k,h,w]-μ[k])²
        for (int k = 0; k < K; ++k) {
            double var = 0.0;
            float muk = mu[k];
            for (int n = 0; n < N; ++n)
            for (int hw = 0; hw < HWout; ++hw) {
                float d = tmp[(n * K + k) * HWout + hw] - muk;
                var += d * d;
            }
            invStd[k] = 1.0f / std::sqrt(static_cast<float>(var * inv_nhw) + epsilon);
        }

        // Normalize: y = γ*(x-μ)/σ + β  written directly to BN output
        float* yf = static_cast<float*>(bnYPtr);
        for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            float gisv  = scalePtr[k] * invStd[k];
            float shift = biasPtr[k] - gisv * mu[k];
            int base = (n * K + k) * HWout;
            for (int hw = 0; hw < HWout; ++hw)
                yf[base + hw] = gisv * tmp[base + hw] + shift;
        }

        // Write saved statistics to output tensors if callers provided them
        if (meanId && dataPtrs.count(meanId) && dataPtrs.at(meanId)) {
            float* outMean = static_cast<float*>(dataPtrs.at(meanId));
            for (int k = 0; k < K; ++k) outMean[k] = mu[k];
        }
        if (invVarId && dataPtrs.count(invVarId) && dataPtrs.at(invVarId)) {
            float* outInvStd = static_cast<float*>(dataPtrs.at(invVarId));
            for (int k = 0; k < K; ++k) outInvStd[k] = invStd[k];
        }

        VGRE_LOG_DEBUG("cuDNNBN", "CONV_NORM training 2-pass: K=" + std::to_string(K) +
                       " N=" + std::to_string(N) + " HWout=" + std::to_string(HWout));
    }

    return CUDNN_STATUS_SUCCESS;
}

extern "C" {

// ── Descriptor lifecycle ───────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t type, void** descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    uintptr_t id = g_nextBackendId++;
    BackendNode& node = g_backendNodes[id];
    node.type = type;
    node.finalized = false;
    *descriptor = reinterpret_cast<void*>(id);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_SUCCESS;
    g_backendNodes.erase(reinterpret_cast<uintptr_t>(descriptor));
    return CUDNN_STATUS_SUCCESS;
}

// ── Attribute set/get ──────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendSetAttribute(void* descriptor,
                                       cudnnBackendAttributeName_t attrName,
                                       int attributeType, // 0=char,1=float,2=double,3=int32,4=int64,5=uint64,6=descriptor
                                       int64_t elementCount,
                                       const void* arrayOfElements) {
    if (!descriptor || !arrayOfElements || elementCount <= 0) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;

    BackendNode& node = it->second;
    std::vector<uint64_t>& vec = node.attrs[static_cast<int>(attrName)];
    vec.resize(static_cast<size_t>(elementCount));
    const size_t elemSize = backendAttributeElementSize(attributeType);
    const auto *bytes = static_cast<const uint8_t*>(arrayOfElements);
    for (int64_t i = 0; i < elementCount; ++i)
        vec[static_cast<size_t>(i)] = readBackendAttributeElement(bytes + static_cast<size_t>(i) * elemSize, elemSize);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendGetAttribute(void* descriptor,
                                       cudnnBackendAttributeName_t attrName,
                                       int attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) {
    if (!descriptor || !elementCount || !arrayOfElements) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;

    const BackendNode& node = it->second;
    auto attrIt = node.attrs.find(static_cast<int>(attrName));
    if (attrIt == node.attrs.end()) {
        *elementCount = 0;
        return CUDNN_STATUS_INVALID_VALUE;
    }

    int64_t n = static_cast<int64_t>(attrIt->second.size());
    *elementCount = n;
    int64_t copyN = std::min(requestedElementCount, n);
    const size_t elemSize = backendAttributeElementSize(attributeType);
    auto *bytes = static_cast<uint8_t*>(arrayOfElements);
    for (int64_t i = 0; i < copyN; ++i)
        writeBackendAttributeElement(bytes + static_cast<size_t>(i) * elemSize, elemSize,
                                     attrIt->second[static_cast<size_t>(i)]);
    return (requestedElementCount >= n) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_INVALID_VALUE;
}

// ── Finalize ─────────────────────────────────────────────────────────────────

cudnnStatus_t cudnnBackendFinalize(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    auto it = g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor));
    if (it == g_backendNodes.end()) return CUDNN_STATUS_INVALID_VALUE;
    BackendNode& node = it->second;
    node.finalized = true;

    // ── Engine heuristic finalization ─────────────────────────────────────────
    // VGRE has exactly one engine; populate ENGINEHEUR_RESULTS with a single
    // engine-config descriptor so callers of cudnnGetAttribute(ENGINEHEUR_RESULTS)
    // get a valid (non-empty) list without having to call cudnnBackendExecute.
    if (node.type == CUDNN_BACKEND_ENGINHEUR_DESCRIPTOR) {
        // Only populate if results not already set by the caller.
        int resultsAttr = static_cast<int>(CUDNN_ATTR_ENGINEHEUR_RESULTS);
        if (node.attrs.find(resultsAttr) == node.attrs.end()) {
            // Create a minimal engine-config descriptor and record its id.
            uintptr_t cfgId = g_nextBackendId++;
            BackendNode cfgNode;
            cfgNode.type      = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
            cfgNode.finalized = true;

            // Build a minimal ENGINE child descriptor.
            uintptr_t engId = g_nextBackendId++;
            BackendNode engNode;
            engNode.type      = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
            engNode.finalized = true;
            // Link engine to the operation graph stored in the heuristic node.
            uintptr_t opGraphId = getAttrUint64(&node,
                static_cast<int>(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH), 0);
            if (opGraphId) {
                engNode.attrs[static_cast<int>(CUDNN_ATTR_ENGINE_OPERATION_GRAPH)]
                    .push_back(static_cast<uint64_t>(opGraphId));
            }
            g_backendNodes[engId] = std::move(engNode);

            // Link engine into config.
            cfgNode.attrs[static_cast<int>(CUDNN_ATTR_ENGINECFG_ENGINE)]
                .push_back(static_cast<uint64_t>(engId));
            g_backendNodes[cfgId] = std::move(cfgNode);

            // Store config id as the single ENGINEHEUR_RESULTS entry.
            node.attrs[resultsAttr].push_back(static_cast<uint64_t>(cfgId));
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

// ── Initialize / Populate / Execute ──────────────────────────────────────────

cudnnStatus_t cudnnBackendInitialize(void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    if (g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor)) == g_backendNodes.end())
        return CUDNN_STATUS_INVALID_VALUE;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnBackendPopulate(void* /*handle*/, void* descriptor) {
    if (!descriptor) return CUDNN_STATUS_INVALID_VALUE;
    if (g_backendNodes.find(reinterpret_cast<uintptr_t>(descriptor)) == g_backendNodes.end())
        return CUDNN_STATUS_INVALID_VALUE;
    return CUDNN_STATUS_SUCCESS;
}

// Minimal cublas typedefs needed for matmul dispatch
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
static constexpr int CUBLAS_OP_N = 0;
static constexpr int CUDA_R_32F = 0;
static constexpr int CUBLAS_GEMM_DEFAULT = 0;
static constexpr int CUBLAS_STATUS_SUCCESS = 0;

// ── Forward declarations of legacy cuDNN shim functions ────────────────────
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnFilterDescriptor_t, const void*, cudnnConvolutionDescriptor_t, int, void*, size_t, const void*, cudnnTensorDescriptor_t, void*);
cublasStatus_t cublasGemmEx(cublasHandle_t, int, int, int, int, int, const void*, const void*, int, int, const void*, int, int, const void*, void*, int, int, int, int);
cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t, const void*, cudnnFilterDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, void*, size_t, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdFilterAlgo_t, void*, size_t, const void*, cudnnFilterDescriptor_t, void*);
cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnPoolingForward(cudnnHandle_t, cudnnPoolingDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t, cudnnPoolingDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t, int, int, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t, int, int, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnReduceTensor(cudnnHandle_t, cudnnReduceTensorDescriptor_t, void*, size_t, void*, size_t, const void*, cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t, int, const void*, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, const void*, double, void*, void*, double, void*, void*);
cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t, int, const void*, const void*, const void*, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*, void*, void*, double, const void*, const void*);
cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnHandle_t, cudnnTensorDescriptor_t, const void*, const void*, void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnHandle_t, cudnnTensorDescriptor_t, const void*, const void*, const void*, void*, cudnnTensorDescriptor_t, void*);
cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t, void* /*rnnDesc*/, int, cudnnTensorDescriptor_t*, const void*, cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, void* /*wDesc*/, const void*, cudnnTensorDescriptor_t*, void*, cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, void*, void*, size_t);

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle, void* plan, void* variantPack) {
    if (!plan || !variantPack) return CUDNN_STATUS_INVALID_VALUE;

    // Parse variant pack: map from backend node ID -> data pointer
    auto dataPtrs = parseVariantPack(variantPack);

    // Traverse plan -> engine config -> engine -> operation graph
    uintptr_t opSetId = traversePlanToOpSet(reinterpret_cast<uintptr_t>(plan));
    if (!opSetId) return CUDNN_STATUS_INVALID_VALUE;

    // Get list of operations
    std::vector<uintptr_t> ops = parseOperationSet(opSetId);
    if (ops.empty()) return CUDNN_STATUS_INVALID_VALUE;

    // Execute each operation in order (topological sort omitted for simple graphs)
    for (uintptr_t opId : ops) {
        const BackendNode* opNode = getNode(opId);
        if (!opNode) return CUDNN_STATUS_INVALID_VALUE;

        switch (opNode->type) {
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X);
            uintptr_t wId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y);
            uintptr_t cId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, 0.0f);
            void* xPtr = dataPtrs[xId];
            void* wPtr = dataPtrs[wId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !wPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            FilterDesc wDesc = buildFilterDesc(wId);
            TensorDesc yDesc = buildTensorDesc(yId);
            ConvDesc   cDesc = buildConvDesc(cId);
            cudnnStatus_t s = cudnnConvolutionForward(handle, &alpha, &xDesc, xPtr,
                                                        &wDesc, wPtr, &cDesc,
                                                        0, nullptr, 0, &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR: {
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY);
            uintptr_t wId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX);
            uintptr_t cId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, 0.0f);
            void* dyPtr = dataPtrs[dyId];
            void* wPtr  = dataPtrs[wId];
            void* dxPtr = dataPtrs[dxId];
            if (!dyPtr || !wPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc dyDesc = buildTensorDesc(dyId);
            FilterDesc wDesc  = buildFilterDesc(wId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            ConvDesc   cDesc  = buildConvDesc(cId);
            cudnnStatus_t s = cudnnConvolutionBackwardData(handle, &alpha, &wDesc, wPtr,
                                                             &dyDesc, dyPtr, &cDesc,
                                                             static_cast<cudnnConvolutionBwdDataAlgo_t>(0),
                                                             nullptr, 0, &beta, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: {
            uintptr_t xId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X);
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY);
            uintptr_t dwId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW);
            uintptr_t cId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, 0.0f);
            void* xPtr  = dataPtrs[xId];
            void* dyPtr = dataPtrs[dyId];
            void* dwPtr = dataPtrs[dwId];
            if (!xPtr || !dyPtr || !dwPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc  = buildTensorDesc(xId);
            TensorDesc dyDesc = buildTensorDesc(dyId);
            FilterDesc dwDesc = buildFilterDesc(dwId);
            ConvDesc   cDesc  = buildConvDesc(cId);
            cudnnStatus_t s = cudnnConvolutionBackwardFilter(handle, &alpha, &xDesc, xPtr,
                                                               &dyDesc, dyPtr, &cDesc,
                                                               static_cast<cudnnConvolutionBwdFilterAlgo_t>(0),
                                                               nullptr, 0, &beta, &dwDesc, dwPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_ACTIVATION_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            uintptr_t aId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_FORWARD_BETA, 0.0f);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            ActDesc* actDesc = reinterpret_cast<ActDesc*>(aId);
            if (!actDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnActivationForward(handle, actDesc, &alpha, &xDesc, xPtr,
                                                       &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_ACTIVATION_BACKWARD_DESCRIPTOR: {
            uintptr_t yId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_YDESC);
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC);
            uintptr_t aId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_BETA, 0.0f);
            void* yPtr  = dataPtrs[yId];
            void* dyPtr = dataPtrs[dyId];
            void* dxPtr = dataPtrs[dxId];
            if (!yPtr || !dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc yDesc  = buildTensorDesc(yId);
            TensorDesc dyDesc = buildTensorDesc(dyId);
            // Activation backward also needs xDesc; assume same shape as dxDesc
            TensorDesc xDesc  = buildTensorDesc(dxId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            ActDesc* actDesc = reinterpret_cast<ActDesc*>(aId);
            if (!actDesc) return CUDNN_STATUS_INVALID_VALUE;
            // x and dx share the same buffer in this Backend API mapping
            void* xPtr = dxPtr;
            cudnnStatus_t s = cudnnActivationBackward(handle, actDesc, &alpha,
                                                        &yDesc, yPtr, &dyDesc, dyPtr,
                                                        &xDesc, xPtr, &beta, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_YDESC);
            uintptr_t pId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_PDESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_FORWARD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_FORWARD_BETA, 0.0f);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            PoolDesc* poolDesc = reinterpret_cast<PoolDesc*>(pId);
            if (!poolDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnPoolingForward(handle, poolDesc, &alpha, &xDesc, xPtr,
                                                    &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_POOLING_BACKWARD_DESCRIPTOR: {
            // Pooling backward: computes the data gradient dx from upstream dy.
            // For max pooling, also needs the forward x (to reconstruct the argmax)
            // and optionally the forward y (the pooled output).  These are stored as
            // separate tensor IDs — BWD_XDESC (78) and BWD_YDESC (79) — so that the
            // caller can wire them from the same variant pack as the forward pass.
            uintptr_t dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_DYDESC);
            uintptr_t dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_DXDESC);
            uintptr_t xId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_XDESC);
            uintptr_t yId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_YDESC);
            uintptr_t pId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATIONPOOLING_BWD_PDESC);
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_BWD_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_POOLING_BWD_BETA, 0.0f);

            void* dyPtr = dataPtrs.count(dyId) ? dataPtrs.at(dyId) : nullptr;
            void* dxPtr = dataPtrs.count(dxId) ? dataPtrs.at(dxId) : nullptr;
            // Forward x — required for max-pooling argmax reconstruction; if absent, fall back to dx.
            void* xPtr  = (xId && dataPtrs.count(xId)) ? dataPtrs.at(xId)
                                                        : dxPtr;   // best-effort fallback for avg pool
            // Forward y — needed by some implementations; fall back to dy if absent.
            void* yPtr  = (yId && dataPtrs.count(yId)) ? dataPtrs.at(yId) : dyPtr;
            if (!dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc dyDesc = buildTensorDesc(dyId);
            TensorDesc dxDesc = buildTensorDesc(dxId);
            // xDesc: use forward x shape if available, otherwise mirror dx shape.
            TensorDesc xDesc  = (xId && xId != dxId) ? buildTensorDesc(xId) : buildTensorDesc(dxId);
            // yDesc: forward output y — must match dy shape; mirror dyDesc if absent.
            TensorDesc yDesc  = (yId && yId != dyId) ? buildTensorDesc(yId) : buildTensorDesc(dyId);

            PoolDesc* poolDesc = reinterpret_cast<PoolDesc*>(pId);
            if (!poolDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnPoolingBackward(handle, poolDesc, &alpha,
                                                     &yDesc, yPtr, &dyDesc, dyPtr,
                                                     &xDesc, xPtr, &beta, &dxDesc, dxPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR: {
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_SOFTMAX_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_SOFTMAX_BETA, 0.0f);
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            cudnnStatus_t s = cudnnSoftmaxForward(handle, 0, 0, &alpha, &xDesc, xPtr,
                                                    &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: {
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_REDUCTION_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_REDUCTION_BETA, 0.0f);
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            uintptr_t rId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_REDUCTION_DESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            ReduceTensorDesc* reduceDesc = reinterpret_cast<ReduceTensorDesc*>(rId);
            if (!reduceDesc) return CUDNN_STATUS_INVALID_VALUE;
            cudnnStatus_t s = cudnnReduceTensor(handle, reduceDesc,
                                                  nullptr, 0, nullptr, 0,
                                                  &alpha, &xDesc, xPtr, &beta, &yDesc, yPtr);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: {
            float alpha = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_MATMUL_ALPHA, 1.0f);
            float beta  = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_MATMUL_BETA, 0.0f);
            uintptr_t aId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_MATMUL_ADESC);
            uintptr_t bId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_MATMUL_BDESC);
            uintptr_t cId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_MATMUL_CDESC);
            void* aPtr = dataPtrs[aId];
            void* bPtr = dataPtrs[bId];
            void* cPtr = dataPtrs[cId];
            if (!aPtr || !bPtr || !cPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc aDesc = buildTensorDesc(aId);
            TensorDesc bDesc = buildTensorDesc(bId);
            TensorDesc cDesc = buildTensorDesc(cId);
            int m = cDesc.n;
            int n = cDesc.c;
            int k = aDesc.c;
            if (m == 0 || n == 0 || k == 0) return CUDNN_STATUS_INVALID_VALUE;
            cublasStatus_t s = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             m, n, k, &alpha,
                                             aPtr, CUDA_R_32F, m,
                                             bPtr, CUDA_R_32F, k,
                                             &beta, cPtr, CUDA_R_32F, m,
                                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (s != CUBLAS_STATUS_SUCCESS) return CUDNN_STATUS_INTERNAL_ERROR;
            break;
        }
        case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR: {
            // BN finalize statistics — two usage patterns:
            //
            // Pattern A (preferred, v8 backend): caller supplies accumulated ΣX / ΣX² per
            //   channel via CUDNN_ATTR_BN_FINALIZE_STATS_SUM_DESC / _SQ_SUM_DESC and
            //   optional running-mean/var update tensors.  Computes:
            //     mean[c]    = sumX[c] / accumCount
            //     var[c]     = sumXsq[c] / accumCount - mean[c]²
            //     runMean[c] = (1-f)*prevMean[c] + f*mean[c]
            //     runVar[c]  = (1-f)*prevVar[c]  + f*var[c]
            //     savedMean[c]   = mean[c]
            //     savedInvStd[c] = 1/sqrt(var[c] + epsilon)
            //
            // Pattern B (legacy / test fallback): caller supplies x via
            //   CUDNN_ATTR_OPERATION_ACTIVATION_XDESC; we compute statistics from raw x,
            //   apply BN forward, and write y via ACTIVATION_YDESC.

            uintptr_t sumId    = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_STATS_SUM_DESC);
            uintptr_t sqSumId  = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_STATS_SQ_SUM_DESC);

            const float* sumX   = (sumId   && dataPtrs.count(sumId))   ?
                                   static_cast<const float*>(dataPtrs.at(sumId))   : nullptr;
            const float* sumXsq = (sqSumId && dataPtrs.count(sqSumId)) ?
                                   static_cast<const float*>(dataPtrs.at(sqSumId)) : nullptr;

            if (sumX && sumXsq) {
                // ── Pattern A: statistics from pre-accumulated sums ───────────────
                uintptr_t pMeanId = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_PREV_RUNNING_MEAN_DESC);
                uintptr_t pVarId  = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_PREV_RUNNING_VAR_DESC);
                uintptr_t rMeanId = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC);
                uintptr_t rVarId  = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC);
                uintptr_t sMeanId = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_SAVED_MEAN_DESC);
                uintptr_t sInvId  = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_SAVED_INV_STD_DESC);

                const float* pMean = (pMeanId && dataPtrs.count(pMeanId)) ?
                                      static_cast<const float*>(dataPtrs.at(pMeanId)) : nullptr;
                const float* pVar  = (pVarId  && dataPtrs.count(pVarId))  ?
                                      static_cast<const float*>(dataPtrs.at(pVarId))  : nullptr;
                float* rMean = (rMeanId && dataPtrs.count(rMeanId)) ?
                                static_cast<float*>(dataPtrs.at(rMeanId)) : nullptr;
                float* rVar  = (rVarId  && dataPtrs.count(rVarId))  ?
                                static_cast<float*>(dataPtrs.at(rVarId))  : nullptr;
                float* sMean = (sMeanId && dataPtrs.count(sMeanId)) ?
                                static_cast<float*>(dataPtrs.at(sMeanId)) : nullptr;
                float* sInv  = (sInvId  && dataPtrs.count(sInvId))  ?
                                static_cast<float*>(dataPtrs.at(sInvId))  : nullptr;

                TensorDesc sumDesc = buildTensorDesc(sumId);
                int C = std::max(sumDesc.c, std::max(sumDesc.n, 1));

                uintptr_t accumAttr = getAttrUint64(opNode, CUDNN_ATTR_BN_FINALIZE_ACCUM_COUNT);
                double accumCount   = (accumAttr > 0) ? static_cast<double>(accumAttr)
                                                      : static_cast<double>(std::max(sumDesc.n, 1));
                float expAvgFactor  = getAttrFloat(opNode, CUDNN_ATTR_BN_FINALIZE_EXP_AVG_FACTOR, 0.0f);
                float epsilon       = getAttrFloat(opNode, CUDNN_ATTR_BN_FINALIZE_EPSILON, 1e-5f);

                for (int c = 0; c < C; ++c) {
                    double sx   = static_cast<double>(sumX[c]);
                    double sx2  = static_cast<double>(sumXsq[c]);
                    double mean = sx  / accumCount;
                    double var  = sx2 / accumCount - mean * mean;
                    if (var < 0.0) var = 0.0;

                    if (sMean) sMean[c] = static_cast<float>(mean);
                    if (sInv)  sInv[c]  = static_cast<float>(1.0 / std::sqrt(var + epsilon));
                    if (rMean)
                        rMean[c] = pMean ? (1.0f - expAvgFactor) * pMean[c] + expAvgFactor * static_cast<float>(mean)
                                         : static_cast<float>(mean);
                    if (rVar)
                        rVar[c]  = pVar  ? (1.0f - expAvgFactor) * pVar[c]  + expAvgFactor * static_cast<float>(var)
                                         : static_cast<float>(var);
                }
            } else {
                // ── Pattern B: raw input x — compute statistics and apply BN forward ──
                uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
                uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
                const float* xPtr = xId ? static_cast<const float*>(
                    dataPtrs.count(xId) ? dataPtrs.at(xId) : nullptr) : nullptr;
                float* yPtr = yId ? static_cast<float*>(
                    dataPtrs.count(yId) ? dataPtrs.at(yId) : nullptr) : nullptr;
                if (!xPtr) return CUDNN_STATUS_INVALID_VALUE;

                TensorDesc xd = buildTensorDesc(xId);
                int N = std::max(xd.n, 1), C = std::max(xd.c, 1);
                int H = std::max(xd.h, 1), W = std::max(xd.w, 1);
                int NHW = N * H * W;
                float eps = 1e-5f;

                // Compute per-channel mean and variance from x, then apply identity BN
                // (γ=1, β=0) and write to y if provided.
                for (int c = 0; c < C; ++c) {
                    double sum = 0.0, sq = 0.0;
                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                double v = static_cast<double>(xPtr[((n*C+c)*H+h)*W+w]);
                                sum += v; sq += v * v;
                            }
                    double mean = sum / NHW;
                    double var  = sq  / NHW - mean * mean;
                    if (var < 0.0) var = 0.0;
                    float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
                    if (yPtr) {
                        for (int n = 0; n < N; ++n)
                            for (int h = 0; h < H; ++h)
                                for (int w = 0; w < W; ++w) {
                                    int idx = ((n*C+c)*H+h)*W+w;
                                    yPtr[idx] = (xPtr[idx] - static_cast<float>(mean)) * inv;
                                }
                    }
                }
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR: {
            // Try new dedicated NORM_FWD attrs first; fall back to legacy activation attr slots
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, 0);
            if (!xId) xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, 0);
            if (!yId) yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);

            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc xDesc = buildTensorDesc(xId);
            int N = xDesc.n, C = xDesc.c, H = xDesc.h, W = xDesc.w;
            int HW = H * W;
            int phase = static_cast<int>(getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, 0));

            uintptr_t scaleId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, 0);
            uintptr_t biasId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, 0);
            uintptr_t meanId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, 0);
            uintptr_t invVarId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_INV_VAR_DESC, 0);
            uintptr_t epsId    = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, 0);

            float epsilon = 1e-5f;
            if (epsId && dataPtrs.count(epsId) && dataPtrs.at(epsId))
                epsilon = *static_cast<const float*>(dataPtrs.at(epsId));

            const float* xf    = static_cast<const float*>(xPtr);
            float*       yf    = static_cast<float*>(yPtr);
            const float* scale = (scaleId && dataPtrs.count(scaleId)) ?
                                  static_cast<const float*>(dataPtrs.at(scaleId)) : nullptr;
            const float* bias  = (biasId  && dataPtrs.count(biasId))  ?
                                  static_cast<const float*>(dataPtrs.at(biasId))  : nullptr;

            std::vector<float> defScale, defBias;
            if (!scale) { defScale.assign(C, 1.0f); scale = defScale.data(); }
            if (!bias)  { defBias.assign(C, 0.0f);  bias  = defBias.data();  }

            std::vector<float> computedMean(C, 0.0f), computedInvStd(C, 0.0f);

            if (phase == 0) {
                // ── INFERENCE: use provided running mean + running variance ────
                const float* runMean = (meanId   && dataPtrs.count(meanId))   ?
                                        static_cast<const float*>(dataPtrs.at(meanId))   : nullptr;
                const float* runVar  = (invVarId && dataPtrs.count(invVarId)) ?
                                        static_cast<const float*>(dataPtrs.at(invVarId)) : nullptr;
                if (runMean && runVar) {
                    for (int c = 0; c < C; ++c)
                        computedInvStd[c] = 1.0f / std::sqrt(runVar[c] + epsilon);
                    for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c) {
                        float gisv = scale[c] * computedInvStd[c];
                        float shift = bias[c] - gisv * runMean[c];
                        for (int hw = 0; hw < HW; ++hw) {
                            int idx = (n * C + c) * HW + hw;
                            yf[idx] = gisv * xf[idx] + shift;
                        }
                    }
                } else {
                    // No running stats: identity normalization (γ=1, β=0, μ=0, σ²=1)
                    for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c) {
                        float sc = scale[c], bs = bias[c];
                        for (int hw = 0; hw < HW; ++hw) {
                            int idx = (n * C + c) * HW + hw;
                            yf[idx] = sc * xf[idx] + bs;
                        }
                    }
                }
            } else {
                // ── TRAINING: compute per-channel mean and variance ────────────
                float inv_nhw = 1.0f / static_cast<float>(N * HW);

                for (int c = 0; c < C; ++c) {
                    double sum = 0.0;
                    for (int n = 0; n < N; ++n)
                    for (int hw = 0; hw < HW; ++hw)
                        sum += xf[(n * C + c) * HW + hw];
                    computedMean[c] = static_cast<float>(sum * inv_nhw);
                }
                for (int c = 0; c < C; ++c) {
                    double var = 0.0;
                    float muc = computedMean[c];
                    for (int n = 0; n < N; ++n)
                    for (int hw = 0; hw < HW; ++hw) {
                        float d = xf[(n * C + c) * HW + hw] - muc;
                        var += d * d;
                    }
                    computedInvStd[c] = 1.0f / std::sqrt(static_cast<float>(var * inv_nhw) + epsilon);
                }

                for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    float gisv  = scale[c] * computedInvStd[c];
                    float shift = bias[c] - gisv * computedMean[c];
                    for (int hw = 0; hw < HW; ++hw) {
                        int idx = (n * C + c) * HW + hw;
                        yf[idx] = gisv * xf[idx] + shift;
                    }
                }

                // Write saved mean and saved inv_std to output tensors if provided
                if (meanId && dataPtrs.count(meanId) && dataPtrs.at(meanId)) {
                    float* outMean = static_cast<float*>(dataPtrs.at(meanId));
                    for (int c = 0; c < C; ++c) outMean[c] = computedMean[c];
                }
                if (invVarId && dataPtrs.count(invVarId) && dataPtrs.at(invVarId)) {
                    float* outInvStd = static_cast<float*>(dataPtrs.at(invVarId));
                    for (int c = 0; c < C; ++c) outInvStd[c] = computedInvStd[c];
                }
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: {
            // Normalization backward: computes dx, dγ, dβ for BN / LayerNorm / InstanceNorm.
            // Mirrors the NORM_FORWARD backward pass (Ioffe & Szegedy, Ba et al.).
            //
            // For BN_spatial over [N, C, H, W] with m = N*H*W elements per channel:
            //   x_hat[i] = (x[i] - mean[c]) * invStd[c]
            //   dβ[c]    = Σ_i dy[i]
            //   dγ[c]    = Σ_i dy[i] * x_hat[i]
            //   dx[i]    = γ[c]*invStd[c]/m * (m*dy[i] - dβ[c] - x_hat[i]*dγ[c])
            //
            // For LayerNorm / InstanceNorm, the reduction axis changes:
            //   LN:  m = C*H*W  (reduce over all channels and spatial)
            //   IN:  m = H*W    (reduce per sample per channel)

            // Resolve tensor IDs — try dedicated BWD attrs first, then legacy activation attrs.
            uintptr_t xId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_XDESC);
            if (!xId) xId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t dyId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC);
            if (!dyId) dyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC);
            uintptr_t dxId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC);
            if (!dxId) dxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC);

            const float* xPtr  = xId  ? static_cast<const float*>(
                dataPtrs.count(xId)  ? dataPtrs.at(xId)  : nullptr) : nullptr;
            const float* dyPtr = dyId ? static_cast<const float*>(
                dataPtrs.count(dyId) ? dataPtrs.at(dyId) : nullptr) : nullptr;
            float* dxPtr       = dxId ? static_cast<float*>(
                dataPtrs.count(dxId) ? dataPtrs.at(dxId) : nullptr) : nullptr;
            if (!xPtr || !dyPtr || !dxPtr) return CUDNN_STATUS_INVALID_VALUE;

            uintptr_t scId    = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC);
            uintptr_t meanId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC);
            uintptr_t invId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_INV_VAR_DESC);
            uintptr_t dscId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC);
            uintptr_t dbId    = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC);
            uintptr_t epsId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC);

            const float* scale   = (scId   && dataPtrs.count(scId))   ?
                                    static_cast<const float*>(dataPtrs.at(scId))   : nullptr;
            const float* savMean = (meanId && dataPtrs.count(meanId)) ?
                                    static_cast<const float*>(dataPtrs.at(meanId)) : nullptr;
            const float* savInv  = (invId  && dataPtrs.count(invId))  ?
                                    static_cast<const float*>(dataPtrs.at(invId))  : nullptr;
            float* dScale = (dscId && dataPtrs.count(dscId)) ?
                             static_cast<float*>(dataPtrs.at(dscId)) : nullptr;
            float* dBias  = (dbId  && dataPtrs.count(dbId))  ?
                             static_cast<float*>(dataPtrs.at(dbId))  : nullptr;
            float epsilon = 1e-5f;
            if (epsId && dataPtrs.count(epsId) && dataPtrs.at(epsId))
                epsilon = *static_cast<const float*>(dataPtrs.at(epsId));

            TensorDesc xDesc = buildTensorDesc(xId);
            int N = xDesc.n, C = xDesc.c, H = xDesc.h, W = xDesc.w;
            if (H <= 0) H = 1; if (W <= 0) W = 1;
            int HW = H * W;

            int normMode = static_cast<int>(
                getAttrUint64(opNode, CUDNN_ATTR_OPERATION_NORM_BWD_MODE));
            // 0=BN_spatial, 1=LayerNorm, 2=InstanceNorm

            // If saved stats absent, recompute from x.
            std::vector<float> tmpMean, tmpInv;
            if (!savMean || !savInv) {
                if (normMode == 0) {
                    // BN: reduce over N*H*W per channel
                    tmpMean.assign(C, 0.0f); tmpInv.assign(C, 0.0f);
                    int NHW = N * HW;
                    for (int n = 0; n < N; ++n)
                        for (int c = 0; c < C; ++c)
                            for (int hw = 0; hw < HW; ++hw)
                                tmpMean[c] += xPtr[((n*C+c)*H + hw/W)*W + hw%W];
                    for (int c = 0; c < C; ++c) tmpMean[c] /= NHW;
                    for (int n = 0; n < N; ++n)
                        for (int c = 0; c < C; ++c)
                            for (int hw = 0; hw < HW; ++hw) {
                                float d = xPtr[((n*C+c)*H + hw/W)*W + hw%W] - tmpMean[c];
                                tmpInv[c] += d * d;
                            }
                    for (int c = 0; c < C; ++c)
                        tmpInv[c] = 1.0f / std::sqrt(tmpInv[c] / NHW + epsilon);
                    savMean = tmpMean.data(); savInv = tmpInv.data();
                } else {
                    // LN/IN: reuse the BN-bwd-weights path below; set dummy all-zero mean / all-one invStd
                    tmpMean.assign(C, 0.0f); tmpInv.assign(C, 1.0f);
                    savMean = tmpMean.data(); savInv = tmpInv.data();
                }
            }

            // Allocate output grad buffers if not provided
            std::vector<float> tmpDscale, tmpDbias;
            if (!dScale) { tmpDscale.assign(C, 0.0f); dScale = tmpDscale.data(); }
            if (!dBias)  { tmpDbias.assign(C, 0.0f);  dBias  = tmpDbias.data();  }

            if (normMode == 0) {
                // ── Batch Norm spatial backward (Ioffe-Szegedy 2015) ────────────
                int NHW = N * HW;
                std::fill(dScale, dScale + C, 0.0f);
                std::fill(dBias,  dBias  + C, 0.0f);
                // Pass 1: accumulate dγ and dβ
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - savMean[c]) * savInv[c];
                                dScale[c] += dyPtr[idx] * xhat;
                                dBias[c]  += dyPtr[idx];
                            }
                // Pass 2: compute dx
                float mf = static_cast<float>(NHW);
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c) {
                        float g   = scale ? scale[c] : 1.0f;
                        float inv = savInv[c];
                        float db  = dBias[c], dg = dScale[c];
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - savMean[c]) * inv;
                                dxPtr[idx] = g * inv / mf * (mf * dyPtr[idx] - db - xhat * dg);
                            }
                    }
            } else if (normMode == 1) {
                // ── Layer Norm backward (Ba et al. 2016) ──────────────────────
                // LN reduces over the entire feature vector [C*H*W] per sample n.
                // γ and β have shape [C,H,W] (or [C]) per the standard VGRE encoding.
                int CHW = C * HW;
                std::fill(dScale, dScale + C, 0.0f);
                std::fill(dBias,  dBias  + C, 0.0f);
                for (int n = 0; n < N; ++n) {
                    // Compute mean and inv-std for this sample over [C*H*W]
                    double sum = 0.0, sq = 0.0;
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                double v = xPtr[((n*C+c)*H+h)*W+w];
                                sum += v; sq += v * v;
                            }
                    float mean_n = static_cast<float>(sum / CHW);
                    float inv_n  = static_cast<float>(1.0 / std::sqrt(sq / CHW - (sum/CHW)*(sum/CHW) + epsilon));

                    // dβ[c] += Σ_{n,h,w} dy; dγ[c] += Σ_{n,h,w} dy*xhat
                    // Per-sample data gradient
                    float db_n = 0.0f, dg_n = 0.0f;
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - mean_n) * inv_n;
                                float g    = scale ? scale[c] : 1.0f;
                                dScale[c] += dyPtr[idx] * xhat;
                                dBias[c]  += dyPtr[idx];
                                db_n += dyPtr[idx] * g;
                                dg_n += dyPtr[idx] * g * xhat;
                            }
                    float mf = static_cast<float>(CHW);
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - mean_n) * inv_n;
                                float g    = scale ? scale[c] : 1.0f;
                                dxPtr[idx] = g * inv_n / mf *
                                    (mf * dyPtr[idx] - db_n - xhat * dg_n);
                            }
                }
            } else {
                // ── Instance Norm backward (mode==2): reduce over H*W per (n,c) ──
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c) {
                        double sum = 0.0, sq = 0.0;
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                double v = xPtr[((n*C+c)*H+h)*W+w];
                                sum += v; sq += v * v;
                            }
                        float mean_nc = static_cast<float>(sum / HW);
                        float inv_nc  = static_cast<float>(1.0 / std::sqrt(sq / HW - (sum/HW)*(sum/HW) + epsilon));

                        float db_nc = 0.0f, dg_nc = 0.0f;
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - mean_nc) * inv_nc;
                                float g    = scale ? scale[c] : 1.0f;
                                dScale[c] += dyPtr[idx] * xhat;
                                dBias[c]  += dyPtr[idx];
                                db_nc += dyPtr[idx] * g;
                                dg_nc += dyPtr[idx] * g * xhat;
                            }
                        float mf = static_cast<float>(HW);
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - mean_nc) * inv_nc;
                                float g    = scale ? scale[c] : 1.0f;
                                dxPtr[idx] = g * inv_nc / mf *
                                    (mf * dyPtr[idx] - db_nc - xhat * dg_nc);
                            }
                    }
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_RNN_DESCRIPTOR: {
            // Resolve X/Y tensor IDs — try RNN-specific attrs first, fall back to activation attrs.
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_XDESC);
            if (!xId) xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_YDESC);
            if (!yId) yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            uintptr_t wId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_WDESC);
            uintptr_t hxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_HXDESC);
            uintptr_t hyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_HYDESC);
            uintptr_t cxId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_CXDESC);
            uintptr_t cyId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_CYDESC);

            void* xPtr  = dataPtrs.count(xId)  ? dataPtrs[xId]  : nullptr;
            void* yPtr  = dataPtrs.count(yId)  ? dataPtrs[yId]  : nullptr;
            void* wPtr  = dataPtrs.count(wId)  ? dataPtrs[wId]  : nullptr;
            void* hxPtr = dataPtrs.count(hxId) ? dataPtrs[hxId] : nullptr;
            void* hyPtr = dataPtrs.count(hyId) ? dataPtrs[hyId] : nullptr;
            void* cxPtr = dataPtrs.count(cxId) ? dataPtrs[cxId] : nullptr;
            void* cyPtr = dataPtrs.count(cyId) ? dataPtrs[cyId] : nullptr;
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;

            // Retrieve (or construct) the RNN descriptor.
            // CUDNN_ATTR_OPERATION_RNN_HANDLE stores the cudnnRNNDescriptor_t as uint64.
            uintptr_t rnnHandle = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_HANDLE);
            RNNDesc* rnnPtr = nullptr;
            RNNDesc  tmpDesc{};   // used if the caller didn't set a handle

            if (rnnHandle) {
                rnnPtr = reinterpret_cast<RNNDesc*>(rnnHandle);
            } else {
                // Derive dimensions from backend tensor descriptors.
                // Backend encoding: dims[0]=B, dims[1]=T (seq), dims[2]=feature_size.
                // cudnnRNNForwardInference reads I = xt->c*xt->h*xt->w from xDescArr[0],
                // so each element of xDescArr must carry [B, I, 1, 1].
                // We detect whether the tensor is 3-D [B,T,I] or 2-D [B,I] by checking
                // whether the third dimension (h) is > 0.
                const BackendNode* xNode = getNode(xId);
                const BackendNode* yNode = getNode(yId);
                auto* xDims = xNode ? getAttrVec(xNode, CUDNN_ATTR_TENSOR_DIMENSIONS) : nullptr;
                auto* yDims = yNode ? getAttrVec(yNode, CUDNN_ATTR_TENSOR_DIMENSIONS) : nullptr;

                int B = 1, T = 1, inputSize = 128, hiddenSize = 128;
                if (xDims && xDims->size() >= 3) {
                    B = static_cast<int>((*xDims)[0]);
                    T = static_cast<int>((*xDims)[1]);
                    inputSize = static_cast<int>((*xDims)[2]);
                } else if (xDims && xDims->size() >= 2) {
                    B = static_cast<int>((*xDims)[0]);
                    inputSize = static_cast<int>((*xDims)[1]);
                }
                if (yDims && yDims->size() >= 3)
                    hiddenSize = static_cast<int>((*yDims)[2]);
                else if (yDims && yDims->size() >= 2)
                    hiddenSize = static_cast<int>((*yDims)[1]);

                // T may also be set explicitly via CUDNN_ATTR_OPERATION_RNN_SEQ_LEN.
                uintptr_t seqAttr = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_SEQ_LEN);
                if (seqAttr) T = static_cast<int>(seqAttr);

                tmpDesc.hiddenSize = hiddenSize;
                tmpDesc.numLayers  = 1;
                tmpDesc.inputSize  = inputSize;
                tmpDesc.mode       = CUDNN_RNN_TANH;
                tmpDesc.peephole   = false;
                (void)B;   // B is read from the per-timestep xDesc below
                rnnPtr = &tmpDesc;
            }

            // Sequence length: CUDNN_ATTR_OPERATION_RNN_SEQ_LEN overrides tensor shape.
            int T = 1;
            {
                uintptr_t seqAttr = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNN_SEQ_LEN);
                if (seqAttr) {
                    T = static_cast<int>(seqAttr);
                } else {
                    // Try to read T from the 3-D input tensor (dim[1]).
                    const BackendNode* xNode = getNode(xId);
                    auto* xDims = xNode ? getAttrVec(xNode, CUDNN_ATTR_TENSOR_DIMENSIONS) : nullptr;
                    if (xDims && xDims->size() >= 3)
                        T = static_cast<int>((*xDims)[1]);
                }
            }

            // Build per-timestep tensor descriptor arrays required by the legacy API.
            // Each element is [B, inputSize, 1, 1] — the time axis is unrolled into the
            // T-element arrays, with xPtr/yPtr laid out as contiguous [T][B][feature].
            TensorDesc perStepXDesc{};
            {
                const BackendNode* xNode = getNode(xId);
                auto* xDims = xNode ? getAttrVec(xNode, CUDNN_ATTR_TENSOR_DIMENSIONS) : nullptr;
                if (xDims && xDims->size() >= 3) {
                    perStepXDesc.n = static_cast<int>((*xDims)[0]);   // B
                    perStepXDesc.c = static_cast<int>((*xDims)[2]);   // inputSize
                    perStepXDesc.h = 1; perStepXDesc.w = 1;
                } else {
                    TensorDesc tmp = buildTensorDesc(xId);
                    perStepXDesc = tmp;
                }
            }
            TensorDesc perStepYDesc{};
            {
                const BackendNode* yNode = getNode(yId);
                auto* yDims = yNode ? getAttrVec(yNode, CUDNN_ATTR_TENSOR_DIMENSIONS) : nullptr;
                if (yDims && yDims->size() >= 3) {
                    perStepYDesc.n = static_cast<int>((*yDims)[0]);   // B
                    perStepYDesc.c = static_cast<int>((*yDims)[2]);   // hiddenSize
                    perStepYDesc.h = 1; perStepYDesc.w = 1;
                } else {
                    perStepYDesc = buildTensorDesc(yId);
                }
            }
            std::vector<cudnnTensorDescriptor_t> xDescArr(static_cast<size_t>(T), &perStepXDesc);
            std::vector<cudnnTensorDescriptor_t> yDescArr(static_cast<size_t>(T), &perStepYDesc);

            // Allocate workspace sized for training-reserve layout.
            int H  = rnnPtr->hiddenSize;
            int nL = rnnPtr->numLayers;
            int B  = perStepXDesc.n;
            int fs = (rnnPtr->mode == CUDNN_LSTM) ? 5 : (rnnPtr->mode == CUDNN_GRU ? 3 : 0);
            size_t wsSz = static_cast<size_t>(nL) * T * B * H * fs * sizeof(float);
            if (wsSz == 0) wsSz = static_cast<size_t>(nL) * T * B * H * sizeof(float);
            std::vector<float> wsVec(wsSz / sizeof(float) + 1, 0.f);

            cudnnStatus_t s = cudnnRNNForwardInference(
                handle, rnnPtr, T, xDescArr.data(), xPtr,
                nullptr, hxPtr, nullptr, cxPtr, nullptr, wPtr,
                yDescArr.data(), yPtr, nullptr, hyPtr, nullptr, cyPtr,
                wsVec.data(), wsSz);
            if (s != CUDNN_STATUS_SUCCESS) return s;
            break;
        }
        case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR: {
            // Simple concatenation: copy inputs contiguously into output buffer
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            TensorDesc yDesc = buildTensorDesc(yId);
            size_t xBytes = static_cast<size_t>(xDesc.n) * xDesc.c * xDesc.h * xDesc.w * sizeof(float);
            size_t yBytes = static_cast<size_t>(yDesc.n) * yDesc.c * yDesc.h * yDesc.w * sizeof(float);
            memcpy(yPtr, xPtr, xBytes);
            // Second input (if present) appended after first
            uintptr_t x2Id = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X); // reuse attr slot
            void* x2Ptr = dataPtrs.count(x2Id) ? dataPtrs[x2Id] : nullptr;
            if (x2Ptr && x2Id != xId)
                memcpy(static_cast<char*>(yPtr) + xBytes, x2Ptr, yBytes - xBytes);
            break;
        }
        case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR: {
            // Signal is a synchronization barrier; no-op in single-threaded CPU emulation
            break;
        }
        case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR: {
            // Generate statistics (mean, variance) per-channel
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;
            TensorDesc xDesc = buildTensorDesc(xId);
            int N = xDesc.n, C = xDesc.c, HW = xDesc.h * xDesc.w;
            const float* xf = static_cast<const float*>(xPtr);
            float* meanOut = static_cast<float*>(yPtr);
            float* varOut  = meanOut + C;
            #ifdef _OPENMP
            #pragma omp parallel for if (C > 4)
            #endif
            for (int c = 0; c < C; ++c) {
                double sum = 0.0, sq = 0.0;
                for (int n = 0; n < N; ++n)
                for (int hw = 0; hw < HW; ++hw) {
                    float v = xf[((n * C + c) * xDesc.h + (hw / xDesc.w)) * xDesc.w + (hw % xDesc.w)];
                    sum += v;
                    sq += v * v;
                }
                double count = static_cast<double>(N * HW);
                meanOut[c] = static_cast<float>(sum / count);
                varOut[c]  = static_cast<float>(sq / count - meanOut[c] * meanOut[c]);
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR: {
            // Batch normalisation weight-gradient (dγ, dβ) and optional data-gradient (dx).
            // Requires the saved mean and saved inv-std from the matching forward training pass.
            //
            // Per Ioffe & Szegedy (2015), for spatial BN over [N, C, H, W]:
            //   x_hat[n,c,h,w] = (x[n,c,h,w] - mean[c]) * invStd[c]
            //   dβ[c]     = Σ_{n,h,w} dy[n,c,h,w]
            //   dγ[c]     = Σ_{n,h,w} dy[n,c,h,w] · x_hat[n,c,h,w]
            //
            // dx (data gradient for fused conv+BN backward) follows the standard BN backward:
            //   m  = N*H*W
            //   dx = γ * invStd * (m*dy - dβ - x_hat*dγ) / m

            // Resolve tensor IDs — try dedicated BWD_WEIGHTS attrs first, fall back to activation attrs.
            uintptr_t xId     = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_X_DESC);
            if (!xId) xId     = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_XDESC);
            uintptr_t dyId    = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_DY_DESC);
            if (!dyId) dyId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DYDESC);
            uintptr_t dxId    = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_DX_DESC);
            if (!dxId) dxId   = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_BWD_DXDESC);
            uintptr_t scId    = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_SCALE_DESC);
            uintptr_t meanId  = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_MEAN_DESC);
            uintptr_t invId   = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_INVSTD_DESC);
            uintptr_t dscId   = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_DSCALE_DESC);
            uintptr_t dbId    = getAttrUint64(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_DBIAS_DESC);

            const float* xPtr  = xId  ? static_cast<const float*>(
                                   dataPtrs.count(xId)  ? dataPtrs.at(xId)  : nullptr) : nullptr;
            const float* dyPtr = dyId ? static_cast<const float*>(
                                   dataPtrs.count(dyId) ? dataPtrs.at(dyId) : nullptr) : nullptr;
            float* dxPtr       = dxId ? static_cast<float*>(
                                   dataPtrs.count(dxId) ? dataPtrs.at(dxId) : nullptr) : nullptr;
            if (!xPtr || !dyPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc xDesc = buildTensorDesc(xId);
            int N = xDesc.n, C = xDesc.c, H = xDesc.h, W = xDesc.w;
            int HW  = H * W;
            int NHW = N * HW;

            float epsilon = getAttrFloat(opNode, CUDNN_ATTR_BN_BWD_WEIGHTS_EPSILON, 1e-5f);

            // Load optional per-channel parameters — fall back to identity/zeros if absent.
            const float* scale   = (scId   && dataPtrs.count(scId))   ?
                                    static_cast<const float*>(dataPtrs.at(scId))   : nullptr;
            const float* savMean = (meanId && dataPtrs.count(meanId)) ?
                                    static_cast<const float*>(dataPtrs.at(meanId)) : nullptr;
            const float* savInv  = (invId  && dataPtrs.count(invId))  ?
                                    static_cast<const float*>(dataPtrs.at(invId))  : nullptr;
            float*       dScale  = (dscId  && dataPtrs.count(dscId))  ?
                                    static_cast<float*>(dataPtrs.at(dscId))        : nullptr;
            float*       dBias   = (dbId   && dataPtrs.count(dbId))   ?
                                    static_cast<float*>(dataPtrs.at(dbId))         : nullptr;

            // If saved statistics are absent, fall back to the full Ioffe–Szegedy formula
            // (recompute mean/variance on the fly — slower but always correct).
            std::vector<float> tmpMean, tmpInv;
            if (!savMean || !savInv) {
                tmpMean.assign(C, 0.0f);
                tmpInv.assign(C, 0.0f);
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w)
                                tmpMean[c] += xPtr[((n*C+c)*H+h)*W+w];
                for (int c = 0; c < C; ++c) tmpMean[c] /= static_cast<float>(NHW);
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                float d = xPtr[((n*C+c)*H+h)*W+w] - tmpMean[c];
                                tmpInv[c] += d * d;
                            }
                for (int c = 0; c < C; ++c)
                    tmpInv[c] = 1.0f / std::sqrt(tmpInv[c] / static_cast<float>(NHW) + epsilon);
                savMean = tmpMean.data();
                savInv  = tmpInv.data();
            }

            // Compute dγ, dβ, and (optionally) dx.
            std::vector<float> tmpDscale, tmpDbias;
            if (!dScale) { tmpDscale.assign(C, 0.0f); dScale = tmpDscale.data(); }
            if (!dBias)  { tmpDbias.assign(C, 0.0f);  dBias  = tmpDbias.data();  }

            std::fill(dScale, dScale + C, 0.0f);
            std::fill(dBias,  dBias  + C, 0.0f);

            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            int idx = ((n*C+c)*H+h)*W+w;
                            float xhat = (xPtr[idx] - savMean[c]) * savInv[c];
                            dScale[c] += dyPtr[idx] * xhat;
                            dBias[c]  += dyPtr[idx];
                        }

            // Data gradient dx (used in fused conv+BN backward).
            if (dxPtr) {
                float mf = static_cast<float>(NHW);
                float gammaSc = scale ? 1.0f : 1.0f; // γ is applied outside if absent
                for (int c = 0; c < C; ++c) {
                    float g  = scale ? scale[c] : 1.0f;
                    float inv = savInv[c];
                    float db = dBias[c], dg = dScale[c];
                    for (int n = 0; n < N; ++n)
                        for (int h = 0; h < H; ++h)
                            for (int w = 0; w < W; ++w) {
                                int idx = ((n*C+c)*H+h)*W+w;
                                float xhat = (xPtr[idx] - savMean[c]) * inv;
                                dxPtr[idx] = g * inv / mf * (mf * dyPtr[idx] - db - xhat * dg);
                            }
                }
                (void)gammaSc;
            }
            break;
        }
        case CUDNN_BACKEND_OPERATION_ATTENTION_DESCRIPTOR: {
            // FlashAttention-2 tiled scaled dot-product attention.
            // Algorithm: Dao et al., "FlashAttention-2" (NeurIPS 2023).
            //
            // Standard SDPA: O = softmax(Q·K^T / scale) · V
            //   Memory: O(N²) for the N×N score matrix.
            //
            // FlashAttention tiling: process Q in Br-row blocks, stream K/V in Bc-col blocks.
            //   Online softmax: maintain running (m_i, l_i, O_i) per Q-block row:
            //     m_new = max(m_old, rowmax(S_ij))
            //     O_new = exp(m_old - m_new)·O_old + exp(S_ij - m_new)·V_j  (no division yet)
            //     l_new = exp(m_old - m_new)·l_old + rowsum(exp(S_ij - m_new))
            //   Final: O_i /= l_i
            //   Memory: O((Br + Bc) · head_dim) instead of O(N · head_dim).
            uintptr_t qId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_QDESC);
            uintptr_t kId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_KDESC);
            uintptr_t vId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_VDESC);
            uintptr_t oId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_ODESC);
            void *qPtr = dataPtrs[qId], *kPtr = dataPtrs[kId];
            void *vPtr = dataPtrs[vId], *oPtr = dataPtrs[oId];
            if (!qPtr || !kPtr || !vPtr || !oPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc qDesc = buildTensorDesc(qId);
            TensorDesc kDesc = buildTensorDesc(kId);
            TensorDesc vDesc = buildTensorDesc(vId);

            int batch    = qDesc.n;
            int heads    = qDesc.c;
            int seqQ     = qDesc.h;
            int head_dim = qDesc.w;
            int seqK     = kDesc.h;

            float attnScale = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_ATTENTION_SCALE,
                                            1.0f / std::sqrt(static_cast<float>(head_dim)));
            bool causal = (getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ATTENTION_CAUSAL_MASK) != 0);

            const float *Q = static_cast<const float*>(qPtr);
            const float *K = static_cast<const float*>(kPtr);
            const float *V = static_cast<const float*>(vPtr);
            float       *O = static_cast<float*>(oPtr);

            // Tile sizes: 64 rows × 64 cols covers typical head_dim (64-128).
            // At head_dim=128: S_tile = 64×64×4 = 16 KB, O_tile = 64×128×4 = 32 KB → fits L1.
            static constexpr int Br = 64, Bc = 64;

            // Per-thread accumulators (reused across heads).
            std::vector<float> m_i(Br), l_i(Br);
            std::vector<float> O_i(static_cast<size_t>(Br) * head_dim);
            std::vector<float> S_ij(static_cast<size_t>(Br) * Bc);
            std::vector<float> P_ij(static_cast<size_t>(Br) * Bc);

            for (int b = 0; b < batch; ++b)
            for (int h = 0; h < heads; ++h) {
                const float* Qh = Q + (b * heads + h) * seqQ * head_dim;
                const float* Kh = K + (b * heads + h) * seqK * head_dim;
                const float* Vh = V + (b * heads + h) * seqK * head_dim;
                float*       Oh = O + (b * heads + h) * seqQ * head_dim;

                for (int i0 = 0; i0 < seqQ; i0 += Br) {
                    int br = std::min(Br, seqQ - i0);

                    // Initialise per-Q-block running stats.
                    for (int i = 0; i < br; ++i) { m_i[i] = -1e30f; l_i[i] = 0.0f; }
                    for (int x = 0; x < br * head_dim; ++x) O_i[x] = 0.0f;

                    for (int j0 = 0; j0 < seqK; j0 += Bc) {
                        int bc = std::min(Bc, seqK - j0);

                        // Step 1: compute score tile S_ij[br × bc] = Q_i · K_j^T · scale.
                        for (int i = 0; i < br; ++i) {
                            const float* qi = Qh + (i0 + i) * head_dim;
                            // Absolute query position for causal masking.
                            int qi_pos = (seqK - seqQ) + i0 + i;
                            for (int j = 0; j < bc; ++j) {
                                int kj = j0 + j;
                                if (causal && kj > qi_pos) {
                                    S_ij[static_cast<size_t>(i * bc + j)] = -1e30f;
                                    continue;
                                }
                                const float* kj_ptr = Kh + kj * head_dim;
                                float dot = 0.0f;
                                for (int d = 0; d < head_dim; ++d)
                                    dot += qi[d] * kj_ptr[d];
                                S_ij[static_cast<size_t>(i * bc + j)] = dot * attnScale;
                            }
                        }

                        // Step 2: online softmax update per Q-block row.
                        for (int i = 0; i < br; ++i) {
                            const float* s = S_ij.data() + i * bc;

                            // rowmax of this tile.
                            float m_tilde = -1e30f;
                            for (int j = 0; j < bc; ++j)
                                if (s[j] > m_tilde) m_tilde = s[j];

                            // exp(S - m_tilde) and rowsum.
                            float l_tilde = 0.0f;
                            float* p = P_ij.data() + i * bc;
                            for (int j = 0; j < bc; ++j) {
                                p[j] = (s[j] > -9e29f) ? std::exp(s[j] - m_tilde) : 0.0f;
                                l_tilde += p[j];
                            }

                            // Online update: rescale old O_i and l_i to new max.
                            float m_new   = (m_i[i] > m_tilde) ? m_i[i] : m_tilde;
                            float alpha   = std::exp(m_i[i] - m_new); // correction for old stats
                            float beta    = std::exp(m_tilde - m_new); // correction for new tile

                            float* oi = O_i.data() + i * head_dim;
                            for (int d = 0; d < head_dim; ++d)
                                oi[d] *= alpha;

                            // O_i += beta * P_ij * V_j
                            for (int j = 0; j < bc; ++j) {
                                float pv = beta * p[j];
                                if (pv == 0.0f) continue;
                                const float* vj = Vh + (j0 + j) * head_dim;
                                for (int d = 0; d < head_dim; ++d)
                                    oi[d] += pv * vj[d];
                            }

                            l_i[i] = alpha * l_i[i] + beta * l_tilde;
                            m_i[i] = m_new;
                        }
                    } // end K/V tile loop

                    // Normalise and write output block.
                    for (int i = 0; i < br; ++i) {
                        float inv_l = (l_i[i] > 0.0f) ? 1.0f / l_i[i] : 0.0f;
                        float* oi = O_i.data() + i * head_dim;
                        float* oh = Oh + (i0 + i) * head_dim;
                        for (int d = 0; d < head_dim; ++d)
                            oh[d] = oi[d] * inv_l;
                    }
                } // end Q tile loop
            } // end batch × heads loop
            (void)vDesc;
            break;
        }
        case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: {
            // Element-wise operation between tensors, dispatching on the mode
            // stored in a nested CUDNN_BACKEND_POINTWISE_DESCRIPTOR.
            uintptr_t xId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_XDESC);
            uintptr_t yId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_YDESC);
            uintptr_t bId  = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_BDESC);
            uintptr_t pwId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR);
            float alpha1   = getAttrFloat (opNode, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, 1.0f);
            float alpha2   = getAttrFloat (opNode, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, 1.0f);

            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            void* bPtr = bId ? dataPtrs[bId] : nullptr;
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;

            cudnnPointwiseMode_t mode = CUDNN_POINTWISE_IDENTITY;
            if (pwId) {
                const BackendNode* pw = getNode(pwId);
                if (pw) {
                    auto* mv = getAttrVec(pw, CUDNN_ATTR_POINTWISE_MODE);
                    if (mv && !mv->empty())
                        mode = static_cast<cudnnPointwiseMode_t>((*mv)[0]);
                }
            }

            TensorDesc xDesc = buildTensorDesc(xId);
            size_t nelems = static_cast<size_t>(xDesc.n) * xDesc.c * xDesc.h * xDesc.w;

            const float* X = static_cast<const float*>(xPtr);
            const float* B = bPtr ? static_cast<const float*>(bPtr) : nullptr;
            float*       Y = static_cast<float*>(yPtr);

            for (size_t i = 0; i < nelems; ++i) {
                float x = X[i] * alpha1;
                float b = B ? B[i] * alpha2 : 0.0f;
                float y = x;
                switch (mode) {
                case CUDNN_POINTWISE_ADD:        y = x + b; break;
                case CUDNN_POINTWISE_MUL:        y = x * b; break;
                case CUDNN_POINTWISE_MIN:        y = x < b ? x : b; break;
                case CUDNN_POINTWISE_MAX:        y = x > b ? x : b; break;
                case CUDNN_POINTWISE_DIV:        y = b != 0.f ? x / b : 0.f; break;
                case CUDNN_POINTWISE_SQRT:       y = std::sqrt(x > 0.f ? x : 0.f); break;
                case CUDNN_POINTWISE_EXP:        y = std::exp(x); break;
                case CUDNN_POINTWISE_LOG:        y = x > 0.f ? std::log(x) : -1e30f; break;
                case CUDNN_POINTWISE_NEG:        y = -x; break;
                case CUDNN_POINTWISE_ABS:        y = std::abs(x); break;
                case CUDNN_POINTWISE_CEIL:       y = std::ceil(x); break;
                case CUDNN_POINTWISE_FLOOR:      y = std::floor(x); break;
                case CUDNN_POINTWISE_RECIPROCAL: y = x != 0.f ? 1.f / x : 0.f; break;
                case CUDNN_POINTWISE_RELU_FWD:   y = x > 0.f ? x : 0.f; break;
                case CUDNN_POINTWISE_TANH_FWD:   y = std::tanh(x); break;
                case CUDNN_POINTWISE_SIGMOID_FWD:y = 1.f / (1.f + std::exp(-x)); break;
                case CUDNN_POINTWISE_ELU_FWD:    y = x > 0.f ? x : std::exp(x) - 1.f; break;
                case CUDNN_POINTWISE_SWISH_FWD:  y = x / (1.f + std::exp(-x)); break;
                case CUDNN_POINTWISE_SOFTPLUS_FWD:y = std::log1p(std::exp(x)); break;
                case CUDNN_POINTWISE_GELU_FWD:
                case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD: {
                    // GELU: x * Φ(x) ≈ 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))
                    float t = 0.7978845608f * (x + 0.044715f * x * x * x);
                    y = 0.5f * x * (1.f + std::tanh(t));
                    break;
                }
                case CUDNN_POINTWISE_RELU_BWD:   y = (b > 0.f) ? x : 0.f; break;
                case CUDNN_POINTWISE_TANH_BWD:   y = x * (1.f - b * b); break;
                case CUDNN_POINTWISE_SIGMOID_BWD:y = x * b * (1.f - b); break;
                case CUDNN_POINTWISE_ELU_BWD:    y = x * (b >= 0.f ? 1.f : b + 1.f); break;
                case CUDNN_POINTWISE_GELU_BWD:
                case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD: {
                    float t = 0.7978845608f * (b + 0.044715f * b * b * b);
                    float th = std::tanh(t); float sech2 = 1.f - th * th;
                    float dgelu = 0.5f * (1.f + th) + 0.5f * b * sech2 * 0.7978845608f * (1.f + 3.f * 0.044715f * b * b);
                    y = x * dgelu; break;
                }
                case CUDNN_POINTWISE_SOFTPLUS_BWD:y = x / (1.f + std::exp(-b)); break;
                case CUDNN_POINTWISE_SWISH_BWD: {
                    float sig = 1.f / (1.f + std::exp(-b));
                    y = x * (sig + b * sig * (1.f - sig)); break;
                }
                case CUDNN_POINTWISE_MOD:        y = b != 0.f ? std::fmod(x, b) : 0.f; break;
                case CUDNN_POINTWISE_ADD_SQUARE:  y = x + b * b; break;
                case CUDNN_POINTWISE_POW:        y = std::pow(x, b); break;
                case CUDNN_POINTWISE_CMP_EQ:     y = (x == b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_NEQ:    y = (x != b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_GT:     y = (x >  b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_GE:     y = (x >= b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_LT:     y = (x <  b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_CMP_LE:     y = (x <= b) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_LOGICAL_AND:y = (x != 0.f && b != 0.f) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_LOGICAL_OR: y = (x != 0.f || b != 0.f) ? 1.f : 0.f; break;
                case CUDNN_POINTWISE_LOGICAL_NOT:y = (x == 0.f) ? 1.f : 0.f; break;
                default: y = x; break;  // IDENTITY and unknown
                }
                Y[i] = y;
            }
            break;
        }

        case CUDNN_BACKEND_OPERATION_RESAMPLE_DESCRIPTOR:
            // Deprecated combined resample descriptor (cuDNN ≤ v8.8, no explicit
            // direction attribute). Treat as forward resample.
            [[fallthrough]];
        case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR: {
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_YDESC);
            void* xPtr = dataPtrs[xId];
            void* yPtr = dataPtrs[yId];
            if (!xPtr || !yPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc xD = buildTensorDesc(xId);
            TensorDesc yD = buildTensorDesc(yId);
            float scaleH = (yD.h > 0 && xD.h > 0) ? static_cast<float>(xD.h) / yD.h : 1.f;
            float scaleW = (yD.w > 0 && xD.w > 0) ? static_cast<float>(xD.w) / yD.w : 1.f;

            const float* X = static_cast<const float*>(xPtr);
            float*       Y = static_cast<float*>(yPtr);

            // Bilinear resample (covers both upsample and downsample)
            for (int n = 0; n < yD.n; ++n) {
                for (int c = 0; c < yD.c; ++c) {
                    for (int hy = 0; hy < yD.h; ++hy) {
                        for (int wy = 0; wy < yD.w; ++wy) {
                            float sx = (hy + 0.5f) * scaleH - 0.5f;
                            float sy = (wy + 0.5f) * scaleW - 0.5f;
                            sx = std::max(0.f, std::min(static_cast<float>(xD.h - 1), sx));
                            sy = std::max(0.f, std::min(static_cast<float>(xD.w - 1), sy));
                            int x0 = static_cast<int>(sx);
                            int x1 = std::min(xD.h - 1, x0 + 1);
                            int y0 = static_cast<int>(sy);
                            int y1 = std::min(xD.w - 1, y0 + 1);
                            float wx = sx - x0;
                            float wy_ = sy - y0;
                            size_t base = static_cast<size_t>(n * xD.c + c) * xD.h * xD.w;
                            float v = X[base + x0*xD.w + y0] * (1-wx)*(1-wy_)
                                    + X[base + x0*xD.w + y1] * (1-wx)*wy_
                                    + X[base + x1*xD.w + y0] * wx*(1-wy_)
                                    + X[base + x1*xD.w + y1] * wx*wy_;
                            Y[static_cast<size_t>(n*yD.c + c)*yD.h*yD.w + hy*yD.w + wy] = v;
                        }
                    }
                }
            }
            break;
        }

        case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR: {
            // Backward pass: dX = scatter-add of bilinear-weighted dY gradients.
            // xDesc = dX (gradient w.r.t. input),  yDesc = dY (upstream gradient)
            uintptr_t xId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_XDESC);
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RESAMPLE_YDESC);
            void* dxPtr = dataPtrs[xId];
            void* dyPtr = dataPtrs[yId];
            if (!dxPtr || !dyPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc xD = buildTensorDesc(xId);  // input (dX) dimensions
            TensorDesc yD = buildTensorDesc(yId);   // output (dY) dimensions
            float scaleH = (yD.h > 0 && xD.h > 0) ? static_cast<float>(xD.h) / yD.h : 1.f;
            float scaleW = (yD.w > 0 && xD.w > 0) ? static_cast<float>(xD.w) / yD.w : 1.f;

            float*       dX = static_cast<float*>(dxPtr);
            const float* dY = static_cast<const float*>(dyPtr);

            // Zero-initialize dX
            size_t dxCount = static_cast<size_t>(xD.n) * xD.c * xD.h * xD.w;
            std::memset(dX, 0, dxCount * sizeof(float));

            // Scatter-add: for each output pixel, distribute gradient to 4 input neighbors
            for (int n = 0; n < yD.n; ++n) {
                for (int c = 0; c < yD.c; ++c) {
                    size_t xBase = static_cast<size_t>(n * xD.c + c) * xD.h * xD.w;
                    size_t yBase = static_cast<size_t>(n * yD.c + c) * yD.h * yD.w;
                    for (int hy = 0; hy < yD.h; ++hy) {
                        for (int wy = 0; wy < yD.w; ++wy) {
                            float sx = (hy + 0.5f) * scaleH - 0.5f;
                            float sy = (wy + 0.5f) * scaleW - 0.5f;
                            sx = std::max(0.f, std::min(static_cast<float>(xD.h - 1), sx));
                            sy = std::max(0.f, std::min(static_cast<float>(xD.w - 1), sy));
                            int x0 = static_cast<int>(sx);
                            int x1 = std::min(xD.h - 1, x0 + 1);
                            int y0 = static_cast<int>(sy);
                            int y1 = std::min(xD.w - 1, y0 + 1);
                            float wx = sx - x0;
                            float wy_ = sy - y0;
                            float grad = dY[yBase + hy * yD.w + wy];
                            dX[xBase + x0 * xD.w + y0] += grad * (1 - wx) * (1 - wy_);
                            dX[xBase + x0 * xD.w + y1] += grad * (1 - wx) * wy_;
                            dX[xBase + x1 * xD.w + y0] += grad * wx * (1 - wy_);
                            dX[xBase + x1 * xD.w + y1] += grad * wx * wy_;
                        }
                    }
                }
            }
            break;
        }

        case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR: {
            // Philox 4×32 counter-based PRNG — identical statistical quality to
            // CUDA cuRAND Philox4_32_10 used on real NVIDIA hardware.
            //
            // Philox rounds (10 iterations):
            //   k = (seed_lo, seed_hi)
            //   (L0,R0,L1,R1) = philox_round^10((ctr,0,0,0), k)
            //
            // Distributions:
            //   uniform[0,1): top-23-bit mantissa trick → [1.0, 2.0) → subtract 1
            //   normal(0,1) : Box-Muller: N = sqrt(-2 ln U1) * cos(2π U2)
            //   Bernoulli(p): (uniform < p) ? 1.0f : 0.0f  — dropout mask

            // Try dedicated RNG Y attr first, fall back to activation Y slot
            uintptr_t yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNG_YDESC, 0);
            if (!yId) yId = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_ACTIVATION_YDESC);
            void* yPtr = yId ? dataPtrs[yId] : nullptr;
            if (!yPtr) return CUDNN_STATUS_INVALID_VALUE;

            TensorDesc yDesc  = buildTensorDesc(yId);
            int64_t totalElem = static_cast<int64_t>(yDesc.n) * yDesc.c * yDesc.h * yDesc.w;
            if (totalElem <= 0) return CUDNN_STATUS_INVALID_VALUE;

            uint64_t seed = getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNG_SEED, 0xDEADBEEF12345678ULL);
            int dist      = static_cast<int>(getAttrUint64(opNode, CUDNN_ATTR_OPERATION_RNG_DIST, 0));
            float prob    = getAttrFloat(opNode, CUDNN_ATTR_OPERATION_RNG_BERNOULLI_PROB, 0.5f);

            float* yf = static_cast<float*>(yPtr);

            // ── Philox 4×32 (10 rounds) inline ───────────────────────────────
            // Weyl sequence increments
            constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
            constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;
            // Round multipliers
            constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
            constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;

            uint32_t seedLo = static_cast<uint32_t>(seed);
            uint32_t seedHi = static_cast<uint32_t>(seed >> 32);

            // Process 4 outputs per Philox call
            int64_t blocks = (totalElem + 3) / 4;

            for (int64_t blk = 0; blk < blocks; ++blk) {
                // Counter: (blk_lo, blk_hi, 0, 0)
                uint32_t L0 = static_cast<uint32_t>(blk);
                uint32_t R0 = static_cast<uint32_t>(static_cast<uint64_t>(blk) >> 32);
                uint32_t L1 = 0u, R1 = 0u;
                uint32_t k0 = seedLo, k1 = seedHi;

                for (int r = 0; r < 10; ++r) {
                    uint32_t nL0 = PHILOX_M1 * R1;
                    uint32_t nR0 = static_cast<uint32_t>(
                                    (static_cast<uint64_t>(PHILOX_M1) * R1) >> 32)
                                    ^ k1 ^ L1;
                    uint32_t nL1 = PHILOX_M0 * R0;
                    uint32_t nR1 = static_cast<uint32_t>(
                                    (static_cast<uint64_t>(PHILOX_M0) * R0) >> 32)
                                    ^ k0 ^ L0;
                    L0 = nL0; R0 = nR0; L1 = nL1; R1 = nR1;
                    k0 += PHILOX_W0; k1 += PHILOX_W1;
                }
                uint32_t v[4] = {L0, R0, L1, R1};

                // Convert each of the 4 raw uint32 values to the desired distribution
                for (int lane = 0; lane < 4; ++lane) {
                    int64_t idx = blk * 4 + lane;
                    if (idx >= totalElem) break;

                    // Uniform [0,1) via IEEE 754 mantissa trick
                    uint32_t raw = v[lane];
                    uint32_t bits = 0x3F800000u | (raw >> 9);
                    float u;
                    memcpy(&u, &bits, sizeof(u));
                    u -= 1.0f;   // [0, 1)

                    if (dist == 0) {
                        // Uniform [0, 1)
                        yf[idx] = u;
                    } else if (dist == 1) {
                        // Normal(0,1) via Box-Muller.
                        // Pair consecutive lanes: (lane 0,1) → (n0, n1), (lane 2,3) → (n2, n3)
                        // We accumulate pairs within the block to avoid re-running Philox.
                        // For odd lanes, retrieve the paired even-lane uniform.
                        if (lane % 2 == 0) {
                            // Even lane: compute Box-Muller pair
                            float u2_raw = 0x3F800000u;
                            uint32_t v2_raw = v[lane + 1 < 4 ? lane + 1 : lane];
                            uint32_t bits2 = 0x3F800000u | (v2_raw >> 9);
                            float u2;
                            memcpy(&u2, &bits2, sizeof(u2));
                            u2 -= 1.0f;
                            if (u < 1e-37f) u = 1e-37f;  // avoid log(0)
                            float r_bm   = std::sqrt(-2.0f * std::log(u));
                            float theta  = 6.283185307f * u2;   // 2π·u2
                            yf[idx]      = r_bm * std::cos(theta);
                            // Store second normal for next (odd) lane if in bounds
                            if (lane + 1 < 4 && blk * 4 + lane + 1 < totalElem)
                                yf[blk * 4 + lane + 1] = r_bm * std::sin(theta);
                        }
                        // Odd lane already written by the preceding even lane pass
                    } else {
                        // Bernoulli(prob): keep mask — 1.0 with probability prob
                        yf[idx] = (u < prob) ? 1.0f : 0.0f;
                    }
                }
            }
            break;
        }

        default:
            return CUDNN_STATUS_NOT_SUPPORTED;
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Re-executable plan creation ──────────────────────────────────────────────

cudnnStatus_t cudnnBackendCreateReexecutable(cudnnHandle_t /*handle*/, void* opGraph, void** plan) {
    if (!opGraph || !plan) return CUDNN_STATUS_INVALID_VALUE;
    uintptr_t graphId = reinterpret_cast<uintptr_t>(opGraph);
    const BackendNode *graph = getNode(graphId);
    if (!graph) return CUDNN_STATUS_INVALID_VALUE;

    uintptr_t engineId = g_nextBackendId++;
    BackendNode &engine = g_backendNodes[engineId];
    engine.type = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
    engine.finalized = true;
    engine.attrs[static_cast<int>(CUDNN_ATTR_ENGINE_OPERATION_GRAPH)] = {graphId};

    uintptr_t configId = g_nextBackendId++;
    BackendNode &config = g_backendNodes[configId];
    config.type = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
    config.finalized = true;
    config.attrs[static_cast<int>(CUDNN_ATTR_ENGINECFG_ENGINE)] = {engineId};

    uintptr_t planId = g_nextBackendId++;
    BackendNode &planNode = g_backendNodes[planId];
    planNode.type = CUDNN_BACKEND_HANDLE_DESCRIPTOR;
    planNode.finalized = true;
    planNode.attrs[static_cast<int>(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG)] = {configId};
    planNode.attrs[static_cast<int>(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)] = {0};

    *plan = reinterpret_cast<void*>(planId);
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
