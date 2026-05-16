// CUDA Graph basic shim wrappers: create/destroy/capture/launch/clone/update.
// External-semaphore graph nodes also live here.
// Split from cudart_shim.cpp to keep that file under 800 lines.

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace vgre::api;

using cudaGraph_t     = uint64_t;
using cudaGraphExec_t = uint64_t;
using cudaGraphNode_t = uint64_t;
using cudaGraphExecUpdateResult = vgre::api::CUDAInterceptor::cudaGraphExecUpdateResult;
using cudaMemcpy3DParms = vgre::api::cudaMemcpy3DParms;

using cudaExternalSemaphore_t = uint64_t;
struct cudaExternalSemaphoreSignalParams {
    struct { uint64_t value; } fence;
    unsigned int reserved[16];
    unsigned int flags;
};
struct cudaExternalSemaphoreWaitParams {
    struct { uint64_t value; } fence;
    unsigned int reserved[16];
    unsigned int flags;
};
extern "C" int cudaSignalExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *, const cudaExternalSemaphoreSignalParams *,
    unsigned int, void *);
extern "C" int cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *, const cudaExternalSemaphoreWaitParams *,
    unsigned int, void *);

struct cudaExternalSemaphoreSignalNodeParams {
    const cudaExternalSemaphore_t *extSemArray;
    const cudaExternalSemaphoreSignalParams *paramsArray;
    unsigned int numExtSems;
};
struct cudaExternalSemaphoreWaitNodeParams {
    const cudaExternalSemaphore_t *extSemArray;
    const cudaExternalSemaphoreWaitParams *paramsArray;
    unsigned int numExtSems;
};

namespace {
struct GraphExtSemNodeState {
    bool isWait = false;
    std::vector<cudaExternalSemaphore_t> sems;
    std::vector<cudaExternalSemaphoreSignalParams> signalParams;
    std::vector<cudaExternalSemaphoreWaitParams> waitParams;
};

std::mutex& getGraphExtSemMutex() {
    static std::mutex mu; return mu;
}
std::unordered_map<cudaGraphNode_t, std::shared_ptr<GraphExtSemNodeState>>&
getGraphExtSemNodes() {
    static std::unordered_map<cudaGraphNode_t, std::shared_ptr<GraphExtSemNodeState>> nodes;
    return nodes;
}
static int vgre_ext_sem_graph_callback(void *ctx) {
    auto *state = static_cast<GraphExtSemNodeState *>(ctx);
    if (!state || state->sems.empty()) return 0;
    if (state->isWait)
        (void)cudaWaitExternalSemaphoresAsync(state->sems.data(), state->waitParams.data(),
                                              (unsigned)state->sems.size(), nullptr);
    else
        (void)cudaSignalExternalSemaphoresAsync(state->sems.data(), state->signalParams.data(),
                                                (unsigned)state->sems.size(), nullptr);
    return 0;
}
} // namespace

extern "C" cudaError_t vgreGraphExtSemGetSignalNodeParams(
    cudaGraphNode_t node, cudaExternalSemaphoreSignalNodeParams *params_out) {
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    auto it = getGraphExtSemNodes().find(node);
    if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
    auto *s = it->second.get();
    if (s->isWait || !params_out) return cudaErrorInvalidValue;
    params_out->extSemArray = s->sems.data();
    params_out->paramsArray = s->signalParams.data();
    params_out->numExtSems  = (unsigned)s->sems.size();
    return cudaSuccess;
}
extern "C" cudaError_t vgreGraphExtSemGetWaitNodeParams(
    cudaGraphNode_t node, cudaExternalSemaphoreWaitNodeParams *params_out) {
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    auto it = getGraphExtSemNodes().find(node);
    if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
    auto *s = it->second.get();
    if (!s->isWait || !params_out) return cudaErrorInvalidValue;
    params_out->extSemArray = s->sems.data();
    params_out->paramsArray = s->waitParams.data();
    params_out->numExtSems  = (unsigned)s->sems.size();
    return cudaSuccess;
}

extern "C" {

cudaError_t cudaGraphCreate(cudaGraph_t *g, unsigned int f) {
    return CUDAInterceptor::instance().graphCreate(g, f);
}
cudaError_t cudaStreamBeginCapture(cudaStream_t s, unsigned int) {
    return CUDAInterceptor::instance().streamBeginCapture(s);
}
cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t *g) {
    return CUDAInterceptor::instance().streamEndCapture(s, g);
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pExec, cudaGraph_t graph,
                                 cudaGraphNode_t *pErrorNode, char *pLog, size_t logSz) {
    cudaError_t r = CUDAInterceptor::instance().graphInstantiate(pExec, graph);
    if (r != cudaSuccess) {
        if (pErrorNode) *pErrorNode = 0;
        if (pLog && logSz > 0) {
            const char* msg = (r == cudaErrorInvalidValue)
                ? "Graph instantiation failed: invalid node or dependency cycle"
                : "Graph instantiation failed: internal error";
            std::strncpy(pLog, msg, logSz - 1); pLog[logSz - 1] = '\0';
        }
    }
    return r;
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t s) {
    return CUDAInterceptor::instance().graphLaunch(e, s);
}
cudaError_t cudaGraphClone(cudaGraph_t *c, cudaGraph_t o) {
    return CUDAInterceptor::instance().graphClone(c, o);
}
cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    return CUDAInterceptor::instance().graphDestroy(g);
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) {
    return CUDAInterceptor::instance().graphExecDestroy(e);
}
cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pNode, cudaGraph_t graph,
    const cudaGraphNode_t *deps, size_t nDeps, const cudaMemcpy3DParms *p) {
    return CUDAInterceptor::instance().graphAddMemcpyNode(pNode, graph, deps, nDeps, p);
}
cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pNode, cudaGraph_t graph,
    const cudaGraphNode_t *deps, size_t nDeps,
    void *dst, const void *src, size_t count, cudaMemcpyKind_t kind) {
    return CUDAInterceptor::instance().graphAddMemcpyNode1D(pNode, graph, deps, nDeps,
                                                            dst, src, count, kind);
}
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hExec, cudaGraph_t hGraph,
                                cudaGraphNode_t *hErr, cudaGraphExecUpdateResult *res) {
    return CUDAInterceptor::instance().graphExecUpdate(hExec, hGraph, hErr,
        reinterpret_cast<CUDAInterceptor::cudaGraphExecUpdateResult*>(res));
}
cudaError_t cudaGraphExecUpdate_v2(cudaGraphExec_t hExec, cudaGraph_t hGraph,
    const cudaGraphNode_t *nodeList, size_t listSize,
    cudaGraphNode_t *hErr, cudaGraphExecUpdateResult *res) {
    return CUDAInterceptor::instance().graphExecUpdateV2(hExec, hGraph, nodeList, listSize, hErr,
        reinterpret_cast<CUDAInterceptor::cudaGraphExecUpdateResult*>(res));
}
cudaError_t cudaGraphAddConditionalNode(cudaGraphNode_t *pNode, cudaGraph_t graph,
    const cudaGraphNode_t *deps, size_t nDeps,
    int (*condFn)(void *), void *condCtx, cudaGraph_t bodyGraph,
    unsigned int flags, unsigned int maxIter) {
    if (!pNode) return cudaError_t(1);
    auto &engine = vgre::core::RuntimeEngine::instance();
    if (!engine.isInitialized()) return cudaError_t(3);
    std::vector<uint64_t> d(deps, deps + nDeps);
    vgre::core::GraphCondType ct = vgre::core::GraphCondType::IF;
    if (flags == 1) ct = vgre::core::GraphCondType::WHILE;
    if (flags == 2) ct = vgre::core::GraphCondType::SWITCH;
    unsigned int iters = maxIter ? maxIter : 65536u;
    uint64_t outId = 0;
    auto r = engine.graphAddConditionalNode(
        (vgre::GraphId)graph, condFn, condCtx, (vgre::GraphId)bodyGraph, ct, iters, d, outId);
    if (r != vgre::VGREResult::SUCCESS) return cudaError_t(1);
    *pNode = outId;
    return cudaError_t(0);
}
cudaError_t cudaGraphAddExternalSemaphoreSignalNode(
    cudaGraphNode_t *pNode, cudaGraph_t graph,
    const cudaGraphNode_t *deps, size_t nDeps,
    const cudaExternalSemaphoreSignalNodeParams *p) {
    if (!pNode || !p || !p->numExtSems || !p->extSemArray || !p->paramsArray)
        return cudaErrorInvalidValue;
    auto state = std::make_shared<GraphExtSemNodeState>();
    state->isWait = false;
    state->sems.assign(p->extSemArray, p->extSemArray + p->numExtSems);
    state->signalParams.assign(p->paramsArray, p->paramsArray + p->numExtSems);
    std::vector<uint64_t> d(deps && nDeps ? deps : nullptr,
                            deps && nDeps ? deps + nDeps : nullptr);
    uint64_t nodeId = 0;
    auto r = vgre::core::RuntimeEngine::instance().graphAddConditionalNode(
        (vgre::GraphId)graph, vgre_ext_sem_graph_callback, state.get(),
        0, vgre::core::GraphCondType::IF, 1, d, nodeId);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
    { std::lock_guard<std::mutex> lk(getGraphExtSemMutex()); getGraphExtSemNodes()[nodeId] = std::move(state); }
    *pNode = nodeId;
    return cudaSuccess;
}
cudaError_t cudaGraphAddExternalSemaphoreWaitNode(
    cudaGraphNode_t *pNode, cudaGraph_t graph,
    const cudaGraphNode_t *deps, size_t nDeps,
    const cudaExternalSemaphoreWaitNodeParams *p) {
    if (!pNode || !p || !p->numExtSems || !p->extSemArray || !p->paramsArray)
        return cudaErrorInvalidValue;
    auto state = std::make_shared<GraphExtSemNodeState>();
    state->isWait = true;
    state->sems.assign(p->extSemArray, p->extSemArray + p->numExtSems);
    state->waitParams.assign(p->paramsArray, p->paramsArray + p->numExtSems);
    std::vector<uint64_t> d(deps && nDeps ? deps : nullptr,
                            deps && nDeps ? deps + nDeps : nullptr);
    uint64_t nodeId = 0;
    auto r = vgre::core::RuntimeEngine::instance().graphAddConditionalNode(
        (vgre::GraphId)graph, vgre_ext_sem_graph_callback, state.get(),
        0, vgre::core::GraphCondType::IF, 1, d, nodeId);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
    { std::lock_guard<std::mutex> lk(getGraphExtSemMutex()); getGraphExtSemNodes()[nodeId] = std::move(state); }
    *pNode = nodeId;
    return cudaSuccess;
}
cudaError_t cudaGraphExecExternalSemaphoreSignalNodeSetParams(
    cudaGraphExec_t, cudaGraphNode_t node,
    const cudaExternalSemaphoreSignalNodeParams *p) {
    if (!p || !p->numExtSems || !p->extSemArray || !p->paramsArray) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    auto it = getGraphExtSemNodes().find(node);
    if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
    auto &s = *it->second;
    s.isWait = false;
    s.sems.assign(p->extSemArray, p->extSemArray + p->numExtSems);
    s.signalParams.assign(p->paramsArray, p->paramsArray + p->numExtSems);
    s.waitParams.clear();
    return cudaSuccess;
}
cudaError_t cudaGraphExecExternalSemaphoreWaitNodeSetParams(
    cudaGraphExec_t, cudaGraphNode_t node,
    const cudaExternalSemaphoreWaitNodeParams *p) {
    if (!p || !p->numExtSems || !p->extSemArray || !p->paramsArray) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lk(getGraphExtSemMutex());
    auto it = getGraphExtSemNodes().find(node);
    if (it == getGraphExtSemNodes().end()) return cudaErrorInvalidValue;
    auto &s = *it->second;
    s.isWait = true;
    s.sems.assign(p->extSemArray, p->extSemArray + p->numExtSems);
    s.waitParams.assign(p->paramsArray, p->paramsArray + p->numExtSems);
    s.signalParams.clear();
    return cudaSuccess;
}

} // extern "C"
