#include "interpreter_backend.h"

#include "vgre/common/logger.h"
#include "vgre/debug/ptx_interpreter.h"
#include "vgre/xla/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <string>
#include <utility>

namespace vgre {
namespace compiler {
namespace backend {

namespace {

// Holds the PTX text + entry name. The PtxInterpreter carries per-launch
// execution state, so a fresh instance is constructed for each launch to keep
// launches independent; the parsed text is cheap to re-parse and this keeps the
// handle trivially reusable across concurrent launches.
class InterpreterKernel : public PreparedKernel {
public:
    InterpreterKernel(std::string ptx, std::string entry)
        : ptx_(std::move(ptx)), entry_(std::move(entry)) {}

    const std::string& entry() const override { return entry_; }
    const std::string& ptx() const { return ptx_; }

private:
    std::string ptx_;
    std::string entry_;
};

}  // namespace

std::unique_ptr<PreparedKernel> InterpreterBackend::preparePtx(
    const std::string& ptx, const std::string& entry) {
    try {
        // Constructing the interpreter parses the PTX and resolves the entry, so
        // a malformed kernel fails here (at prepare) rather than at launch.
        vgre::debug::PtxInterpreter probe(ptx, entry);
        (void)probe;
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("InterpreterBackend",
                       std::string("preparePtx failed: ") + e.what());
        return nullptr;
    } catch (...) {
        // PtxInterpreter lives in a separate component; on macOS a std:: exception
        // thrown across that boundary can miss the typed catch above (RTTI is not
        // unified across libraries). A malformed kernel must still fail cleanly,
        // not std::terminate — so catch anything and report it as unpreparable.
        VGRE_LOG_ERROR("InterpreterBackend", "preparePtx failed (malformed PTX)");
        return nullptr;
    }
    return std::unique_ptr<PreparedKernel>(new InterpreterKernel(ptx, entry));
}

bool InterpreterBackend::launch(PreparedKernel& kernel, const LaunchConfig& cfg,
                                void* const* args, int numArgs) {
    auto* k = dynamic_cast<InterpreterKernel*>(&kernel);
    if (!k) {
        VGRE_LOG_ERROR("InterpreterBackend", "launch: foreign kernel handle");
        return false;
    }

    const vgre::debug::Dim3 grid{static_cast<int>(cfg.gridDim[0]),
                                 static_cast<int>(cfg.gridDim[1]),
                                 static_cast<int>(cfg.gridDim[2])};
    const vgre::debug::Dim3 block{static_cast<int>(cfg.blockDim[0]),
                                  static_cast<int>(cfg.blockDim[1]),
                                  static_cast<int>(cfg.blockDim[2])};
    const int gridTotal = grid.total();
    auto& pool = vgre::xla::ThreadPool::global();
    const int workers = static_cast<int>(pool.concurrency());

    // Small grids / single-core: keep the plain single-instance path (also the
    // one the debugger uses), avoiding any pool overhead.
    if (gridTotal <= 1 || workers <= 1) {
        try {
            vgre::debug::PtxInterpreter in(k->ptx(), k->entry());
            in.launch(grid, block, args, numArgs);
            return in.resume() == vgre::debug::StopReason::Exited;
        } catch (const std::exception& e) {
            VGRE_LOG_ERROR("InterpreterBackend",
                           std::string("launch failed: ") + e.what());
            return false;
        } catch (...) {                                  // cross-library RTTI safety (see preparePtx)
            VGRE_LOG_ERROR("InterpreterBackend", "launch failed (interpreter error)");
            return false;
        }
    }

    // Parallel: partition the grid into `chunks` contiguous CTA ranges, each run
    // by its own interpreter instance (independent per-CTA shared memory + thread
    // state) on a pool worker. Tier-0 has no cross-CTA atomics, and barrier
    // kernels write disjoint output tiles per block, so this is race-free.
    const int chunks = std::min(workers, gridTotal);
    std::atomic<bool> ok{true};
    pool.parallelFor(chunks, 1, [&](int64_t c) {
        const int begin = static_cast<int>(c * gridTotal / chunks);
        const int end   = static_cast<int>((c + 1) * gridTotal / chunks);
        if (begin >= end) return;
        try {
            vgre::debug::PtxInterpreter in(k->ptx(), k->entry());
            if (!in.runCtaRange(grid, block, args, numArgs, begin, end))
                ok.store(false, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            ok.store(false, std::memory_order_relaxed);
            VGRE_LOG_ERROR("InterpreterBackend",
                           std::string("launch failed: ") + e.what());
        } catch (...) {                                  // cross-library RTTI safety
            ok.store(false, std::memory_order_relaxed);
            VGRE_LOG_ERROR("InterpreterBackend", "launch failed (interpreter error)");
        }
    });
    return ok.load(std::memory_order_relaxed);
}

}  // namespace backend
}  // namespace compiler
}  // namespace vgre
