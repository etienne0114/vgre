#include "interpreter_backend.h"

#include "vgre/common/logger.h"
#include "vgre/debug/ptx_interpreter.h"

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

    // The PtxInterpreter is 1D today; Track Z will extend it to full 3D. Warn if
    // the launch actually used the y/z dimensions (which the kernel cannot yet
    // observe), then run the x dimension.
    if (cfg.gridDim[1] != 1 || cfg.gridDim[2] != 1 ||
        cfg.blockDim[1] != 1 || cfg.blockDim[2] != 1) {
        VGRE_LOG_WARN("InterpreterBackend",
            "3D launch geometry not yet supported by the interpreter tier — "
            "running the x dimension only (Track Z: 3D interpreter pending).");
    }

    try {
        vgre::debug::PtxInterpreter in(k->ptx(), k->entry());
        in.launch(static_cast<int>(cfg.gridDim[0]),
                  static_cast<int>(cfg.blockDim[0]), args, numArgs);
        return in.resume() == vgre::debug::StopReason::Exited;
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("InterpreterBackend",
                       std::string("launch failed: ") + e.what());
        return false;
    }
}

}  // namespace backend
}  // namespace compiler
}  // namespace vgre
