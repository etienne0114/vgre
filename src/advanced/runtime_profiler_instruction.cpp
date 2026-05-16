#include "vgre/advanced/runtime_profiler.h"
#include "vgre/core/runtime_engine.h"

#include <algorithm>
#include <sstream>

namespace vgre {
namespace advanced {
namespace {

static std::string trimIRLine(std::string line) {
  auto semicolon = line.find(';');
  if (semicolon != std::string::npos) line.erase(semicolon);
  auto first = line.find_first_not_of(" \t");
  if (first == std::string::npos) return {};
  auto last = line.find_last_not_of(" \t\r\n");
  return line.substr(first, last - first + 1);
}

static bool containsToken(const std::string &line, const char *token) {
  return line.find(token) != std::string::npos;
}

static InstructionSample classifyLLVMInstructionMix(const std::string &irCode) {
  InstructionSample sample;
  std::istringstream in(irCode);
  std::string line;
  while (std::getline(in, line)) {
    line = trimIRLine(line);
    if (line.empty() || line.back() == ':' || line.front() == '}' ||
        line.rfind("define ", 0) == 0 || line.rfind("declare ", 0) == 0)
      continue;

    std::string op = line;
    auto eq = op.find(" = ");
    if (eq != std::string::npos) op = op.substr(eq + 3);

    if (op.rfind("load ", 0) == 0 || containsToken(op, " load ")) {
      ++sample.loadCount;
    } else if (op.rfind("store ", 0) == 0 || containsToken(op, " store ")) {
      ++sample.storeCount;
    } else if (op.rfind("br ", 0) == 0 || op.rfind("switch ", 0) == 0 ||
               op.rfind("indirectbr ", 0) == 0) {
      ++sample.branchCount;
    } else if (containsToken(op, "syncthreads") || containsToken(op, "barrier") ||
               containsToken(op, "llvm.nvvm.barrier")) {
      ++sample.barrierCount;
    } else if (op.rfind("fadd ", 0) == 0 || op.rfind("fsub ", 0) == 0 ||
               op.rfind("fmul ", 0) == 0 || op.rfind("fdiv ", 0) == 0 ||
               op.rfind("frem ", 0) == 0 || op.rfind("add ", 0) == 0 ||
               op.rfind("sub ", 0) == 0 || op.rfind("mul ", 0) == 0 ||
               op.rfind("sdiv ", 0) == 0 || op.rfind("udiv ", 0) == 0 ||
               op.rfind("srem ", 0) == 0 || op.rfind("urem ", 0) == 0 ||
               op.rfind("shl ", 0) == 0 || op.rfind("lshr ", 0) == 0 ||
               op.rfind("ashr ", 0) == 0 || op.rfind("and ", 0) == 0 ||
               op.rfind("or ", 0) == 0 || op.rfind("xor ", 0) == 0 ||
               containsToken(op, "llvm.fma") || containsToken(op, "llvm.sqrt") ||
               containsToken(op, "llvm.sin") || containsToken(op, "llvm.cos") ||
               containsToken(op, "llvm.exp") || containsToken(op, "llvm.log")) {
      ++sample.aluCount;
    } else if (op.rfind("ret ", 0) == 0) {
      ++sample.branchCount;
    } else {
      ++sample.otherCount;
    }
  }
  return sample;
}

static void scaleInstructionSample(InstructionSample &sample, uint64_t multiplier) {
  sample.loadCount *= multiplier;
  sample.storeCount *= multiplier;
  sample.aluCount *= multiplier;
  sample.barrierCount *= multiplier;
  sample.branchCount *= multiplier;
  sample.otherCount *= multiplier;
}

} // namespace

void RuntimeProfiler::recordInstructionSample(const std::string& kernelName,
                                               const InstructionSample& sample) {
    if (!enabled_.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto& mix = instructionMixes_[kernelName];
    mix.loadCount    += sample.loadCount;
    mix.storeCount   += sample.storeCount;
    mix.aluCount     += sample.aluCount;
    mix.barrierCount += sample.barrierCount;
    mix.branchCount  += sample.branchCount;
    mix.otherCount   += sample.otherCount;
}

void RuntimeProfiler::estimateInstructions(const std::string& kernelName,
                                            const dim3& gridDim,
                                            const dim3& blockDim,
                                            uint64_t instructionsPerThread) {
    if (!enabled_.load(std::memory_order_relaxed)) return;

    uint64_t totalThreads = static_cast<uint64_t>(gridDim.x) * gridDim.y * gridDim.z *
                            static_cast<uint64_t>(blockDim.x) * blockDim.y * blockDim.z;
    uint64_t totalInsns = instructionsPerThread * totalThreads;

    auto& re = vgre::core::RuntimeEngine::instance();
    KernelId kid = re.lookupKernelIdByName(kernelName.c_str());
    InstructionSample sample{};
    if (kid != 0) {
        const KernelIR* ir = re.getKernelIR(kid);
        if (ir && !ir->irCode.empty()) {
            sample = classifyLLVMInstructionMix(ir->irCode);
            scaleInstructionSample(sample, totalThreads);
            recordInstructionSample(kernelName, sample);
            return;
        }
    }

    sample.otherCount = totalInsns;
    recordInstructionSample(kernelName, sample);
}

InstructionSample RuntimeProfiler::getInstructionMix(const std::string& kernelName) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = instructionMixes_.find(kernelName);
    if (it != instructionMixes_.end()) return it->second;
    return InstructionSample{};
}

} // namespace advanced
} // namespace vgre
