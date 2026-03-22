#include <iostream>
#include <cassert>
#include <memory>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/DerivedTypes.h>

// Standalone implementation of the analysis logic for verification
uint64_t analyzeStaticFlops(const llvm::Module &module) {
    uint64_t flops = 0;
    for (const auto &F : module) {
        if (F.isDeclaration()) continue;
        for (const auto &BB : F) {
            for (const auto &I : BB) {
                unsigned opcode = I.getOpcode();
                uint64_t opFlops = 0;
                switch (opcode) {
                    case llvm::Instruction::FAdd:
                    case llvm::Instruction::FSub:
                    case llvm::Instruction::FMul:
                    case llvm::Instruction::FDiv:
                    case llvm::Instruction::FRem:
                        opFlops = 1;
                        break;
                    case llvm::Instruction::Call: {
                        const auto *call = llvm::cast<llvm::CallInst>(&I);
                        const auto *callee = call->getCalledFunction();
                        if (callee && callee->isIntrinsic()) {
                            auto id = callee->getIntrinsicID();
                            if (id == llvm::Intrinsic::fma) {
                                opFlops = 2;
                            } else if (id == llvm::Intrinsic::sqrt || id == llvm::Intrinsic::sin || 
                                       id == llvm::Intrinsic::cos || id == llvm::Intrinsic::exp || 
                                       id == llvm::Intrinsic::log || id == llvm::Intrinsic::pow) {
                                opFlops = 8;
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
                
                if (opFlops > 0) {
                    // If it's a vector instruction, multiply by vector width
                    if (auto *vTy = llvm::dyn_cast<llvm::FixedVectorType>(I.getType())) {
                        opFlops *= vTy->getNumElements();
                    } else if (I.getNumOperands() > 0) {
                        for (unsigned i = 0; i < I.getNumOperands(); ++i) {
                             if (auto *ovTy = llvm::dyn_cast<llvm::FixedVectorType>(I.getOperand(i)->getType())) {
                                 opFlops *= ovTy->getNumElements();
                                 break;
                             }
                        }
                    }
                    flops += opFlops;
                }
            }
        }
    }
    return flops;
}

void test_basic_arithmetic() {
    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("test_module", context);
    llvm::IRBuilder<> builder(context);

    auto* funcType = llvm::FunctionType::get(builder.getVoidTy(), {builder.getFloatTy(), builder.getFloatTy()}, false);
    auto* func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "test_flops", module.get());
    auto* entry = llvm::BasicBlock::Create(context, "entry", func);
    builder.SetInsertPoint(entry);

    auto* a = func->getArg(0);
    auto* b = func->getArg(1);

    builder.CreateFAdd(a, b);    // 1 FLOP
    builder.CreateFSub(a, b);    // 1 FLOP
    builder.CreateFMul(a, b);    // 1 FLOP
    builder.CreateFDiv(a, b);    // 1 FLOP
    builder.CreateRetVoid();

    uint64_t flops = analyzeStaticFlops(*module);
    std::cout << "[INFO] Basic Arithmetic FLOPs detected: " << flops << std::endl;
    assert(flops == 4);
}

void test_fma_and_vector() {
    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("test_module", context);
    llvm::IRBuilder<> builder(context);

    // Vector FMA: float4
    auto* vecType = llvm::FixedVectorType::get(builder.getFloatTy(), 4);
    auto* funcType = llvm::FunctionType::get(builder.getVoidTy(), {vecType, vecType, vecType}, false);
    auto* func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "test_fma", module.get());
    auto* entry = llvm::BasicBlock::Create(context, "entry", func);
    builder.SetInsertPoint(entry);

    auto* a = func->getArg(0);
    auto* b = func->getArg(1);
    auto* c = func->getArg(2);

    llvm::Function* fmaFunc = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::fma, {vecType});
    builder.CreateCall(fmaFunc, {a, b, c}); // 2 FLOPs * 4 lanes = 8 FLOPs

    uint64_t flops = analyzeStaticFlops(*module);
    std::cout << "[INFO] FMA/Vector FLOPs detected: " << flops << std::endl;
    assert(flops == 8);
}

int main() {
    std::cout << "=== VGRE Standalone FLOP Counting Verification ===" << std::endl;
    test_basic_arithmetic();
    test_fma_and_vector();
    std::cout << "\nAll FLOP counting verification cases passed!" << std::endl;
    return 0;
}
