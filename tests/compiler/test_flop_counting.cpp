#include <iostream>
#include <cassert>
#include <memory>
#include "vgre/compiler/llvm_translation_engine.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/DerivedTypes.h>

using namespace vgre::compiler;

void test_basic_arithmetic() {
    LLVMTranslationEngine engine;
    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("test_module", context);
    llvm::IRBuilder<> builder(context);

    auto* funcType = llvm::FunctionType::get(builder.getVoidTy(), {builder.getFloatTy(), builder.getFloatTy()}, false);
    auto* func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "test_flops", module.get());
    auto* entry = llvm::BasicBlock::Create(context, "entry", func);
    builder.SetInsertPoint(entry);

    auto* a = func->getArg(0);
    auto* b = func->getArg(1);

    auto* sum = builder.CreateFAdd(a, b);    // 1 FLOP
    auto* diff = builder.CreateFSub(sum, b); // 1 FLOP
    auto* prod = builder.CreateFMul(diff, a); // 1 FLOP
    auto* quot = builder.CreateFDiv(prod, b); // 1 FLOP
    builder.CreateRetVoid();

    uint64_t flops = engine.analyzeStaticFlops(*module);
    std::cout << "[INFO] Basic Arithmetic FLOPs: " << flops << std::endl;
    assert(flops == 4);
    std::cout << "[PASS] Basic Arithmetic FLOP counting" << std::endl;
}

void test_fma_and_vector() {
    LLVMTranslationEngine engine;
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

    // Create FMA intrinsic call
    llvm::Function* fmaFunc = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::fma, {vecType});
    builder.CreateCall(fmaFunc, {a, b, c}); // 2 FLOPs * 4 lanes = 8 FLOPs

    uint64_t flops = engine.analyzeStaticFlops(*module);
    std::cout << "[INFO] FMA/Vector FLOPs: " << flops << std::endl;
    assert(flops == 8);
    std::cout << "[PASS] FMA/Vector FLOP counting" << std::endl;
}

int main() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    std::cout << "=== VGRE FLOP Counting Unit Tests ===" << std::endl;
    try {
        test_basic_arithmetic();
        test_fma_and_vector();
        std::cout << "\nAll FLOP counting tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
