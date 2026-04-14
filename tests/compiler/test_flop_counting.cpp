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

// ── helpers ──────────────────────────────────────────────────────────────────

// Make a simple scalar void(float, float) function and return its entry block
// builder + the two arguments so each test can emit instructions freely.
struct ScalarFn {
    llvm::LLVMContext ctx;
    std::unique_ptr<llvm::Module> mod;
    llvm::IRBuilder<> builder;
    llvm::Value *a = nullptr, *b = nullptr;
    ScalarFn(const char *name)
        : mod(std::make_unique<llvm::Module>(name, ctx)), builder(ctx) {
        auto *ft = llvm::FunctionType::get(
            builder.getVoidTy(),
            {builder.getFloatTy(), builder.getFloatTy()}, false);
        auto *f = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage, name, mod.get());
        auto *bb = llvm::BasicBlock::Create(ctx, "entry", f);
        builder.SetInsertPoint(bb);
        a = f->getArg(0);
        b = f->getArg(1);
    }
};

// ── test cases ────────────────────────────────────────────────────────────────

// Existing: fadd + fsub + fmul + fdiv = 4 FLOPs
void test_basic_arithmetic() {
    ScalarFn fn("test_basic");
    auto &B = fn.builder;
    auto *s  = B.CreateFAdd(fn.a, fn.b);  // 1
    auto *d  = B.CreateFSub(s,    fn.b);  // 1
    auto *p  = B.CreateFMul(d,    fn.a);  // 1
    (void)     B.CreateFDiv(p,    fn.b);  // 1
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] Basic Arithmetic FLOPs: " << flops << std::endl;
    assert(flops == 4 && "fadd+fsub+fmul+fdiv must yield 4 FLOPs");
    std::cout << "[PASS] Basic Arithmetic FLOP counting" << std::endl;
}

// Existing: <4 x float> FMA = 2 * 4 = 8 FLOPs
void test_fma_vector() {
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("test_fma", ctx);
    llvm::IRBuilder<> B(ctx);
    auto *vecTy = llvm::FixedVectorType::get(B.getFloatTy(), 4);
    auto *ft = llvm::FunctionType::get(B.getVoidTy(), {vecTy, vecTy, vecTy}, false);
    auto *f  = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fma4", mod.get());
    auto *bb = llvm::BasicBlock::Create(ctx, "entry", f);
    B.SetInsertPoint(bb);
    auto *fmaF = llvm::Intrinsic::getDeclaration(mod.get(), llvm::Intrinsic::fma, {vecTy});
    B.CreateCall(fmaF, {f->getArg(0), f->getArg(1), f->getArg(2)}); // 2*4=8
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*mod);
    std::cout << "[INFO] FMA/Vector FLOPs: " << flops << std::endl;
    assert(flops == 8 && "<4 x float> fma must yield 8 FLOPs");
    std::cout << "[PASS] FMA/Vector FLOP counting" << std::endl;
}

// New: frem + fneg (LLVM 10+) = 1 + 1 = 2 FLOPs
void test_frem_and_fneg() {
    ScalarFn fn("test_frem_fneg");
    auto &B = fn.builder;
    auto *r = B.CreateFRem(fn.a, fn.b);  // 1 FLOP
    (void)    B.CreateFNeg(r);            // 1 FLOP
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] FRem + FNeg FLOPs: " << flops << std::endl;
    assert(flops == 2 && "frem + fneg must yield 2 FLOPs");
    std::cout << "[PASS] FRem + FNeg FLOP counting" << std::endl;
}

// New: fcmp (FP comparison) = 1 FLOP
void test_fcmp() {
    ScalarFn fn("test_fcmp");
    auto &B = fn.builder;
    // fcmp returns i1 (not float); result type is scalar — vector check must
    // fall through to operand inspection.
    (void) B.CreateFCmpOLT(fn.a, fn.b);  // 1 FLOP
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] FCmp FLOPs: " << flops << std::endl;
    assert(flops == 1 && "scalar fcmp must yield 1 FLOP");
    std::cout << "[PASS] FCmp FLOP counting" << std::endl;
}

// New: vector fcmp — 4 lanes → 4 FLOPs
void test_fcmp_vector() {
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("test_fcmp_vec", ctx);
    llvm::IRBuilder<> B(ctx);
    auto *vecTy = llvm::FixedVectorType::get(B.getFloatTy(), 4);
    auto *ft = llvm::FunctionType::get(B.getVoidTy(), {vecTy, vecTy}, false);
    auto *f  = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fcmp4", mod.get());
    auto *bb = llvm::BasicBlock::Create(ctx, "entry", f);
    B.SetInsertPoint(bb);
    (void) B.CreateFCmpOLT(f->getArg(0), f->getArg(1)); // 4 FLOPs
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*mod);
    std::cout << "[INFO] Vector FCmp FLOPs: " << flops << std::endl;
    assert(flops == 4 && "<4 x float> fcmp must yield 4 FLOPs");
    std::cout << "[PASS] Vector FCmp FLOP counting" << std::endl;
}

// New: sqrt = 4 FLOPs (hardware-accelerated, not lumped with sin/cos)
void test_sqrt() {
    ScalarFn fn("test_sqrt");
    auto &B = fn.builder;
    auto *sqrtF = llvm::Intrinsic::getDeclaration(
        fn.mod.get(), llvm::Intrinsic::sqrt, {B.getFloatTy()});
    B.CreateCall(sqrtF, {fn.a});  // 4 FLOPs
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] sqrt FLOPs: " << flops << std::endl;
    assert(flops == 4 && "sqrt must yield 4 FLOPs");
    std::cout << "[PASS] sqrt FLOP counting" << std::endl;
}

// New: fabs + floor + ceil + round + minnum = 5 × 1 = 5 FLOPs
void test_cheap_fp_intrinsics() {
    ScalarFn fn("test_cheap_fp");
    auto *fty = fn.builder.getFloatTy();
    auto &B   = fn.builder;
    auto *mod = fn.mod.get();

    auto call1 = [&](llvm::Intrinsic::ID id) {
        auto *f = llvm::Intrinsic::getDeclaration(mod, id, {fty});
        B.CreateCall(f, {fn.a});
    };
    auto call2 = [&](llvm::Intrinsic::ID id) {
        auto *f = llvm::Intrinsic::getDeclaration(mod, id, {fty});
        B.CreateCall(f, {fn.a, fn.b});
    };

    call1(llvm::Intrinsic::fabs);      // 1 FLOP
    call1(llvm::Intrinsic::floor);     // 1 FLOP
    call1(llvm::Intrinsic::ceil);      // 1 FLOP
    call1(llvm::Intrinsic::round);     // 1 FLOP
    call2(llvm::Intrinsic::minnum);    // 1 FLOP
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] Cheap FP intrinsics FLOPs: " << flops << std::endl;
    assert(flops == 5 && "fabs+floor+ceil+round+minnum must yield 5 FLOPs");
    std::cout << "[PASS] Cheap FP intrinsics FLOP counting" << std::endl;
}

// New: sin + cos = 8 + 8 = 16 FLOPs
void test_sin_cos() {
    ScalarFn fn("test_sincos");
    auto *fty = fn.builder.getFloatTy();
    auto &B   = fn.builder;
    auto *mod = fn.mod.get();
    auto *sinF = llvm::Intrinsic::getDeclaration(mod, llvm::Intrinsic::sin, {fty});
    auto *cosF = llvm::Intrinsic::getDeclaration(mod, llvm::Intrinsic::cos, {fty});
    B.CreateCall(sinF, {fn.a});  // 8 FLOPs
    B.CreateCall(cosF, {fn.b});  // 8 FLOPs
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] sin + cos FLOPs: " << flops << std::endl;
    assert(flops == 16 && "sin+cos must yield 16 FLOPs");
    std::cout << "[PASS] sin/cos FLOP counting" << std::endl;
}

// New: exp + log + exp2 + log2 + log10 = 5 × 8 = 40 FLOPs
void test_exp_log_family() {
    ScalarFn fn("test_explogs");
    auto *fty = fn.builder.getFloatTy();
    auto &B   = fn.builder;
    auto *mod = fn.mod.get();
    auto call1 = [&](llvm::Intrinsic::ID id) {
        auto *f = llvm::Intrinsic::getDeclaration(mod, id, {fty});
        B.CreateCall(f, {fn.a});
    };
    call1(llvm::Intrinsic::exp);    // 8
    call1(llvm::Intrinsic::log);    // 8
    call1(llvm::Intrinsic::exp2);   // 8
    call1(llvm::Intrinsic::log2);   // 8
    call1(llvm::Intrinsic::log10);  // 8
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] exp/log family FLOPs: " << flops << std::endl;
    assert(flops == 40 && "exp+log+exp2+log2+log10 must yield 40 FLOPs");
    std::cout << "[PASS] exp/log family FLOP counting" << std::endl;
}

// New: pow = 16 FLOPs (most expensive — exp(y*log(x)))
void test_pow() {
    ScalarFn fn("test_pow");
    auto *fty = fn.builder.getFloatTy();
    auto &B   = fn.builder;
    auto *powF = llvm::Intrinsic::getDeclaration(fn.mod.get(), llvm::Intrinsic::pow, {fty});
    B.CreateCall(powF, {fn.a, fn.b});  // 16 FLOPs
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*fn.mod);
    std::cout << "[INFO] pow FLOPs: " << flops << std::endl;
    assert(flops == 16 && "pow must yield 16 FLOPs");
    std::cout << "[PASS] pow FLOP counting" << std::endl;
}

// New: <4 x float> sqrt — 4 * 4 = 16 FLOPs (vector scale × intrinsic weight)
void test_sqrt_vector() {
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("test_sqrt_vec", ctx);
    llvm::IRBuilder<> B(ctx);
    auto *vecTy = llvm::FixedVectorType::get(B.getFloatTy(), 4);
    auto *ft = llvm::FunctionType::get(B.getVoidTy(), {vecTy}, false);
    auto *f  = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "sqrt4", mod.get());
    auto *bb = llvm::BasicBlock::Create(ctx, "entry", f);
    B.SetInsertPoint(bb);
    auto *sqrtF = llvm::Intrinsic::getDeclaration(mod.get(), llvm::Intrinsic::sqrt, {vecTy});
    B.CreateCall(sqrtF, {f->getArg(0)});  // 4 FLOPs * 4 lanes = 16
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*mod);
    std::cout << "[INFO] Vector sqrt FLOPs: " << flops << std::endl;
    assert(flops == 16 && "<4 x float> sqrt must yield 16 FLOPs");
    std::cout << "[PASS] Vector sqrt FLOP counting" << std::endl;
}

// New: instruction count excludes phi/alloca/unreachable pseudo-instructions.
// A function with 1 fadd + 1 phi + 1 alloca + 1 unreachable should count
// only 1 real instruction (the fadd) in outInstCount.
void test_inst_count_excludes_pseudos() {
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("test_pseudo", ctx);
    llvm::IRBuilder<> B(ctx);
    auto *fty = llvm::FunctionType::get(B.getVoidTy(), {B.getFloatTy()}, false);
    auto *f   = llvm::Function::Create(fty, llvm::Function::ExternalLinkage, "pseudo", mod.get());

    // Build:  entry → loop (with phi + fadd) → exit (unreachable)
    auto *entryBB = llvm::BasicBlock::Create(ctx, "entry", f);
    auto *loopBB  = llvm::BasicBlock::Create(ctx, "loop",  f);
    auto *exitBB  = llvm::BasicBlock::Create(ctx, "exit",  f);

    // entry: alloca + branch to loop
    B.SetInsertPoint(entryBB);
    B.CreateAlloca(B.getFloatTy());  // pseudo
    B.CreateBr(loopBB);

    // loop: phi + fadd + branch back (infinite loop for test purposes)
    B.SetInsertPoint(loopBB);
    auto *phi = B.CreatePHI(B.getFloatTy(), 2); // pseudo
    auto *arg = f->getArg(0);
    auto *sum = B.CreateFAdd(phi, arg);           // 1 real instruction
    phi->addIncoming(arg, entryBB);
    phi->addIncoming(sum, loopBB);
    B.CreateBr(loopBB);

    // exit: unreachable
    B.SetInsertPoint(exitBB);
    B.CreateUnreachable();  // pseudo

    uint64_t instCount = 0;
    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*mod, &instCount);
    std::cout << "[INFO] Pseudo-inst exclusion — FLOPs=" << flops
              << " instCount=" << instCount << std::endl;
    // The only real instructions are: branch (entry→loop), fadd, branch (loop→loop).
    // phi, alloca, unreachable are excluded.
    assert(flops == 1 && "Only the fadd should be counted as a FLOP");
    // 3 real instructions: br(entry), fadd, br(loop)
    assert(instCount == 3 && "instCount must exclude phi/alloca/unreachable");
    std::cout << "[PASS] Pseudo-instruction exclusion" << std::endl;
}

// New: zero-FP kernel (only integer ops) should produce 0 FLOPs
void test_zero_fp_kernel() {
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("test_zero_fp", ctx);
    llvm::IRBuilder<> B(ctx);
    auto *fty = llvm::FunctionType::get(B.getVoidTy(),
                                        {B.getInt32Ty(), B.getInt32Ty()}, false);
    auto *f   = llvm::Function::Create(fty, llvm::Function::ExternalLinkage, "int_only", mod.get());
    auto *bb  = llvm::BasicBlock::Create(ctx, "entry", f);
    B.SetInsertPoint(bb);
    B.CreateAdd(f->getArg(0), f->getArg(1));  // integer add — not a FLOP
    B.CreateRetVoid();

    uint64_t flops = LLVMTranslationEngine().analyzeStaticFlops(*mod);
    std::cout << "[INFO] Zero-FP kernel FLOPs: " << flops << std::endl;
    assert(flops == 0 && "integer-only kernel must yield 0 FLOPs");
    std::cout << "[PASS] Zero-FP kernel correctly reports 0 FLOPs" << std::endl;
}

// ── driver ───────────────────────────────────────────────────────────────────

int main() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    std::cout << "=== VGRE FLOP Counting Unit Tests ===" << std::endl;
    try {
        // Existing tests (must remain passing)
        test_basic_arithmetic();
        test_fma_vector();

        // New tests covering full instruction taxonomy
        test_frem_and_fneg();
        test_fcmp();
        test_fcmp_vector();
        test_sqrt();
        test_cheap_fp_intrinsics();
        test_sin_cos();
        test_exp_log_family();
        test_pow();
        test_sqrt_vector();
        test_inst_count_excludes_pseudos();
        test_zero_fp_kernel();

        std::cout << "\nAll FLOP counting tests passed!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "\n[FAIL] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
