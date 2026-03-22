#include <iostream>
#include <cassert>
#include "vgre/compiler/clang_kernel_parser.h"

using namespace vgre;
using namespace vgre::compiler;

void test_basic_kernel() {
    ClangKernelParser parser;
    KernelIR ir;
    std::string source = R"(
        #include "vgre/compiler/cpu_cuda_env.h"
        __global__ void vectorAdd(float* a, float* b, float* c, int n) {
            int i = 0;
        }
    )";
    
    VGREResult result = parser.parse("vectorAdd", source, ir);
    assert(result == VGREResult::SUCCESS);
    assert(ir.name == "vectorAdd");
    assert(ir.argTypes.size() == 4);
    assert(ir.argTypes[0] == ArgType::POINTER);
    assert(ir.argTypes[3] == ArgType::INT32);
    assert(ir.argTypeNames[0] == "float *");
    assert(ir.argTypeNames[3] == "int");
    std::cout << "[PASS] Basic Kernel Parsing" << std::endl;
}

void test_templated_kernel() {
    ClangKernelParser parser;
    KernelIR ir;
    std::string source = R"(
        #include "vgre/compiler/cpu_cuda_env.h"
        template<typename T>
        __global__ void templateAdd(T* a, T* b, int n) {
            int i = 0;
        }
        
        template __global__ void templateAdd<float>(float*, float*, int);
    )";
    
    VGREResult result = parser.parse("templateAdd", source, ir);
    assert(result == VGREResult::SUCCESS);
    assert(ir.name == "templateAdd");
    assert(ir.argTypes.size() == 3);
    assert(ir.argTypes[0] == ArgType::POINTER);
    assert(ir.argTypes[2] == ArgType::INT32);
    std::cout << "[PASS] Templated Kernel Parsing" << std::endl;
}

int main() {
    std::cout << "=== VGRE ClangKernelParser Unit Tests ===" << std::endl;
    test_basic_kernel();
    test_templated_kernel();
    std::cout << "\nAll ClangKernelParser tests passed!" << std::endl;
    return 0;
}
