# §  vgre_xla — XLA/HLO execution engine (the compute core of a PJRT backend)
add_library(vgre_xla OBJECT
    hlo_evaluator.cpp
    xla_c_api.cpp
    thread_pool.cpp
    blas_gemm.cpp
    gemm/intree_gemm_f32.cpp
    autograd/autograd.cpp
    autograd/optim.cpp
    model/transformer.cpp
    model/checkpoint.cpp
    model_c_api.cpp
    autograd_c_api.cpp
    safetensors.cpp
    gguf.cpp
    generation.cpp
    tokenizer.cpp
    parallel.cpp
    parallel_rdma.cpp
    gspmd.cpp
    quant_gemm.cpp
    gptq.cpp
)

target_include_directories(vgre_xla PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# Real BLAS GEMM for Dot/DotGeneral. The def + cblas include go on this OBJECT
# lib (where blas_gemm.cpp compiles); the library itself is linked on the final
# shared targets by _vgre_configure_library(). Without cblas, blas_gemm.cpp
# falls back to a built-in cache-tiled kernel.
if(_VGRE_XLA_BLAS_FOUND)
    target_compile_definitions(vgre_xla PRIVATE VGRE_XLA_HAS_BLAS)
    target_include_directories(vgre_xla PRIVATE "${_VGRE_XLA_BLAS_INCLUDE}")
endif()

set_target_properties(vgre_xla PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Always optimize the compute kernels, even in Debug builds. A numerical GEMM /
# autograd path at -O0 is ~10-15× slower (the packing, accumulation and threading
# scaffolding around the SIMD micro-kernel go unvectorized), which makes the
# whole engine — and the test suite's timing — misleading. The SIMD micro-kernel
# itself selects AVX2/FMA via a per-function target attribute, so no global
# -march is needed. Sanitizer builds set their own flags and are unaffected here.
if(NOT CMAKE_CXX_FLAGS MATCHES "fsanitize")
    if(MSVC)
        set_source_files_properties(
            gemm/intree_gemm_f32.cpp autograd/autograd.cpp model/transformer.cpp
            PROPERTIES COMPILE_OPTIONS "/O2")
    else()
        set_source_files_properties(
            gemm/intree_gemm_f32.cpp autograd/autograd.cpp model/transformer.cpp
            PROPERTIES COMPILE_OPTIONS "-O3")
    endif()
endif()

if(TARGET Threads::Threads)
    target_link_libraries(vgre_xla PUBLIC Threads::Threads)
endif()
# NB: the interpreter parallelizes hot loops with std::thread (see parallelFor in
# hlo_evaluator.cpp), NOT OpenMP — libomp's TLS is incompatible with the
# RTLD_DEEPBIND loading used for in-process TensorFlow coexistence.
