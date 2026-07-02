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
    vector_index.cpp
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

# ── libvgre_nn — lightweight, LLVM-free, dependency-free ML framework ─────────
# The in-tree ML stack (SIMD GEMM + autograd + optimizers + VGRE-LM + tokenizer +
# checkpoints, with the vgre.nn C-ABI) uses no LLVM, no BLAS and no CUDA — LLVM is
# only needed by the CUDA-JIT path, which this library deliberately excludes. It
# ships as its own small shared object so "import vgre.nn; train a model" works on
# any platform with just a C++ runtime + threads. Distributed training uses a
# real, self-contained TCP all-reduce (cluster_collective.cpp) — no LLVM, no core
# runtime — configured via VGRE_NN_WORLD_SIZE/RANK/MASTER_HOST/MASTER_PORT. The
# full libvgre instead routes through its richer TCP/RDMA TCPClusterManager.
option(VGRE_BUILD_NN_LIB "Build the lightweight LLVM-free ML library (libvgre_nn)" ON)
if(VGRE_BUILD_NN_LIB)
    add_library(vgre_nn SHARED
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
        hlo_evaluator.cpp
        xla_c_api.cpp
        parallel.cpp
        gspmd.cpp
        cluster_collective.cpp
        vector_index.cpp
        ${PROJECT_SOURCE_DIR}/src/common/json.cpp
    )
    target_include_directories(vgre_nn PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    if(TARGET Threads::Threads)
        target_link_libraries(vgre_nn PUBLIC Threads::Threads)
    endif()
    if(_VGRE_XLA_BLAS_FOUND)
        # Optional: faster GEMM if a system BLAS is present, but NOT required.
        target_compile_definitions(vgre_nn PRIVATE VGRE_XLA_HAS_BLAS)
        target_include_directories(vgre_nn PRIVATE "${_VGRE_XLA_BLAS_INCLUDE}")
    endif()
    set_target_properties(vgre_nn PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(NOT CMAKE_CXX_FLAGS MATCHES "fsanitize")
        set_source_files_properties(gemm/intree_gemm_f32.cpp autograd/autograd.cpp
            model/transformer.cpp PROPERTIES COMPILE_OPTIONS
            $<IF:$<BOOL:${MSVC}>,/O2,-O3>)
    endif()
endif()
