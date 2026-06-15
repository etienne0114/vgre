# §  vgre_xla — XLA/HLO execution engine (the compute core of a PJRT backend)
add_library(vgre_xla OBJECT
    hlo_evaluator.cpp
    xla_c_api.cpp
    thread_pool.cpp
    blas_gemm.cpp
    safetensors.cpp
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

if(TARGET Threads::Threads)
    target_link_libraries(vgre_xla PUBLIC Threads::Threads)
endif()
# NB: the interpreter parallelizes hot loops with std::thread (see parallelFor in
# hlo_evaluator.cpp), NOT OpenMP — libomp's TLS is incompatible with the
# RTLD_DEEPBIND loading used for in-process TensorFlow coexistence.
