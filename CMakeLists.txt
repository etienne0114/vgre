cmake_minimum_required(VERSION 3.16)
project(VGRE
    VERSION 0.1.0
    DESCRIPTION "Virtual GPU Runtime Engine"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VGRE_ENABLE_NATIVE_SIMD
    "Enable host-specific SIMD compile flags (AVX/AVX2/AVX512). Leave OFF for portable binaries."
    OFF)

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ── Compiler flags ──────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
    if(VGRE_ENABLE_NATIVE_SIMD)
        # SIMD — enable at the highest level the host supports.
        include(CheckCXXCompilerFlag)
        # Detect host CPU ISA capabilities via /proc/cpuinfo (Linux) so that
        # we only enable instruction-set extensions the build machine can execute.
        # This prevents SIGILL when the compiler supports a flag but the CPU doesn't.
        set(HOST_HAS_AVX512F FALSE)
        set(HOST_HAS_AVX2    FALSE)
        set(HOST_HAS_SSE4    FALSE)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/proc/cpuinfo")
            execute_process(
                COMMAND grep -c "avx512f" /proc/cpuinfo
                OUTPUT_VARIABLE _CPUINFO_AVX512F OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            execute_process(
                COMMAND grep -c " avx2 " /proc/cpuinfo
                OUTPUT_VARIABLE _CPUINFO_AVX2 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            execute_process(
                COMMAND grep -c " sse4_1 " /proc/cpuinfo
                OUTPUT_VARIABLE _CPUINFO_SSE4 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_CPUINFO_AVX512F GREATER "0")
                set(HOST_HAS_AVX512F TRUE)
            endif()
            if(_CPUINFO_AVX2 GREATER "0")
                set(HOST_HAS_AVX2 TRUE)
            endif()
            if(_CPUINFO_SSE4 GREATER "0")
                set(HOST_HAS_SSE4 TRUE)
            endif()
        endif()
        message(STATUS "Host CPU ISA: AVX-512F=${HOST_HAS_AVX512F}  AVX2=${HOST_HAS_AVX2}  SSE4=${HOST_HAS_SSE4}")

        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
        check_cxx_compiler_flag("-mavx2"   COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE4)
        check_cxx_compiler_flag("-maes"    COMPILER_SUPPORTS_AES)

        if(COMPILER_SUPPORTS_AVX512F AND HOST_HAS_AVX512F)
            add_compile_options(-mavx512f -mavx2 -mfma)
            add_compile_definitions(VGRE_HAS_AVX512 VGRE_HAS_AVX2)
        elseif(COMPILER_SUPPORTS_AVX2 AND HOST_HAS_AVX2)
            add_compile_options(-mavx2 -mfma)
            add_compile_definitions(VGRE_HAS_AVX2)
        elseif(COMPILER_SUPPORTS_SSE4 AND HOST_HAS_SSE4)
            add_compile_options(-msse4.1)
            add_compile_definitions(VGRE_HAS_SSE4)
        endif()

        # Intel AMX (Sapphire Rapids / Granite Rapids / Zen 5 with AMX extension)
        # Requires: amx_tile + amx_bf16 in /proc/cpuinfo flags
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/proc/cpuinfo")
            execute_process(
                COMMAND grep -c " amx_tile " /proc/cpuinfo
                OUTPUT_VARIABLE _CPUINFO_AMX_TILE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            execute_process(
                COMMAND grep -c " amx_bf16 " /proc/cpuinfo
                OUTPUT_VARIABLE _CPUINFO_AMX_BF16 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        else()
            set(_CPUINFO_AMX_TILE "0")
            set(_CPUINFO_AMX_BF16 "0")
        endif()
        check_cxx_compiler_flag("-mamx-tile"  COMPILER_SUPPORTS_AMX_TILE)
        check_cxx_compiler_flag("-mamx-bf16"  COMPILER_SUPPORTS_AMX_BF16)
        if(COMPILER_SUPPORTS_AMX_TILE AND COMPILER_SUPPORTS_AMX_BF16
           AND _CPUINFO_AMX_TILE GREATER "0" AND _CPUINFO_AMX_BF16 GREATER "0")
            add_compile_options(-mamx-tile -mamx-bf16)
            add_compile_definitions(VGRE_HAS_AMX)
            message(STATUS "Intel AMX: enabled (-mamx-tile -mamx-bf16)")
        else()
            message(STATUS "Intel AMX: not available (requires Sapphire Rapids CPU + compiler support)")
        endif()

        # AES-NI: enable hardware AES acceleration when available (AES-NI is present
        # on all x86 CPUs since Intel Westmere 2010 / AMD Bulldozer 2011).
        if(COMPILER_SUPPORTS_AES)
            add_compile_options(-maes)
            add_compile_definitions(VGRE_HAS_AES_NI)
            message(STATUS "AES-NI: enabled (-maes)")
        else()
            message(STATUS "AES-NI: not supported by compiler — using software AES-256")
        endif()
    else()
        message(STATUS "SIMD tuning disabled for portability (VGRE_ENABLE_NATIVE_SIMD=OFF)")
    endif()
elseif(MSVC)
    # MSVC: warnings, optimisation, C++ exceptions, and UTF-8 source encoding.
    add_compile_options(/W3 /O2 /EHsc /utf-8)
    if(VGRE_ENABLE_NATIVE_SIMD)
        # Never auto-force aggressive SIMD in MSVC unless explicitly requested.
        # Compiler support alone is insufficient and can crash at runtime on CPUs
        # without the selected ISA (e.g. illegal instruction at process start).
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("/arch:AVX2" MSVC_SUPPORTS_AVX2)
        check_cxx_compiler_flag("/arch:AVX"  MSVC_SUPPORTS_AVX)
        if(MSVC_SUPPORTS_AVX2)
            add_compile_options(/arch:AVX2)
            add_compile_definitions(VGRE_HAS_AVX2)
        elseif(MSVC_SUPPORTS_AVX)
            add_compile_options(/arch:AVX)
        endif()
        message(STATUS "MSVC SIMD (opt-in): AVX2=${MSVC_SUPPORTS_AVX2}  AVX=${MSVC_SUPPORTS_AVX}")
    else()
        message(STATUS "MSVC SIMD tuning disabled for portability (VGRE_ENABLE_NATIVE_SIMD=OFF)")
    endif()
endif()

# ── Dependencies ────────────────────────────────────────────────────────────
# macOS: Apple Clang ships without OpenMP; hint CMake to Homebrew's libomp.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS         "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES     "omp")
        set(OpenMP_omp_LIBRARY       "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "macOS: using Homebrew libomp at ${HOMEBREW_LIBOMP_PREFIX}")
    else()
        message(WARNING "macOS: Homebrew libomp not found — install with: brew install libomp")
    endif()
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(UNIX AND NOT APPLE)
    # find_library caches NOTFOUND on first miss and never retries; use NO_DEFAULT_PATH
    # so cmake searches only our explicit candidate dirs, avoiding the system gomp.
    # Also search via llvm-config to handle non-standard LLVM installs.
    if(NOT LIBOMP_PATH)
        execute_process(
            COMMAND llvm-config-18 --libdir
            OUTPUT_VARIABLE LLVM_LIBDIR OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT LLVM_LIBDIR)
            set(LLVM_LIBDIR "/usr/lib/llvm-18/lib")
        endif()
        find_library(LIBOMP_PATH
            NAMES omp
            HINTS "${LLVM_LIBDIR}"
            PATHS /usr/lib/llvm-18/lib /usr/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH)
        if(NOT LIBOMP_PATH)
            # Last resort: allow system search
            find_library(LIBOMP_PATH NAMES omp)
        endif()
    endif()
    if(LIBOMP_PATH)
        message(STATUS "Found LLVM OpenMP: ${LIBOMP_PATH}")
        link_libraries("-Wl,--no-as-needed" "${LIBOMP_PATH}" "-Wl,--as-needed")
    else()
        message(STATUS "LLVM libomp not found — using system OpenMP (libgomp)")
    endif()
endif()
find_package(Threads REQUIRED)
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(LZ4 REQUIRED liblz4)
 
# ── LLVM & Support Dependencies ─────────────────────────────────────────────
# Manually hint paths for LibEdit/CURL if CMake's default search misses them
find_path(CURL_INCLUDE_DIR NAMES curl/curl.h PATHS /usr/include/x86_64-linux-gnu /usr/local/include)
find_path(LibEdit_INCLUDE_DIRS NAMES editline/readline.h PATHS /usr/include /usr/local/include)
 
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs Core ExecutionEngine OrcJIT Support Target native)

if(WIN32)
    set(VGRE_LLVM_DIA_FALLBACK "C:/Program Files/Microsoft Visual Studio/2022/Community/DIA SDK/lib/amd64/diaguids.lib")
    if(EXISTS "${VGRE_LLVM_DIA_FALLBACK}")
        if(TARGET LLVMDebugInfoPDB)
            set_target_properties(LLVMDebugInfoPDB PROPERTIES
                INTERFACE_LINK_LIBRARIES
                "${VGRE_LLVM_DIA_FALLBACK};LLVMBinaryFormat;LLVMObject;LLVMSupport;LLVMDebugInfoCodeView;LLVMDebugInfoMSF")
        endif()
        foreach(llvm_target IN LISTS llvm_libs)
            if(TARGET ${llvm_target})
                get_target_property(_vgre_iface_libs ${llvm_target} INTERFACE_LINK_LIBRARIES)
                if(_vgre_iface_libs)
                    set(_vgre_updated_iface_libs "${_vgre_iface_libs}")
                    list(TRANSFORM _vgre_updated_iface_libs REPLACE
                        "C:/Program Files \\(x86\\)/Microsoft Visual Studio/2019/Professional/DIA SDK/lib/amd64/diaguids\\.lib"
                        "${VGRE_LLVM_DIA_FALLBACK}")
                    set_target_properties(${llvm_target} PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${_vgre_updated_iface_libs}")
                endif()
            endif()
        endforeach()
    endif()
endif()

set(VGRE_USE_OPENCL_BACKEND_DEFAULT ON)
if(WIN32)
    set(VGRE_USE_OPENCL_BACKEND_DEFAULT OFF)
endif()
option(VGRE_USE_OPENCL_BACKEND "Enable OpenCL Integrated GPU Backend" ${VGRE_USE_OPENCL_BACKEND_DEFAULT})
if(VGRE_USE_OPENCL_BACKEND)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "Found OpenCL ${OpenCL_VERSION_STRING}")
        add_compile_definitions(VGRE_HAS_OPENCL_BACKEND)
    else()
        message(STATUS "OpenCL not found - continuing without the optional OpenCL backend")
        set(VGRE_USE_OPENCL_BACKEND OFF)
    endif()
endif()

# ── RDMA / RoCE optional transport ──────────────────────────────────────────
# Enable with: cmake -DVGRE_ENABLE_RDMA=ON
# Requires: libibverbs-dev (apt install libibverbs-dev rdma-core)
option(VGRE_ENABLE_RDMA "Enable RDMA/RoCE zero-copy cluster transport (requires libibverbs)" OFF)
if(VGRE_ENABLE_RDMA)
    find_library(IBVERBS_LIB NAMES ibverbs)
    find_path(IBVERBS_INCLUDE NAMES infiniband/verbs.h)
    if(IBVERBS_LIB AND IBVERBS_INCLUDE)
        message(STATUS "RDMA: found libibverbs at ${IBVERBS_LIB}")
        add_compile_definitions(VGRE_HAS_RDMA)
        set(VGRE_RDMA_LIBS ${IBVERBS_LIB})
        include_directories(${IBVERBS_INCLUDE})
    else()
        message(FATAL_ERROR
            "VGRE_ENABLE_RDMA=ON but libibverbs not found.\n"
            "Install with: sudo apt install libibverbs-dev rdma-core\n"
            "or disable RDMA: cmake -DVGRE_ENABLE_RDMA=OFF")
    endif()
else()
    message(STATUS "RDMA: disabled (use -DVGRE_ENABLE_RDMA=ON to enable)")
    set(VGRE_RDMA_LIBS "")
endif()

# ── Optional: gRPC / Protobuf cluster transport ───────────────────────────────
# Enable with: cmake -DVGRE_ENABLE_GRPC=ON
# Requires: libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
#   (apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc)
# When enabled: vgre_advanced gains a gRPC service that exposes KernelLaunch,
# MemAlloc, and MemSync RPCs — allowing distributed ML frameworks (Ray Serve,
# Horovod, DeepSpeed) to offload to a VGRE worker over a gRPC channel.
option(VGRE_ENABLE_GRPC "Enable gRPC/Protobuf cluster transport (requires libgrpc++)" OFF)
if(VGRE_ENABLE_GRPC)
    find_package(Protobuf QUIET)
    find_library(GRPC_LIB    NAMES grpc++)
    find_library(GRPCPP_LIB  NAMES grpc++_reflection)
    find_program(PROTOC      NAMES protoc)
    find_program(GRPC_CPP_PLUGIN NAMES grpc_cpp_plugin)
    if(Protobuf_FOUND AND GRPC_LIB AND PROTOC AND GRPC_CPP_PLUGIN)
        message(STATUS "gRPC: found grpc++ at ${GRPC_LIB}")
        add_compile_definitions(VGRE_HAS_GRPC)
        set(VGRE_GRPC_LIBS ${GRPC_LIB})
        if(GRPCPP_LIB)
            list(APPEND VGRE_GRPC_LIBS ${GRPCPP_LIB})
        endif()
        list(APPEND VGRE_GRPC_LIBS protobuf::libprotobuf)
        include_directories(${Protobuf_INCLUDE_DIRS})
        # Generate protobuf + gRPC stubs from vgre_cluster.proto
        set(VGRE_PROTO_DIR  ${PROJECT_SOURCE_DIR}/proto)
        set(VGRE_PROTO_GEN  ${CMAKE_CURRENT_BINARY_DIR}/proto_gen)
        file(MAKE_DIRECTORY ${VGRE_PROTO_GEN})
        set(VGRE_PROTO_SRC  ${VGRE_PROTO_DIR}/vgre_cluster.proto)
        set(VGRE_PROTO_CC   ${VGRE_PROTO_GEN}/vgre_cluster.pb.cc)
        set(VGRE_PROTO_H    ${VGRE_PROTO_GEN}/vgre_cluster.pb.h)
        set(VGRE_GRPC_CC    ${VGRE_PROTO_GEN}/vgre_cluster.grpc.pb.cc)
        set(VGRE_GRPC_H     ${VGRE_PROTO_GEN}/vgre_cluster.grpc.pb.h)
        add_custom_command(
            OUTPUT ${VGRE_PROTO_CC} ${VGRE_PROTO_H} ${VGRE_GRPC_CC} ${VGRE_GRPC_H}
            COMMAND ${PROTOC}
                --proto_path=${VGRE_PROTO_DIR}
                --cpp_out=${VGRE_PROTO_GEN}
                --grpc_out=${VGRE_PROTO_GEN}
                --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                ${VGRE_PROTO_SRC}
            DEPENDS ${VGRE_PROTO_SRC}
            COMMENT "Generating gRPC stubs from vgre_cluster.proto"
        )
        set(VGRE_GRPC_GENERATED ${VGRE_PROTO_CC} ${VGRE_GRPC_CC})
        include_directories(${VGRE_PROTO_GEN})
    else()
        message(FATAL_ERROR
            "VGRE_ENABLE_GRPC=ON but grpc++/protobuf not found.\n"
            "Install: sudo apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc\n"
            "or disable: cmake -DVGRE_ENABLE_GRPC=OFF")
    endif()
else()
    message(STATUS "gRPC transport: disabled (use -DVGRE_ENABLE_GRPC=ON to enable)")
    set(VGRE_GRPC_LIBS "")
    set(VGRE_GRPC_GENERATED "")
endif()

# ── Global include directory ────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ── Sub-projects ────────────────────────────────────────────────────────────
add_subdirectory(src/common)
add_subdirectory(src/core)
add_subdirectory(src/compiler)
add_subdirectory(src/api)
add_subdirectory(src/runtime)
add_subdirectory(src/advanced)

# ── Shared library ──────────────────────────────────────────────────────────
add_library(vgre SHARED
    $<TARGET_OBJECTS:vgre_common>
    $<TARGET_OBJECTS:vgre_core>
    $<TARGET_OBJECTS:vgre_compiler>
    $<TARGET_OBJECTS:vgre_api>
    $<TARGET_OBJECTS:vgre_runtime>
    $<TARGET_OBJECTS:vgre_advanced>
)
if(WIN32)
    # Use an explicit module-definition file for reliable C API exports.
    # WINDOWS_EXPORT_ALL_SYMBOLS does NOT export extern "C" symbols from
    # OBJECT libraries, so we list them in a .def file instead.
    target_sources(vgre PRIVATE ${PROJECT_SOURCE_DIR}/src/api/vgre_api.def)
    set_target_properties(vgre PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # Critical: Ensure VGRE.dll has minimal static initialization (prevent DLL initialization failures)
    set_target_properties(vgre PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# Platform-specific libraries for hardware token storage
if(UNIX AND NOT APPLE)
    # Linux: keyutils for kernel keyring
    find_library(KEYUTILS_LIB keyutils)
    if(KEYUTILS_LIB)
        target_link_libraries(vgre PUBLIC ${KEYUTILS_LIB})
        message(STATUS "Found keyutils: ${KEYUTILS_LIB}")
    else()
        message(WARNING "keyutils not found - hardware token storage will use fallback")
    endif()
    
    # TPM 2.0 support
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(TSS2_ESYS tss2-esys)
        pkg_check_modules(TSS2_MU tss2-mu)
        pkg_check_modules(TSS2_TCTILDR tss2-tctildr)

        if(TSS2_ESYS_FOUND AND TSS2_MU_FOUND AND TSS2_TCTILDR_FOUND)
            add_compile_definitions(VGRE_HAS_TPM2)
            target_include_directories(vgre PUBLIC ${TSS2_ESYS_INCLUDE_DIRS})
            target_link_libraries(vgre PUBLIC
                ${TSS2_ESYS_LIBRARIES}
                ${TSS2_MU_LIBRARIES}
                ${TSS2_TCTILDR_LIBRARIES}
            )
            message(STATUS "TPM 2.0 support enabled")
            message(STATUS "  tss2-esys: ${TSS2_ESYS_VERSION}")
            message(STATUS "  tss2-mu: ${TSS2_MU_VERSION}")
            message(STATUS "  tss2-tctildr: ${TSS2_TCTILDR_VERSION}")
        else()
            message(STATUS "TPM 2.0 libraries not found - TPM support disabled")
        endif()

        # libsecret (GNOME Keyring / KWallet) — Priority 2 token backend on Linux
        pkg_check_modules(LIBSECRET libsecret-1)
        if(LIBSECRET_FOUND)
            add_compile_definitions(VGRE_HAS_LIBSECRET)
            target_include_directories(vgre PUBLIC ${LIBSECRET_INCLUDE_DIRS})
            target_link_libraries(vgre PUBLIC ${LIBSECRET_LIBRARIES})
            message(STATUS "Found libsecret ${LIBSECRET_VERSION}: ${LIBSECRET_LIBRARIES}")
            message(STATUS "  libsecret enabled — GNOME Keyring / KWallet backend active")
        else()
            message(STATUS "libsecret not found - GNOME Keyring backend disabled")
        endif()
    endif()
    
elseif(APPLE)
    # macOS: Security framework for Keychain
    find_library(SECURITY_FRAMEWORK Security)
    if(SECURITY_FRAMEWORK)
        target_link_libraries(vgre PUBLIC ${SECURITY_FRAMEWORK})
        message(STATUS "Found Security framework: ${SECURITY_FRAMEWORK}")
    endif()
elseif(WIN32)
    # Windows: Credential Manager (Advapi32) + Networking (ws2_32) + RNG (bcrypt)
    # bcrypt.lib must be explicit for MinGW — #pragma comment(lib,...) is ignored by GCC.
    target_link_libraries(vgre PUBLIC Advapi32 ws2_32 bcrypt)
endif()

target_link_libraries(vgre PUBLIC
    OpenMP::OpenMP_CXX
    ${OpenMP_CXX_LIBRARIES}
    Threads::Threads
    ${llvm_libs}
)
if(LIBOMP_PATH)
    target_link_libraries(vgre PUBLIC ${LIBOMP_PATH})
endif()
if(UNIX AND NOT APPLE AND LIBOMP_PATH)
    target_link_options(vgre PUBLIC "-Wl,--no-as-needed" "${LIBOMP_PATH}")
endif()
if(VGRE_USE_OPENCL_BACKEND)
    target_link_libraries(vgre PUBLIC OpenCL::OpenCL)
endif()
set_target_properties(vgre PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ── CUDART Shim (LD_PRELOAD wrapper) ────────────────────────────────────────
add_library(vgre_cudart SHARED
    src/api/cudart/cudart_shim.cpp
    src/api/cudart/cudart_shim_stream.cpp
    src/api/cuda_driver/cuda_driver_device_context.cpp
    src/api/cuda_driver/cuda_driver_memory.cpp
    src/api/cuda_driver/cuda_driver_memcpy.cpp
    src/api/cuda_driver/cuda_driver_memset.cpp
    src/api/cuda_driver/cuda_driver_stream_event.cpp
    src/api/cuda_driver/cuda_driver_module.cpp
    src/api/cuda_driver/cuda_driver_occupancy.cpp
    src/api/cuda_driver/cuda_driver_texture.cpp
    src/api/cuda_driver/cuda_driver_errors.cpp
    src/api/cuda_driver/cuda_driver_graph.cpp
    src/api/cuda_driver/cuda_driver_external.cpp
    src/api/cuda_virtual_memory.cpp
    src/api/cuda_external_semaphore.cpp
    src/api/cuda_interceptor.cpp
    src/api/cuda_interceptor_device.cpp
    src/api/cuda_interceptor_memory.cpp
    src/api/cuda_interceptor_graphs.cpp
    src/api/cublas/cublas_core.cpp
    src/api/cublas/cublas_level1.cpp
    src/api/cublas/cublas_level2.cpp
    src/api/cublas/cublas_level2_packed.cpp
    src/api/cublas/cublas_level3.cpp
    src/api/cublas/cublas_level3_blas3.cpp
    src/api/cublas/cublas_complex.cpp
    src/api/cublas/cublas_complex_level2.cpp
    src/api/cublas/cublas_complex_level3.cpp
    src/api/nccl/nccl_core.cpp
    src/api/nccl/nccl_communicator.cpp
    src/api/nccl/nccl_collectives.cpp
    src/api/nccl/nccl_p2p.cpp
    src/api/curand/curand_core.cpp
    src/api/cufft/cufft_core.cpp
    src/api/cusparse/cusparse_core.cpp
    src/api/cusolver/cusolver_core.cpp
    src/api/cublaslt/cublaslt_core.cpp
    src/api/nvtx_shim.cpp
    src/api/vgre_c_api.cpp
    src/api/vgre_c_api_graphs.cpp
    src/api/vgre_c_api_telemetry.cpp
)
if(WIN32)
    set_target_properties(vgre_cudart PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_sources(vgre_cudart PRIVATE
    $<TARGET_OBJECTS:vgre_common>
    $<TARGET_OBJECTS:vgre_core>
    $<TARGET_OBJECTS:vgre_compiler>
    $<TARGET_OBJECTS:vgre_runtime>
    $<TARGET_OBJECTS:vgre_advanced>
    $<TARGET_OBJECTS:vgre_cudart_shims>
)
target_compile_definitions(vgre_cudart PRIVATE VGRE_CUDA_INTERCEPTOR_NO_EXPORTS)
target_link_libraries(vgre_cudart PRIVATE
    OpenMP::OpenMP_CXX
    ${OpenMP_CXX_LIBRARIES}
    Threads::Threads
    ${llvm_libs}
)
if(WIN32)
    target_link_libraries(vgre_cudart PRIVATE Advapi32 ws2_32)
elseif(APPLE)
    find_library(SECURITY_FRAMEWORK Security)
    if(SECURITY_FRAMEWORK)
        target_link_libraries(vgre_cudart PRIVATE ${SECURITY_FRAMEWORK})
    endif()
elseif(UNIX)
    find_library(KEYUTILS_LIB keyutils)
    if(KEYUTILS_LIB)
        target_link_libraries(vgre_cudart PRIVATE ${KEYUTILS_LIB})
    endif()
    # LAPACK / BLAS for cuSOLVER shim
    # On Ubuntu/Debian these live in /usr/lib/x86_64-linux-gnu/{lapack,blas}/
    # Use full paths because target_link_directories doesn't always propagate.
    target_link_libraries(vgre_cudart PRIVATE
        /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3
        /usr/lib/x86_64-linux-gnu/blas/libblas.so.3)
    message(STATUS "Linking cuSOLVER shim against system LAPACK/BLAS")
endif()
if(VGRE_USE_OPENCL_BACKEND)
    target_link_libraries(vgre_cudart PRIVATE OpenCL::OpenCL)
endif()
target_include_directories(vgre_cudart PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(vgre_cudart PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ── Tests ───────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Examples ────────────────────────────────────────────────────────────────
add_subdirectory(examples)

# ── Dashboard (Flutter) ──────────────────────────────────────────────────────
option(VGRE_BUILD_DASHBOARD "Build the Flutter dashboard" ON)
if(VGRE_BUILD_DASHBOARD)
    find_program(FLUTTER_EXECUTABLE flutter)
    if(FLUTTER_EXECUTABLE)
        if(APPLE)
            set(FLUTTER_PLATFORM macos)
            set(FLUTTER_OUT_DIR ${PROJECT_SOURCE_DIR}/vgre_dashboard/build/macos/Build/Products/Release)
            set(LIB_PREFIX "lib")
            set(LIB_SUFFIX ".dylib")
        elseif(WIN32)
            set(FLUTTER_PLATFORM windows)
            set(FLUTTER_OUT_DIR ${PROJECT_SOURCE_DIR}/vgre_dashboard/build/windows/x64/runner/Release)
            set(LIB_PREFIX "")
            set(LIB_SUFFIX ".dll")
        else()
            set(FLUTTER_PLATFORM linux)
            set(FLUTTER_OUT_DIR ${PROJECT_SOURCE_DIR}/vgre_dashboard/build/linux/x64/release/bundle/lib)
            set(LIB_PREFIX "lib")
            set(LIB_SUFFIX ".so")
        endif()

        add_custom_target(vgre_dashboard ALL
            COMMAND ${FLUTTER_EXECUTABLE} build ${FLUTTER_PLATFORM} --release
            
            # Post-build: Copy native binaries into the bundle for true portability
            COMMAND ${CMAKE_COMMAND} -E make_directory ${FLUTTER_OUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vgre> ${FLUTTER_OUT_DIR}/${LIB_PREFIX}vgre${LIB_SUFFIX}
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vgre_cudart> ${FLUTTER_OUT_DIR}/${LIB_PREFIX}vgre_cudart${LIB_SUFFIX}
            
            # True Portability: Bundle JIT headers
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/include/vgre ${FLUTTER_OUT_DIR}/../include/vgre
            
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/vgre_dashboard
            COMMENT "Building VGRE Flutter Dashboard & Bundling native libraries"
            VERBATIM
        )

        if(NOT WIN32 AND NOT APPLE)
            # Desktop Integration: Linux only
            add_custom_command(TARGET vgre_dashboard POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "[Desktop Entry]" > ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Name=VGRE Dashboard" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Comment=Virtual GPU Runtime Engine Dashboard" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Exec=bash -ic \"$ENV{HOME}/.local/share/VGRE/vgre-launch.sh\"" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Icon=$ENV{HOME}/.local/share/VGRE/vgre_icon.png" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Terminal=false" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Type=Application" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E echo "Categories=Development\\;System\\;" >> ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop
                COMMAND ${CMAKE_COMMAND} -E remove $ENV{HOME}/.local/share/applications/vgre.desktop
                COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop $ENV{HOME}/.local/share/applications/vgre-dashboard.desktop
            )
        endif()
        add_dependencies(vgre_dashboard vgre)
    else()
        message(WARNING "flutter executable not found. Dashboard will not be built automatically.")
    endif()
endif()
# ── SLURM GRES Plugin (optional) ────────────────────────────────────────────
option(VGRE_BUILD_SLURM_GRES "Build SLURM GRES plugin" ON)
if(VGRE_BUILD_SLURM_GRES)
    find_path(SLURM_INCLUDE_DIR slurm/slurm.h PATHS /usr/include /usr/local/include)
    if(SLURM_INCLUDE_DIR)
        add_library(vgre_slurm_gres SHARED
            src/deployment/slurm_gres/slurm_gres_vgpu.cpp
        )
        target_include_directories(vgre_slurm_gres PRIVATE
            ${PROJECT_SOURCE_DIR}/src/deployment/slurm_gres
        )
        set_target_properties(vgre_slurm_gres PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION 0
        )
        message(STATUS "Building SLURM GRES plugin")
    else()
        message(STATUS "SLURM headers not found — skipping SLURM GRES plugin")
    endif()
endif()

# ── Install rules ───────────────────────────────────────────────────────────
install(TARGETS vgre vgre_cudart vgre-worker
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/vgre DESTINATION include)
