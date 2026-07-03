# =============================================================================
# VGRE — Virtual GPU Runtime Engine
# Root CMakeLists.txt  (CMake 3.20+, fully modernised)
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VGREPlatform.cmake)

project(VGRE
    VERSION     0.1.0
    DESCRIPTION "Virtual GPU Runtime Engine — CPU-based CUDA API emulator"
    LANGUAGES   C CXX
)

vgre_apply_macos_link_flags()

# Guard against in-source builds.
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not supported.\n"
        "Create a build directory: cmake -B build && cmake --build build")
endif()

# Build identity for vgre_get_build_info() (Track 5). Captured into a generated
# header so a new commit only recompiles build_info.cpp, not the whole library.
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE VGRE_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT VGRE_GIT_HASH)
    set(VGRE_GIT_HASH "unknown")
endif()
configure_file(${CMAKE_SOURCE_DIR}/cmake/build_config.h.in
               ${CMAKE_BINARY_DIR}/generated/vgre/common/build_config.h @ONLY)
include_directories(${CMAKE_BINARY_DIR}/generated)

# ──────────────────────────────────────────────────────────────────────────────
# §1  CMake modules & module path
# ──────────────────────────────────────────────────────────────────────────────
include(GNUInstallDirs)           # CMAKE_INSTALL_{LIBDIR,BINDIR,INCLUDEDIR}
include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)
include(FeatureSummary)
include(CMakeDependentOption)

# Our cmake/ directory must come first so custom Find* modules take priority
# over CMake's built-in ones (e.g. cmake/FindLAPACK.cmake shadows the stock one).
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ──────────────────────────────────────────────────────────────────────────────
# §2  User-configurable options  (grouped at the top; mark developer flags)
# ──────────────────────────────────────────────────────────────────────────────

# ── Build scope ───────────────────────────────────────────────────────────────
# VGRE_MINIMAL: a lean core-runtime profile for size-constrained deployments —
# it forces every optional integration OFF (see the override block below), so the
# image carries only the CUDA/cuBLAS/JIT core. Enable with -DVGRE_MINIMAL=ON.
option(VGRE_MINIMAL           "Lean core runtime: disable all optional integrations" OFF)
option(VGRE_BUILD_TESTS       "Build the VGRE CTest test suite"          ON)
option(VGRE_BUILD_EXAMPLES    "Build the VGRE example programs"          ON)
option(VGRE_BUILD_DASHBOARD   "Build the Flutter dashboard"              ON)
option(VGRE_BUILD_SLURM_GRES  "Build the SLURM GRES vGPU plugin"        ON)

# ── Feature toggles ───────────────────────────────────────────────────────────
option(VGRE_ENABLE_NATIVE_SIMD
    "Enable host-specific SIMD flags (AVX2/AVX-512). OFF = portable binaries."  ON)
# Windows: LAPACK does not ship by default — require explicit opt-in once the
# user has installed Intel oneAPI MKL or vcpkg OpenBLAS (-DVGRE_USE_LAPACK=ON).
# Linux/macOS: default ON because liblapack-dev / Accelerate are readily available.
if(WIN32)
    set(_vgre_lapack_default OFF)
else()
    set(_vgre_lapack_default ON)
endif()
option(VGRE_USE_LAPACK
    "Link against LAPACK/BLAS for the cuSOLVER shim (Windows: needs Intel oneAPI MKL or vcpkg openblas:x64-windows)"
    ${_vgre_lapack_default})
option(VGRE_USE_FFTW3
    "Use FFTW3 as cuFFT backend; falls back to built-in Cooley-Tukey"    ON)
# Auto-detect libibverbs so RDMA defaults ON when the library is present.
find_library(_rdma_autodetect  NAMES ibverbs)
find_path   (_rdma_inc_autodetect NAMES infiniband/verbs.h)
if(_rdma_autodetect AND _rdma_inc_autodetect)
    set(_VGRE_RDMA_AUTO_DEFAULT ON)
else()
    set(_VGRE_RDMA_AUTO_DEFAULT OFF)
endif()
option(VGRE_ENABLE_RDMA
    "Enable RDMA/RoCE zero-copy cluster transport (requires libibverbs)"
    ${_VGRE_RDMA_AUTO_DEFAULT})
# Auto-detect gRPC availability before the option is evaluated so callers
# don't have to pass -DVGRE_ENABLE_GRPC=ON on systems where it is present.
find_library(_grpc_autodetect NAMES grpc++)
find_program(_grpc_plugin_autodetect NAMES grpc_cpp_plugin)
if(_grpc_autodetect AND _grpc_plugin_autodetect)
    set(_VGRE_GRPC_AUTO_DEFAULT ON)
else()
    set(_VGRE_GRPC_AUTO_DEFAULT OFF)
endif()
option(VGRE_ENABLE_GRPC
    "Enable gRPC/Protobuf cluster transport (requires libgrpc++)"
    ${_VGRE_GRPC_AUTO_DEFAULT})

# Default OpenCL OFF on Windows (CL/cl.h rarely present without GPU SDK)
if(WIN32)
    option(VGRE_USE_OPENCL_BACKEND "Enable OpenCL integrated GPU backend" OFF)
else()
    option(VGRE_USE_OPENCL_BACKEND "Enable OpenCL integrated GPU backend" ON)
endif()

# ── Linux-only optional security backends — auto-detect if deps present ────────
if(UNIX AND NOT APPLE)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(_tpm2_probe   QUIET tss2-esys tss2-mu tss2-tctildr)
        pkg_check_modules(_secret_probe QUIET libsecret-1)
    endif()
    if(_tpm2_probe_FOUND)
        set(_VGRE_TPM2_AUTO_DEFAULT ON)
    else()
        set(_VGRE_TPM2_AUTO_DEFAULT OFF)
    endif()
    if(_secret_probe_FOUND)
        set(_VGRE_LIBSECRET_AUTO_DEFAULT ON)
    else()
        set(_VGRE_LIBSECRET_AUTO_DEFAULT OFF)
    endif()
    option(VGRE_USE_TPM2      "Enable TPM 2.0 hardware security (tss2-esys)"
           ${_VGRE_TPM2_AUTO_DEFAULT})
    option(VGRE_USE_LIBSECRET "Enable GNOME Keyring / KWallet (libsecret-1)"
           ${_VGRE_LIBSECRET_AUTO_DEFAULT})
endif()

# ── CI / quality ──────────────────────────────────────────────────────────────
option(VGRE_WARNINGS_AS_ERRORS "Treat all compiler warnings as errors (for CI)" ON)

# ── Runtime feature gates (default ON — validated; OFF provides rollback path) ─
option(ENABLE_VGRE_TLB
    "Thread-local 4-way set-associative TLB for UVM address translation (Phase 8)"
    ON)
option(ENABLE_VGRE_SPSC
    "Lock-free per-worker SPSC rings for stream task dispatch (Phase 12-A)"
    ON)
option(ENABLE_VGRE_KERNEL_FUSION
    "Graph-level kernel fusion (Flash-Attention, LayerNorm, GEMM+GELU patterns)"
    ON)
option(ENABLE_VGRE_MEMORY_COMPRESSION
    "LZ4 memory compression on host↔device transfers above threshold"
    ON)
option(ENABLE_VGRE_AMX
    "AVX-VNNI INT8 GEMM (Alder Lake+) and AMX-BF16 GEMM (Sapphire Rapids+)"
    ON)
option(VGRE_ENABLE_ZLIB
    "Use system zlib to decompress compressed PTX sections in fat binaries"
    ON)

# ── Minimal profile: force every optional integration OFF ─────────────────────
# Applied after the optional options are declared so it overrides their defaults
# (and any cached value) to give a reproducible lean build. The core runtime
# (CUDA/cuBLAS/cuSOLVER via LAPACK, LLVM JIT, scheduler) is always retained.
if(VGRE_MINIMAL)
    foreach(_opt
            VGRE_BUILD_TESTS VGRE_BUILD_EXAMPLES VGRE_BUILD_DASHBOARD
            VGRE_BUILD_SLURM_GRES VGRE_ENABLE_RDMA VGRE_ENABLE_GRPC
            VGRE_USE_FFTW3 VGRE_USE_OPENCL_BACKEND VGRE_USE_TPM2
            VGRE_USE_LIBSECRET VGRE_ENABLE_ZLIB)
        set(${_opt} OFF CACHE BOOL "disabled by VGRE_MINIMAL" FORCE)
    endforeach()
    message(STATUS "VGRE_MINIMAL=ON → optional integrations disabled (lean core runtime)")
endif()

# Hide developer-only knobs from cmake-gui unless explicitly set
mark_as_advanced(VGRE_WARNINGS_AS_ERRORS VGRE_USE_TPM2 VGRE_USE_LIBSECRET)

# ──────────────────────────────────────────────────────────────────────────────
# §3  C++ standard & global properties
# ──────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No CMAKE_BUILD_TYPE specified — defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# Windows ABI / macro hygiene
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §4  Compiler warning/optimisation helper  (target-scoped, no globals)
# ──────────────────────────────────────────────────────────────────────────────
function(vgre_set_compiler_warnings target_name visibility)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(_flags -Wall -Wextra -Wpedantic -O2)
        if(VGRE_WARNINGS_AS_ERRORS)
            list(APPEND _flags -Werror)
        endif()
    elseif(MSVC)
        set(_flags /W3 /O2 /EHsc /utf-8)
        if(VGRE_WARNINGS_AS_ERRORS)
            list(APPEND _flags /WX)
        endif()
    else()
        return()
    endif()
    target_compile_options(${target_name} ${visibility} ${_flags})
endfunction()

# ──────────────────────────────────────────────────────────────────────────────
# §5  Cross-platform CPU feature & SIMD detection
# ──────────────────────────────────────────────────────────────────────────────
set(VGRE_CPU_FEATURES_DETECTED FALSE)
set(HOST_HAS_AVX512F FALSE)
set(HOST_HAS_AVX2    FALSE)
set(HOST_HAS_SSE4    FALSE)
set(HOST_HAS_AMX     FALSE)
set(HOST_HAS_AES_NI  FALSE)
set(VGRE_SIMD_COMPILE_OPTIONS "")
set(VGRE_SIMD_COMPILE_DEFS    "")

if(VGRE_ENABLE_NATIVE_SIMD)
    # Tier 1 detection via /proc/cpuinfo (replaces cmake_host_system_information
    # PROCESSOR_FEATURES which is unreliable on some Linux distros/kernels).
    set(_host_feats "")

    if("avx512f" IN_LIST _host_feats)
        set(HOST_HAS_AVX512F TRUE)
    endif()
    if("avx2" IN_LIST _host_feats)
        set(HOST_HAS_AVX2 TRUE)
    endif()
    if("sse4_1" IN_LIST _host_feats)
        set(HOST_HAS_SSE4 TRUE)
    endif()
    if("aes" IN_LIST _host_feats)
        set(HOST_HAS_AES_NI TRUE)
    endif()

    # ── Tier 2: Linux /proc/cpuinfo supplement (picks up AMX + cross-check) ──
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" _cpuinfo)
        if(_cpuinfo MATCHES " amx_tile " AND _cpuinfo MATCHES " amx_bf16 ")
            set(HOST_HAS_AMX TRUE)
        endif()
        # Cross-check in case cmake_host_system_information returned empty
        if(NOT HOST_HAS_AVX2   AND _cpuinfo MATCHES " avx2 ")
            set(HOST_HAS_AVX2 TRUE)
        endif()
        if(NOT HOST_HAS_AVX512F AND _cpuinfo MATCHES "avx512f")
            set(HOST_HAS_AVX512F TRUE)
        endif()
        if(NOT HOST_HAS_SSE4   AND _cpuinfo MATCHES " sse4_1 ")
            set(HOST_HAS_SSE4 TRUE)
        endif()
        if(NOT HOST_HAS_AES_NI AND _cpuinfo MATCHES " aes ")
            set(HOST_HAS_AES_NI TRUE)
        endif()
    endif()

    # ── Tier 3: macOS sysctl supplement ──────────────────────────────────────
    if(APPLE)
        if(NOT HOST_HAS_AVX512F)
            execute_process(COMMAND sysctl hw.optional.avx512f
                OUTPUT_VARIABLE _s OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_s MATCHES ": 1$")
                set(HOST_HAS_AVX512F TRUE)
            endif()
        endif()
        if(NOT HOST_HAS_AVX2)
            execute_process(COMMAND sysctl hw.optional.avx2_0
                OUTPUT_VARIABLE _s OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_s MATCHES ": 1$")
                set(HOST_HAS_AVX2 TRUE)
            endif()
        endif()
        if(NOT HOST_HAS_AES_NI)
            execute_process(COMMAND sysctl hw.optional.aes
                OUTPUT_VARIABLE _s OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(_s MATCHES ": 1$")
                set(HOST_HAS_AES_NI TRUE)
            endif()
        endif()
    endif()

    # ── Tier 4: MSVC fallback — conservative, defer to flag validation ────────
    if(MSVC AND NOT _host_feats)
        # PROCESSOR_FEATURES may be unpopulated on some MSVC environments;
        # assume the build machine supports what the compiler will accept.
        set(HOST_HAS_AVX2  TRUE)
        set(HOST_HAS_SSE4  TRUE)
        message(STATUS
            "SIMD: MSVC PROCESSOR_FEATURES empty — using compiler-flag validation only")
    endif()

    set(VGRE_CPU_FEATURES_DETECTED TRUE)
    message(STATUS
        "SIMD host ISA: AVX-512F=${HOST_HAS_AVX512F}  AVX2=${HOST_HAS_AVX2}  "
        "SSE4.1=${HOST_HAS_SSE4}  AMX=${HOST_HAS_AMX}  AES-NI=${HOST_HAS_AES_NI}")

    # ── Resolve compile flags (GCC/Clang) ─────────────────────────────────────
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        check_cxx_compiler_flag("-mavx512f"  _cc_avx512)
        check_cxx_compiler_flag("-mavx2"     _cc_avx2)
        check_cxx_compiler_flag("-msse4.1"   _cc_sse4)
        check_cxx_compiler_flag("-maes"      _cc_aes)
        check_cxx_compiler_flag("-mamx-tile" _cc_amx_tile)
        check_cxx_compiler_flag("-mamx-bf16" _cc_amx_bf16)

        if(_cc_avx512 AND HOST_HAS_AVX512F)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS -mavx512f -mavx2 -mfma)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_AVX512 VGRE_HAS_AVX2)
            message(STATUS "SIMD: enabled AVX-512F + AVX2 + FMA")
        elseif(_cc_avx2 AND HOST_HAS_AVX2)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS -mavx2 -mfma)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_AVX2)
            message(STATUS "SIMD: enabled AVX2 + FMA")
        elseif(_cc_sse4 AND HOST_HAS_SSE4)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS -msse4.1)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_SSE4)
            message(STATUS "SIMD: enabled SSE4.1")
        else()
            message(STATUS "SIMD: no matching SIMD extension (compiler/CPU mismatch)")
        endif()
        if(_cc_amx_tile AND _cc_amx_bf16 AND HOST_HAS_AMX)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS -mamx-tile -mamx-bf16)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_AMX)
            message(STATUS "SIMD: enabled Intel AMX (tile + bf16)")
        endif()
        if(_cc_aes AND HOST_HAS_AES_NI)
            check_cxx_compiler_flag("-mpclmul" _cc_pclmul)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS -maes)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_AES_NI)
            if(_cc_pclmul)
                list(APPEND VGRE_SIMD_COMPILE_OPTIONS -mpclmul)
                list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_PCLMUL)
            endif()
            if(_cc_pclmul)
                message(STATUS "SIMD: enabled AES-NI + PCLMUL")
            else()
                message(STATUS "SIMD: enabled AES-NI")
            endif()
        endif()

    elseif(MSVC)
        check_cxx_compiler_flag("/arch:AVX512" _msvc_avx512)
        check_cxx_compiler_flag("/arch:AVX2"   _msvc_avx2)
        check_cxx_compiler_flag("/arch:AVX"    _msvc_avx)

        if(_msvc_avx512 AND HOST_HAS_AVX512F)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS /arch:AVX512)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_AVX512 VGRE_HAS_AVX2)
            message(STATUS "SIMD: MSVC /arch:AVX512 enabled")
        elseif(_msvc_avx2 AND HOST_HAS_AVX2)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS /arch:AVX2)
            list(APPEND VGRE_SIMD_COMPILE_DEFS VGRE_HAS_AVX2)
            message(STATUS "SIMD: MSVC /arch:AVX2 enabled")
        elseif(_msvc_avx)
            list(APPEND VGRE_SIMD_COMPILE_OPTIONS /arch:AVX)
            message(STATUS "SIMD: MSVC /arch:AVX enabled")
        endif()
    endif()

else()
    message(STATUS "SIMD: disabled for portability (VGRE_ENABLE_NATIVE_SIMD=OFF)")
endif()

# ── Control-flow integrity + stack hardening (docs/missingFeatures.md §1.1) ──
# Intel CET (IBT + shadow-stack markers via -fcf-protection) gives control-flow
# integrity on the security-critical paths; stack-clash + stack-protector guard
# against stack-based corruption. Probed so unsupported toolchains are skipped.
option(VGRE_HARDENING "Enable control-flow integrity (Intel CET) + stack hardening" ON)
if(VGRE_HARDENING AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag("-fcf-protection=full"     _cc_cet)
    check_cxx_compiler_flag("-fstack-clash-protection" _cc_stackclash)
    check_cxx_compiler_flag("-fstack-protector-strong" _cc_ssp)
    if(_cc_cet)
        add_compile_options(-fcf-protection=full)
        message(STATUS "Hardening: Intel CET (-fcf-protection=full) enabled")
    endif()
    if(_cc_stackclash AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_compile_options(-fstack-clash-protection)
    endif()
    if(_cc_ssp)
        add_compile_options(-fstack-protector-strong)
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §6  OpenMP  (consolidated, cross-platform, no global link_libraries())
# ──────────────────────────────────────────────────────────────────────────────

# macOS: hint find_package to Homebrew's libomp BEFORE it runs
if(APPLE)
    find_program(_brew brew)
    if(_brew)
        execute_process(COMMAND "${_brew}" --prefix libomp
            OUTPUT_VARIABLE _brew_libomp OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(NOT _brew_libomp OR NOT EXISTS "${_brew_libomp}")
        if(DEFINED ENV{OpenMP_ROOT} AND EXISTS "$ENV{OpenMP_ROOT}")
            set(_brew_libomp "$ENV{OpenMP_ROOT}")
        endif()
    endif()
    if(_brew_libomp AND EXISTS "${_brew_libomp}")
        set(OpenMP_CXX_FLAGS     "-Xpreprocessor -fopenmp -I${_brew_libomp}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY   "${_brew_libomp}/lib/libomp.dylib")
        # Cache the prefix so vgre_target_link_openmp() can see it even when a
        # target is defined in a subdirectory (function scope inherits a copy of
        # the parent scope at add_subdirectory time, but a CACHE var is always
        # visible). libomp is keg-only on Homebrew: omp.h lives under its own
        # prefix, NOT /opt/homebrew/include, so every target needs the explicit
        # -I or `#include <omp.h>` fails (the macOS CI build failure).
        set(VGRE_APPLE_LIBOMP "${_brew_libomp}" CACHE INTERNAL "Homebrew libomp prefix")
        message(STATUS "OpenMP: Homebrew libomp at ${_brew_libomp}")
    else()
        message(WARNING
            "OpenMP: Homebrew libomp not found on macOS. "
            "Install with: brew install libomp")
    endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Linux: prefer Clang's libomp over libgomp to avoid ABI conflicts with LLVM JIT.
set(_VGRE_LIBOMP_EXTRA "")
if(UNIX AND NOT APPLE)
    if(NOT DEFINED CACHE{LIBOMP_PATH})
        find_program(_llvm_cfg NAMES llvm-config-18 llvm-config)
        if(_llvm_cfg)
            execute_process(COMMAND "${_llvm_cfg}" --libdir
                OUTPUT_VARIABLE _llvm_libdir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        endif()
        find_library(LIBOMP_PATH NAMES omp
            HINTS "${_llvm_libdir}"
            PATHS /usr/lib/llvm-18/lib /usr/lib/llvm-17/lib /usr/lib/x86_64-linux-gnu
            DOC "Path to LLVM/Clang libomp.so")
    endif()
    if(LIBOMP_PATH)
        message(STATUS "OpenMP: using LLVM libomp at ${LIBOMP_PATH}")
        set(_VGRE_LIBOMP_EXTRA "${LIBOMP_PATH}")
        add_library(vgre_llvm_omp SHARED IMPORTED)
        set_property(TARGET vgre_llvm_omp PROPERTY IMPORTED_LOCATION "${LIBOMP_PATH}")
    else()
        message(STATUS "OpenMP: LLVM libomp not found — falling back to system libgomp")
    endif()
endif()

# Helper: attach OpenMP to a target without global state.
# On Linux, --no-as-needed ensures the libomp SONAME is recorded in the
# dynamic section even when no symbols are directly referenced — needed for
# LD_PRELOAD scenarios where the loaded library initialises OpenMP lazily.
function(vgre_target_link_openmp target_name visibility)
    if(MSVC)
        # MSVC (cl.exe / clang-cl with MSVC driver): use /openmp flag.
        # /openmp:llvm enables the Clang-compatible runtime on clang-cl.
        # Fall back to /openmp for plain cl.exe which does not accept /openmp:llvm.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${target_name} ${visibility} /openmp:llvm)
        else()
            target_compile_options(${target_name} ${visibility} /openmp)
        endif()
        # MSVC /openmp links the runtime automatically; no extra lib needed.
    elseif(TARGET OpenMP::OpenMP_CXX)
        # Propagates platform-correct flags (e.g. -Xpreprocessor -fopenmp on Apple Clang).
        target_link_libraries(${target_name} ${visibility} OpenMP::OpenMP_CXX)
    else()
        # GCC and non-driver Clang: use -fopenmp compile flag.
        if(APPLE)
            target_compile_options(${target_name} ${visibility}
                "-Xpreprocessor" "-fopenmp")
        else()
            target_compile_options(${target_name} ${visibility} "-fopenmp")
        endif()

        if(APPLE AND VGRE_APPLE_LIBOMP)
            # Homebrew libomp is keg-only: add its include dir (for <omp.h>) and
            # lib dir (for -lomp) explicitly — Homebrew clang@18 accepts -fopenmp
            # but does not know where the separate libomp keg lives.
            target_include_directories(${target_name} ${visibility}
                "${VGRE_APPLE_LIBOMP}/include")
            target_link_directories(${target_name} ${visibility}
                "${VGRE_APPLE_LIBOMP}/lib")
            target_link_libraries(${target_name} ${visibility} "-lomp")
        elseif(_VGRE_LIBOMP_EXTRA)
            # Force the linker to find libomp by passing the directory and -lomp explicitly.
            get_filename_component(_omp_dir "${_VGRE_LIBOMP_EXTRA}" DIRECTORY)
            target_link_directories(${target_name} ${visibility} "${_omp_dir}")
            target_link_libraries(${target_name} ${visibility} "-lomp")

            if(UNIX AND NOT APPLE)
                target_link_options(${target_name} ${visibility} "LINKER:--no-as-needed")
            endif()
        else()
            # Fallback to whatever the compiler's default OpenMP linker flag is
            target_link_libraries(${target_name} ${visibility} "-fopenmp")
        endif()
    endif()
endfunction()

find_package(Threads REQUIRED)

# ── zlib: optional, used to decompress compressed PTX sections in fat binaries ─
if(VGRE_ENABLE_ZLIB)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        message(STATUS "zlib: found ${ZLIB_VERSION_STRING} — compressed fat-binary PTX enabled")
        set(_VGRE_HAS_ZLIB ON)
    else()
        message(STATUS "zlib: not found — compressed fat-binary PTX sections will not be decompressed")
        set(_VGRE_HAS_ZLIB OFF)
    endif()
else()
    set(_VGRE_HAS_ZLIB OFF)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §6b  Optional: OpenSSL — mTLS, ECDSA-P256, certificate pinning
# ──────────────────────────────────────────────────────────────────────────────
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL: found ${OPENSSL_VERSION} — mTLS / ECDSA / cert-pinning enabled")
    set(_VGRE_HAS_SSL ON)
else()
    message(STATUS "OpenSSL: not found — mTLS disabled; auth-token fallback active")
    set(_VGRE_HAS_SSL OFF)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §7  LLVM (version 18 enforced; target-scoped INTERFACE target)
# ──────────────────────────────────────────────────────────────────────────────
# Version specifier in CONFIG mode is unreliable with distro-packaged LLVM
# (LLVMConfigVersion.cmake may use EXACT compatibility). Find first, then
# assert the minimum version explicitly — same effect, better portability.
find_package(LLVM REQUIRED CONFIG)
if(LLVM_PACKAGE_VERSION VERSION_LESS "18")
    message(FATAL_ERROR
        "LLVM >= 18 is required but found ${LLVM_PACKAGE_VERSION}.\n"
        "Install: sudo apt install llvm-18-dev  "
        "or pass -DLLVM_DIR=/usr/lib/llvm-18/cmake")
endif()
message(STATUS "LLVM: version ${LLVM_PACKAGE_VERSION}  path ${LLVM_DIR}")

# llvm_map_components_to_libnames returns the set of CMake IMPORTED targets
# (e.g. LLVMCore;LLVMSupport;…) needed for these components. CMake resolves
# them to the correct static archives or the monolithic dylib based on the
# LLVM installation type. On Ubuntu/Debian the function returns individual
# component names even when LLVM_LINK_LLVM_DYLIB=ON; CMake's transitive
# dependency handling + --start-group/--end-group resolves the circular
# archive references correctly.
llvm_map_components_to_libnames(VGRE_LLVM_LIBS
    Core ExecutionEngine OrcJIT Support Target native)
message(STATUS "LLVM: component libs = ${VGRE_LLVM_LIBS}")
# Alias: src/compiler/CMakeLists.txt and other subdirs use ${llvm_libs}
# (the historic variable name). Keep it set so their PUBLIC link deps are intact.
set(llvm_libs ${VGRE_LLVM_LIBS})

# INTERFACE IMPORTED target so every consumer gets includes + defs + libs
# via a single target_link_libraries() call.
add_library(vgre_llvm_iface INTERFACE)
target_include_directories(vgre_llvm_iface INTERFACE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(vgre_llvm_iface INTERFACE ${LLVM_DEFINITIONS})
target_link_libraries(vgre_llvm_iface INTERFACE ${VGRE_LLVM_LIBS})

# LLVM requires consistent compile definitions across ALL translation units that
# include LLVM headers (ABI guard macros such as _GNU_SOURCE,
# __STDC_CONSTANT_MACROS, etc.). OBJECT libraries in src/ subdirectories inherit
# these from the directory property rather than needing explicit per-target links.
# This is the one legitimate use of a global scope in this file.
if(LLVM_DEFINITIONS)
    add_compile_definitions(${LLVM_DEFINITIONS})
endif()
# Likewise, subdirectory OBJECT libraries need the LLVM include path
# so their #include <llvm/...> directives resolve during compilation.
include_directories(${LLVM_INCLUDE_DIRS})

# ── Windows: DIA SDK discovery via vswhere + multi-edition/year enumeration ───
if(WIN32)
    set(_dia_path "")

    # Strategy 1: vswhere (covers all editions atomically)
    find_program(_vswhere vswhere
        HINTS
            "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer"
            "$ENV{ProgramFiles}/Microsoft Visual Studio/Installer")
    if(_vswhere)
        execute_process(
            COMMAND "${_vswhere}" -latest -property installationPath
            OUTPUT_VARIABLE _vs_root OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        set(_cand "${_vs_root}/DIA SDK/lib/amd64/diaguids.lib")
        if(EXISTS "${_cand}")
            set(_dia_path "${_cand}")
        endif()
    endif()

    # Strategy 2: enumerate known install trees as fallback (vswhere unavailable).
    # Includes both year-named dirs (2022/2019/2017) and version-numbered dirs
    # (18 = VS 2025, 17 = VS 2022) because Microsoft has used both conventions.
    if(NOT _dia_path)
        foreach(_yr IN ITEMS 2025 2022 2019 2017 18 17 16 15)
            foreach(_ed IN ITEMS Community Professional Enterprise BuildTools Preview)
                set(_cand
                    "$ENV{ProgramFiles}/Microsoft Visual Studio/${_yr}/${_ed}/DIA SDK/lib/amd64/diaguids.lib")
                if(EXISTS "${_cand}")
                    set(_dia_path "${_cand}")
                    break()
                endif()
            endforeach()
            if(_dia_path)
                break()
            endif()
        endforeach()
    endif()

    if(_dia_path)
        message(STATUS "LLVM DIA SDK: found at ${_dia_path}")
        # Normalise backslashes → forward slashes BEFORE using the path as a
        # CMake regex REPLACE string.  The replacement expression is scanned for
        # escape sequences (\1, \\, …); backslashes in a Windows path such as
        # C:\Program Files\... produce unknowns like \P, \M, \V and CMake errors.
        # Forward slashes are accepted by all Windows file-I/O APIs.
        string(REPLACE "\\" "/" _dia_path "${_dia_path}")

        # LLVM's LLVMExports.cmake bakes the absolute DIA SDK path used at LLVM
        # compile-time into LLVMDebugInfoPDB's INTERFACE_LINK_LIBRARIES property.
        # (See LLVM issue #152268, #86250, #111829.)
        # Any machine with a different VS version/edition/install path fails at
        # link time with a missing-file error.
        # We patch the ONLY target known to carry the stale path, plus a safety
        # net that scans every LLVM imported target if LLVM_AVAILABLE_LIBS exists.
        if(TARGET LLVMDebugInfoPDB)
            get_target_property(_iface LLVMDebugInfoPDB INTERFACE_LINK_LIBRARIES)
            if(_iface)
                string(REGEX REPLACE "[^;]*/diaguids\.lib" "${_dia_path}"
                    _iface_fixed "${_iface}")
                if(NOT "${_iface_fixed}" STREQUAL "${_iface}")
                    set_target_properties(LLVMDebugInfoPDB PROPERTIES
                        INTERFACE_LINK_LIBRARIES "${_iface_fixed}")
                endif()
            endif()
        endif()

        # Safety-net: scan all LLVM libraries if LLVM_AVAILABLE_LIBS was set
        # (older LLVM versions may not export it, so this is conditional).
        if(DEFINED LLVM_AVAILABLE_LIBS)
            foreach(_lt IN LISTS LLVM_AVAILABLE_LIBS)
                if(TARGET ${_lt})
                    get_target_property(_iface ${_lt} INTERFACE_LINK_LIBRARIES)
                    if(_iface)
                        string(REGEX REPLACE "[^;]*/diaguids\.lib" "${_dia_path}"
                            _iface_fixed "${_iface}")
                        if(NOT "${_iface_fixed}" STREQUAL "${_iface}")
                            set_target_properties(${_lt} PROPERTIES
                                INTERFACE_LINK_LIBRARIES "${_iface_fixed}")
                        endif()
                    endif()
                endif()
            endforeach()
        endif()
    else()
        message(STATUS "LLVM DIA SDK: not found — PDB symbol generation may be limited")
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §8  LAPACK / BLAS  (cmake/FindLAPACK.cmake — cross-platform)
# ──────────────────────────────────────────────────────────────────────────────
# Strategy:
#   1. If VGRE_USE_LAPACK=ON (Linux/macOS default), try to find system LAPACK.
#      If found → link it.  If NOT found → fall back to built-in stub (never fail).
#   2. If VGRE_USE_LAPACK=OFF (Windows default) → use built-in stub immediately.
# The built-in stub (lapack_fallback.cpp) provides portable C++ implementations
# of all routines needed by cusolver_core.cpp: potrf, getrf, getrs, geqrf,
# ormqr, gesvd, syevd, gelsd — fully functional on every platform.
set(VGRE_LAPACK_FALLBACK FALSE)
if(VGRE_USE_LAPACK)
    find_package(LAPACK)     # optional — no REQUIRED, falls back gracefully
    if(LAPACK_FOUND)
        message(STATUS "LAPACK: ${VGRE_LAPACK_DESCRIPTION}")
    else()
        message(STATUS "LAPACK: requested but not found — using built-in fallback")
        set(VGRE_LAPACK_FALLBACK TRUE)
    endif()
else()
    message(STATUS "LAPACK: disabled (VGRE_USE_LAPACK=OFF) — using built-in fallback")
    set(VGRE_LAPACK_FALLBACK TRUE)
endif()
# Always provide LAPACK::LAPACK interface target so link lines are uniform.
if(NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
endif()

# ── CBLAS for the HLO engine's GEMM (Dot/DotGeneral) ──────────────────────────
# The XLA interpreter routes matmul through cblas_sgemm when a real BLAS with the
# C interface is present; otherwise it uses a built-in cache-tiled kernel (see
# src/xla/blas_gemm.cpp). This is independent of VGRE_USE_LAPACK above — we probe
# directly for cblas.h + a library exporting cblas_sgemm so it works with system
# reference BLAS, OpenBLAS, or MKL.
set(_VGRE_XLA_BLAS_FOUND FALSE)
set(_VGRE_CBLAS_SEARCH_PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include)
set(_VGRE_CBLAS_LIB_SEARCH /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib)
if(APPLE)
    vgre_brew_prefix(openblas _VGRE_OPENBLAS_PREFIX)
    if(_VGRE_OPENBLAS_PREFIX)
        list(APPEND _VGRE_CBLAS_SEARCH_PATHS "${_VGRE_OPENBLAS_PREFIX}/include")
        list(APPEND _VGRE_CBLAS_LIB_SEARCH "${_VGRE_OPENBLAS_PREFIX}/lib")
    endif()
endif()
find_path(_VGRE_CBLAS_INCLUDE cblas.h PATHS ${_VGRE_CBLAS_SEARCH_PATHS})
find_library(_VGRE_CBLAS_LIB
    NAMES openblas cblas blas
    PATHS ${_VGRE_CBLAS_LIB_SEARCH})
if(_VGRE_CBLAS_INCLUDE AND _VGRE_CBLAS_LIB)
    set(_VGRE_XLA_BLAS_FOUND TRUE)
    set(_VGRE_XLA_BLAS_INCLUDE "${_VGRE_CBLAS_INCLUDE}")
    set(_VGRE_XLA_BLAS_LIBS    "${_VGRE_CBLAS_LIB}")
    message(STATUS "XLA GEMM: cblas_sgemm via ${_VGRE_CBLAS_LIB}")
else()
    message(STATUS "XLA GEMM: no cblas — using built-in cache-tiled kernel")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §8a  SQLite3  (Nsight Systems .nsys-rep export — Track Q)
# Accept the versioned runtime (libsqlite3.so.0) when no -dev symlink is present;
# only the ABI-stable C API is used (declared locally in nsight_exporter.cpp).
# Always provide a SQLite::SQLite3 interface target so link lines stay uniform;
# the exporter degrades gracefully (returns an error) if SQLite is unavailable.
# ──────────────────────────────────────────────────────────────────────────────
find_library(SQLITE3_LIBRARY
    NAMES sqlite3 libsqlite3.so.0 libsqlite3.so
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib)
if(APPLE AND NOT SQLITE3_LIBRARY)
    vgre_brew_prefix(sqlite3 _VGRE_SQLITE_PREFIX)
    if(_VGRE_SQLITE_PREFIX)
        find_library(SQLITE3_LIBRARY NAMES sqlite3
            PATHS "${_VGRE_SQLITE_PREFIX}/lib" NO_DEFAULT_PATH)
    endif()
endif()
if(NOT TARGET SQLite::SQLite3)
    add_library(SQLite::SQLite3 INTERFACE IMPORTED)
    if(SQLITE3_LIBRARY)
        target_link_libraries(SQLite::SQLite3 INTERFACE ${SQLITE3_LIBRARY})
        message(STATUS "  SQLite3 for Nsight export: ${SQLITE3_LIBRARY}")
    else()
        message(STATUS "  SQLite3 not found — Nsight .nsys-rep export will be a no-op")
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §8b Optional: SuiteSparse / UMFPACK  (full fill-in sparse factorization)
# ──────────────────────────────────────────────────────────────────────────────
option(VGRE_USE_UMFPACK "Link SuiteSparse UMFPACK for full fill-in sparse LU (cuSPARSE extension)" OFF)
set(_VGRE_UMFPACK_FOUND FALSE)
if(VGRE_USE_UMFPACK)
    find_package(PkgConfig QUIET)
    set(_umfpack_found FALSE)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(UMFPACK QUIET IMPORTED_TARGET SuiteSparse::UMFPACK)
        if(UMFPACK_FOUND)
            set(_umfpack_found TRUE)
        endif()
    endif()
    if(NOT _umfpack_found)
        # Fallback: manual header+lib search
        set(_UMFPACK_INC_PATHS /usr/include/suitesparse /usr/local/include/suitesparse /usr/include)
        set(_UMFPACK_LIB_PATHS /usr/lib /usr/local/lib)
        if(APPLE)
            vgre_brew_prefix(suite-sparse _VGRE_SUITESPARSE_PREFIX)
            if(_VGRE_SUITESPARSE_PREFIX)
                list(APPEND _UMFPACK_INC_PATHS "${_VGRE_SUITESPARSE_PREFIX}/include/suitesparse")
                list(APPEND _UMFPACK_LIB_PATHS "${_VGRE_SUITESPARSE_PREFIX}/lib")
            endif()
        endif()
        find_path(UMFPACK_INCLUDE_DIR umfpack.h PATHS ${_UMFPACK_INC_PATHS})
        find_library(UMFPACK_LIBRARY NAMES umfpack PATHS ${_UMFPACK_LIB_PATHS})
        if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
            if(NOT TARGET UMFPACK::UMFPACK)
                add_library(UMFPACK::UMFPACK UNKNOWN IMPORTED)
                set_target_properties(UMFPACK::UMFPACK PROPERTIES
                    IMPORTED_LOCATION "${UMFPACK_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${UMFPACK_INCLUDE_DIR}")
            endif()
            set(_umfpack_found TRUE)
        endif()
    endif()
    if(_umfpack_found)
        set(_VGRE_UMFPACK_FOUND TRUE)
        message(STATUS "SuiteSparse UMFPACK: found — full fill-in sparse LU enabled")
    else()
        message(STATUS "SuiteSparse UMFPACK: NOT found — falling back to ILU(0)")
    endif()
else()
    message(STATUS "SuiteSparse UMFPACK: disabled (VGRE_USE_UMFPACK=OFF)")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §9  Optional: FFTW3
# ──────────────────────────────────────────────────────────────────────────────
set(_VGRE_FFTW3_FOUND FALSE)
if(VGRE_USE_FFTW3)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(FFTW3  QUIET IMPORTED_TARGET fftw3)
        pkg_check_modules(FFTW3F QUIET IMPORTED_TARGET fftw3f)
    endif()
    if(FFTW3_FOUND AND FFTW3F_FOUND)
        set(_VGRE_FFTW3_FOUND TRUE)
        message(STATUS "FFTW3: found (double ${FFTW3_VERSION} / float ${FFTW3F_VERSION})")
    else()
        message(STATUS
            "FFTW3: not found — cuFFT will use built-in Cooley-Tukey/Bluestein O(n log n)")
    endif()
else()
    message(STATUS "FFTW3: disabled (VGRE_USE_FFTW3=OFF)")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §10  Optional: OpenCL backend
# ──────────────────────────────────────────────────────────────────────────────
set(_VGRE_OPENCL_FOUND FALSE)
if(VGRE_USE_OPENCL_BACKEND)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(_VGRE_OPENCL_FOUND TRUE)
        message(STATUS "OpenCL: found ${OpenCL_VERSION_STRING}")
    else()
        message(STATUS "OpenCL: not found — continuing without the optional backend")
        set(VGRE_USE_OPENCL_BACKEND OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §11  Optional: RDMA / RoCE transport
# ──────────────────────────────────────────────────────────────────────────────
set(_VGRE_RDMA_FOUND FALSE)
set(_VGRE_RDMA_LIBS  "")
if(VGRE_ENABLE_RDMA)
    find_library(IBVERBS_LIB  NAMES ibverbs  DOC "libibverbs shared library")
    find_path(IBVERBS_INCLUDE NAMES infiniband/verbs.h DOC "libibverbs include dir")
    if(IBVERBS_LIB AND IBVERBS_INCLUDE)
        set(_VGRE_RDMA_FOUND TRUE)
        set(_VGRE_RDMA_LIBS  "${IBVERBS_LIB}")
        message(STATUS "RDMA: enabled — libibverbs at ${IBVERBS_LIB}")
    else()
        message(WARNING
            "VGRE_ENABLE_RDMA=ON but libibverbs not found — RDMA disabled.\n"
            "  Install: sudo apt install libibverbs-dev rdma-core\n"
            "  TCP cluster transport remains active.")
        set(VGRE_ENABLE_RDMA OFF CACHE BOOL "" FORCE)
    endif()
else()
    message(STATUS "RDMA: disabled (pass -DVGRE_ENABLE_RDMA=ON to enable)")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §12  Optional: gRPC / Protobuf  (WARNING on failure — not FATAL_ERROR)
# ──────────────────────────────────────────────────────────────────────────────
set(_VGRE_GRPC_FOUND FALSE)
set(_VGRE_GRPC_LIBS  "")
if(VGRE_ENABLE_GRPC)
    find_package(Protobuf QUIET)
    find_library(_grpc_lib     NAMES grpc++)
    find_library(_grpcpp_refl  NAMES grpc++_reflection)
    find_program(_protoc       NAMES protoc)
    find_program(_grpc_plugin  NAMES grpc_cpp_plugin)

    if(Protobuf_FOUND AND _grpc_lib AND _protoc AND _grpc_plugin)
        set(_VGRE_GRPC_FOUND TRUE)
        set(_VGRE_GRPC_LIBS  protobuf::libprotobuf "${_grpc_lib}")
        if(_grpcpp_refl)
            list(APPEND _VGRE_GRPC_LIBS "${_grpcpp_refl}")
        endif()
        message(STATUS "gRPC: found grpc++ at ${_grpc_lib}")

        # Proto code generation
        set(_proto_dir "${PROJECT_SOURCE_DIR}/proto")
        set(_gen_dir   "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
        file(MAKE_DIRECTORY "${_gen_dir}")

        set(_proto_src  "${_proto_dir}/vgre_cluster.proto")
        add_custom_command(
            OUTPUT
                "${_gen_dir}/vgre_cluster.pb.cc"
                "${_gen_dir}/vgre_cluster.pb.h"
                "${_gen_dir}/vgre_cluster.grpc.pb.cc"
                "${_gen_dir}/vgre_cluster.grpc.pb.h"
            COMMAND "${_protoc}"
                "--proto_path=${_proto_dir}"
                "--cpp_out=${_gen_dir}"
                "--grpc_out=${_gen_dir}"
                "--plugin=protoc-gen-grpc=${_grpc_plugin}"
                "${_proto_src}"
            DEPENDS "${_proto_src}"
            COMMENT "Generating gRPC stubs from vgre_cluster.proto"
            VERBATIM
        )
        add_library(vgre_grpc_gen OBJECT
            "${_gen_dir}/vgre_cluster.pb.cc"
            "${_gen_dir}/vgre_cluster.grpc.pb.cc")
        target_include_directories(vgre_grpc_gen PUBLIC "${_gen_dir}")
        target_link_libraries(vgre_grpc_gen PUBLIC ${_VGRE_GRPC_LIBS})
    else()
        # Downgraded from FATAL_ERROR — gRPC is optional
        message(WARNING
            "VGRE_ENABLE_GRPC=ON but gRPC/Protobuf components not found. "
            "Continuing without gRPC cluster transport.\n"
            "  Install: sudo apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc\n"
            "  Disable: cmake -DVGRE_ENABLE_GRPC=OFF")
        set(VGRE_ENABLE_GRPC OFF CACHE BOOL "" FORCE)
    endif()
else()
    message(STATUS "gRPC: disabled (pass -DVGRE_ENABLE_GRPC=ON to enable)")
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §13  Platform security / token-storage libraries  (gathered once, reused)
# ──────────────────────────────────────────────────────────────────────────────
set(_VGRE_SECURITY_LIBS  "")
set(_VGRE_SECURITY_DEFS  "")
set(_VGRE_TPM2_FOUND     FALSE)
set(_VGRE_LIBSECRET_FOUND FALSE)

if(WIN32)
    list(APPEND _VGRE_SECURITY_LIBS Advapi32 ws2_32 bcrypt)

elseif(APPLE)
    find_library(_sec_fw Security)
    if(_sec_fw)
        list(APPEND _VGRE_SECURITY_LIBS "${_sec_fw}")
        message(STATUS "Security: using macOS Security.framework")
    endif()

elseif(UNIX)
    find_library(KEYUTILS_LIB keyutils DOC "libkeyutils for Linux kernel keyring")
    if(KEYUTILS_LIB)
        list(APPEND _VGRE_SECURITY_LIBS "${KEYUTILS_LIB}")
        message(STATUS "Security: found keyutils at ${KEYUTILS_LIB}")
    else()
        message(WARNING "Security: keyutils not found — kernel keyring storage unavailable")
    endif()

    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        # TPM 2.0 (only when explicitly requested)
        if(VGRE_USE_TPM2)
            pkg_check_modules(TSS2_ESYS    QUIET IMPORTED_TARGET tss2-esys)
            pkg_check_modules(TSS2_MU      QUIET IMPORTED_TARGET tss2-mu)
            pkg_check_modules(TSS2_TCTILDR QUIET IMPORTED_TARGET tss2-tctildr)
            if(TSS2_ESYS_FOUND AND TSS2_MU_FOUND AND TSS2_TCTILDR_FOUND)
                list(APPEND _VGRE_SECURITY_LIBS
                    PkgConfig::TSS2_ESYS PkgConfig::TSS2_MU PkgConfig::TSS2_TCTILDR)
                list(APPEND _VGRE_SECURITY_DEFS VGRE_HAS_TPM2)
                set(_VGRE_TPM2_FOUND TRUE)
                message(STATUS "TPM 2.0: enabled (tss2-esys ${TSS2_ESYS_VERSION})")
            else()
                message(WARNING
                    "VGRE_USE_TPM2=ON but TSS2 libraries not found. "
                    "Install: sudo apt install libtss2-dev")
            endif()
        endif()

        # libsecret / GNOME Keyring (only when explicitly requested)
        if(VGRE_USE_LIBSECRET)
            pkg_check_modules(LIBSECRET QUIET IMPORTED_TARGET libsecret-1)
            if(LIBSECRET_FOUND)
                list(APPEND _VGRE_SECURITY_LIBS PkgConfig::LIBSECRET)
                list(APPEND _VGRE_SECURITY_DEFS VGRE_HAS_LIBSECRET)
                set(_VGRE_LIBSECRET_FOUND TRUE)
                message(STATUS "libsecret: enabled (${LIBSECRET_VERSION})")
            else()
                message(WARNING
                    "VGRE_USE_LIBSECRET=ON but libsecret-1 not found. "
                    "Install: sudo apt install libsecret-1-dev")
            endif()
        endif()
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §14  Validated add_subdirectory helper
# ──────────────────────────────────────────────────────────────────────────────
# Skips missing subdirectories with a clear diagnostic rather than a
# cryptic CMake error — useful for partial checkouts and CI scenarios.
function(vgre_add_subdirectory rel_path)
    if(EXISTS "${PROJECT_SOURCE_DIR}/${rel_path}/CMakeLists.txt")
        add_subdirectory("${PROJECT_SOURCE_DIR}/${rel_path}")
    else()
        message(WARNING
            "vgre_add_subdirectory: skipping '${rel_path}' "
            "(${PROJECT_SOURCE_DIR}/${rel_path}/CMakeLists.txt not found)")
    endif()
endfunction()

# ──────────────────────────────────────────────────────────────────────────────
# §15  Shared helper: apply common compile settings to a VGRE library target
# ──────────────────────────────────────────────────────────────────────────────
function(_vgre_configure_library target_name visibility)
    vgre_set_compiler_warnings(${target_name} PRIVATE)
    if(VGRE_SIMD_COMPILE_OPTIONS)
        target_compile_options(${target_name} PRIVATE ${VGRE_SIMD_COMPILE_OPTIONS})
    endif()
    if(VGRE_SIMD_COMPILE_DEFS)
        target_compile_definitions(${target_name} PRIVATE ${VGRE_SIMD_COMPILE_DEFS})
    endif()
    vgre_target_link_openmp(${target_name} ${visibility})
    target_link_libraries(${target_name} ${visibility} Threads::Threads)
    # vgre_llvm_iface is an internal CMake wrapper for LLVM — always PRIVATE
    # so it never appears in export sets (downstream users run find_package(LLVM)).
    target_link_libraries(${target_name} PRIVATE vgre_llvm_iface)
    if(_VGRE_SECURITY_LIBS)
        target_link_libraries(${target_name} PRIVATE ${_VGRE_SECURITY_LIBS})
    endif()
    if(_VGRE_SECURITY_DEFS)
        target_compile_definitions(${target_name} PRIVATE ${_VGRE_SECURITY_DEFS})
    endif()
    if(_VGRE_FFTW3_FOUND)
        target_compile_definitions(${target_name} PRIVATE VGRE_HAS_FFTW3)
        target_link_libraries(${target_name} PRIVATE
            PkgConfig::FFTW3 PkgConfig::FFTW3F)
    endif()
    if(_VGRE_UMFPACK_FOUND)
        target_compile_definitions(${target_name} PRIVATE VGRE_HAS_UMFPACK)
        if(TARGET PkgConfig::UMFPACK)
            target_link_libraries(${target_name} PRIVATE PkgConfig::UMFPACK)
        elseif(TARGET UMFPACK::UMFPACK)
            target_link_libraries(${target_name} PRIVATE UMFPACK::UMFPACK)
        endif()
    endif()
    if(_VGRE_OPENCL_FOUND)
        target_compile_definitions(${target_name} PRIVATE VGRE_HAS_OPENCL_BACKEND)
        target_link_libraries(${target_name} PRIVATE OpenCL::OpenCL)
    endif()
    # XLA GEMM: link the BLAS that provides cblas_sgemm (the compile-time
    # VGRE_XLA_HAS_BLAS def + include live on the vgre_xla OBJECT lib).
    if(_VGRE_XLA_BLAS_FOUND)
        target_link_libraries(${target_name} PRIVATE ${_VGRE_XLA_BLAS_LIBS})
    endif()
    if(_VGRE_RDMA_FOUND)
        target_compile_definitions(${target_name} PRIVATE VGRE_HAS_RDMA)
        target_include_directories(${target_name} PRIVATE "${IBVERBS_INCLUDE}")
        target_link_libraries(${target_name} PRIVATE ${_VGRE_RDMA_LIBS})
    endif()
    if(_VGRE_GRPC_FOUND)
        target_compile_definitions(${target_name} PRIVATE VGRE_HAS_GRPC)
        target_link_libraries(${target_name} PRIVATE
            $<TARGET_OBJECTS:vgre_grpc_gen> ${_VGRE_GRPC_LIBS})
    endif()
    # zlib for compressed fat-binary PTX decompression
    if(_VGRE_HAS_ZLIB)
        target_compile_definitions(${target_name} PRIVATE VGRE_HAS_ZLIB)
        target_link_libraries(${target_name} PRIVATE ZLIB::ZLIB)
    endif()
    # OpenSSL: mTLS, ECDSA-P256 verify, certificate pinning.
    # VGRE_ENABLE_SSL is propagated with ${visibility} so that downstream
    # consumers (tests, examples, user code) that include vgre headers and call
    # the mTLS API see the same definition that the library was compiled with.
    # OpenSSL libraries are PRIVATE because callers do not need to link OpenSSL
    # directly — they go through the VGRE API symbols.
    if(_VGRE_HAS_SSL)
        target_compile_definitions(${target_name} ${visibility} VGRE_ENABLE_SSL)
        target_link_libraries(${target_name} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    endif()
    # Runtime feature gates
    if(ENABLE_VGRE_TLB)
        target_compile_definitions(${target_name} PRIVATE ENABLE_VGRE_TLB)
    endif()
    if(ENABLE_VGRE_SPSC)
        target_compile_definitions(${target_name} PRIVATE ENABLE_VGRE_SPSC)
    endif()
    if(ENABLE_VGRE_KERNEL_FUSION)
        target_compile_definitions(${target_name} PRIVATE ENABLE_VGRE_KERNEL_FUSION)
    endif()
    if(ENABLE_VGRE_MEMORY_COMPRESSION)
        target_compile_definitions(${target_name} PRIVATE ENABLE_VGRE_MEMORY_COMPRESSION)
    endif()
    if(ENABLE_VGRE_AMX)
        target_compile_definitions(${target_name} PRIVATE ENABLE_VGRE_AMX)
    endif()
endfunction()

vgre_add_subdirectory(src/common)
vgre_add_subdirectory(src/core)
vgre_add_subdirectory(src/compiler)
vgre_add_subdirectory(src/api)
vgre_add_subdirectory(src/runtime)
vgre_add_subdirectory(src/advanced)
vgre_add_subdirectory(src/compliance)
vgre_add_subdirectory(src/secrets)
vgre_add_subdirectory(src/identity)
vgre_add_subdirectory(src/backup)
vgre_add_subdirectory(src/mig)
vgre_add_subdirectory(src/pqc)
vgre_add_subdirectory(src/debug)
vgre_add_subdirectory(src/xla)


# ──────────────────────────────────────────────────────────────────────────────
# §16  vgre  (main shared library)
# ──────────────────────────────────────────────────────────────────────────────
add_library(vgre SHARED
    $<TARGET_OBJECTS:vgre_common>
    $<TARGET_OBJECTS:vgre_core>
    $<TARGET_OBJECTS:vgre_compiler>
    $<TARGET_OBJECTS:vgre_api>
    $<TARGET_OBJECTS:vgre_runtime>
    $<TARGET_OBJECTS:vgre_advanced>
    $<TARGET_OBJECTS:vgre_compliance>
    $<TARGET_OBJECTS:vgre_secrets>
    $<TARGET_OBJECTS:vgre_identity>
    $<TARGET_OBJECTS:vgre_backup>
    $<TARGET_OBJECTS:vgre_mig>
    $<TARGET_OBJECTS:vgre_pqc>
    $<TARGET_OBJECTS:vgre_debug>
    $<TARGET_OBJECTS:vgre_xla>
)
set_target_properties(vgre PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME vgre
)
target_include_directories(vgre
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
_vgre_configure_library(vgre PUBLIC)
target_link_libraries(vgre PUBLIC SQLite::SQLite3)  # Track Q: Nsight export

if(WIN32)
    target_sources(vgre PRIVATE "${PROJECT_SOURCE_DIR}/src/api/vgre_api.def")
    set_target_properties(vgre PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    # cdp_executor.cpp uses SymFromAddr from dbghelp for kernel name resolution
    target_link_libraries(vgre PRIVATE dbghelp)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §17  vgre_cudart  (LD_PRELOAD CUDA runtime shim)
# ──────────────────────────────────────────────────────────────────────────────
set(_VGRE_CUDART_SOURCES
    src/api/cudart/cudart_shim.cpp
    src/api/cudart/cudart_shim_stream.cpp
    src/hip/hip_runtime.cpp   # HIP/ROCm compat layer (delegates to the cuda* runtime)
    # cudart_shim_graph_basic.cpp is compiled via vgre_cudart_shims OBJECT lib
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
    src/api/cuda_driver/cuda_driver_tma.cpp
    src/api/cuda_driver/cuda_driver_library.cpp
    src/api/cudart/cudart_proc_address.cpp
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
    src/api/cublas/cublas_gemm_ex.cpp
    src/api/cublas/cublas_complex.cpp
    src/api/cublas/cublas_complex_level2.cpp
    src/api/cublas/cublas_complex_level3.cpp
    src/api/cublas/cublas_hermitian.cpp
    src/api/nccl/nccl_core.cpp
    src/api/nccl/nccl_communicator.cpp
    src/api/nccl/nccl_collectives.cpp
    src/api/nccl/nccl_p2p.cpp
    src/api/curand/curand_core.cpp
    src/api/cufft/cufft_core.cpp
    src/api/cufft/cufft_plan.cpp
    src/api/cufft/cufft_xt.cpp
    src/api/cusparse/cusparse_core.cpp
    src/api/cusparse/cusparse_format.cpp
    src/api/cusparse/cusparse_triangular.cpp
    src/api/cusparse/cusparse_factorization.cpp
    src/api/cusparse/cusparse_spvec.cpp
    src/api/cusparse/sparse_view.cpp
    src/api/cusolver/cusolver_core.cpp
    src/api/cusolver/cusolver_type_erasure.cpp
    $<$<BOOL:${VGRE_LAPACK_FALLBACK}>:src/api/cusolver/lapack_fallback.cpp>
    src/api/cublaslt/cublaslt_core.cpp
    src/api/cublaslt/cublaslt_matmul.cpp
    src/api/nvtx_shim.cpp
    src/api/vgre_c_api.cpp
    src/api/vgre_c_api_graphs.cpp
    src/api/vgre_c_api_telemetry.cpp
    src/api/metrics_server.cpp
    src/api/build_info.cpp
)

# Compile the hot GEMM kernel with AVX2+FMA unconditionally when the compiler
# supports it — the kernel guards itself with #if defined(__AVX2__).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2 -mfma" _HAVE_AVX2_FMA)
if(_HAVE_AVX2_FMA)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/api/cublas/cublas_level3.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
endif()

add_library(vgre_cudart SHARED ${_VGRE_CUDART_SOURCES})

target_sources(vgre_cudart PRIVATE
    $<TARGET_OBJECTS:vgre_common>
    $<TARGET_OBJECTS:vgre_core>
    $<TARGET_OBJECTS:vgre_compiler>
    $<TARGET_OBJECTS:vgre_runtime>
    $<TARGET_OBJECTS:vgre_advanced>
    $<TARGET_OBJECTS:vgre_compliance>
    $<TARGET_OBJECTS:vgre_secrets>
    $<TARGET_OBJECTS:vgre_identity>
    $<TARGET_OBJECTS:vgre_backup>
    $<TARGET_OBJECTS:vgre_mig>
    $<TARGET_OBJECTS:vgre_pqc>
    $<TARGET_OBJECTS:vgre_debug>
    $<TARGET_OBJECTS:vgre_xla>
    $<TARGET_OBJECTS:vgre_cudart_shims>
)
set_target_properties(vgre_cudart PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 0
    OUTPUT_NAME vgre_cudart
)
target_include_directories(vgre_cudart
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
_vgre_configure_library(vgre_cudart PRIVATE)

# cuSOLVER shim — link system LAPACK if available, else compile built-in fallback.
# LAPACK::LAPACK is always a valid target (interface stub when fallback is used).
target_link_libraries(vgre_cudart PRIVATE LAPACK::LAPACK)
target_link_libraries(vgre_cudart PUBLIC SQLite::SQLite3)  # Track Q: Nsight export
# build_info.cpp + nsight_exporter.cpp are also compiled into vgre_cudart; mirror
# the VGRE_HAS_SQLITE define so the SQLite path is enabled there too (Track 5/Q).
if(SQLITE3_LIBRARY)
    target_compile_definitions(vgre_cudart PRIVATE VGRE_HAS_SQLITE)
endif()
if(VGRE_LAPACK_FALLBACK)
    target_compile_definitions(vgre_cudart PRIVATE VGRE_LAPACK_FALLBACK)
endif()

if(WIN32)
    set_target_properties(vgre_cudart PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    # cdp_executor.cpp uses SymFromAddr from dbghelp for kernel name resolution
    target_link_libraries(vgre_cudart PRIVATE dbghelp)
endif()

# macOS: cupti_shim.cpp probes IOKit/CoreFoundation to detect a physical GPU.
if(APPLE)
    find_library(_vgre_iokit IOKit)
    find_library(_vgre_corefoundation CoreFoundation)
    if(_vgre_iokit AND _vgre_corefoundation)
        target_link_libraries(vgre        PRIVATE ${_vgre_iokit} ${_vgre_corefoundation})
        target_link_libraries(vgre_cudart PRIVATE ${_vgre_iokit} ${_vgre_corefoundation})
        message(STATUS "macOS: linking IOKit + CoreFoundation for GPU detection")
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §17b  In-process coexistence with TensorFlow / jaxlib (LLVM/protobuf collision)
# ──────────────────────────────────────────────────────────────────────────────
# libvgre statically links LLVM-18 (the JIT) and dynamically links system
# protobuf/abseil (gRPC transport). TensorFlow bundles its OWN LLVM + protobuf
# with global symbols. If TF is loaded into the process first, libvgre's LLVM
# static initializers / protobuf imports would bind to TF's incompatible copies
# and segfault at dlopen.
#
# The fix lives on the *loading* side, not here: the Python loader opens libvgre
# with RTLD_DEEPBIND | RTLD_LOCAL (see frameworks/jax/vgre_jax/runtime.py).
# DEEPBIND makes libvgre prefer its own dependency tree over already-loaded
# globals (so it uses its own LLVM + the system protobuf, not TF's), and
# RTLD_LOCAL keeps libvgre's symbols out of the global scope so it never
# interposes anyone else. This is robust to load order and needs no linker symbol
# hiding — which is important because some unit tests (e.g. test_flop_counting)
# deliberately share libvgre's exported LLVM symbols.

# ──────────────────────────────────────────────────────────────────────────────
# §18  Test helper function  (used by tests/CMakeLists.txt)
# ──────────────────────────────────────────────────────────────────────────────
#
# vgre_register_test(<test_name> <source_file> <link_target>
#     [TIMEOUT <seconds>]
#     [RESOURCE_LOCK <lock_name>]
#     [FIXTURE_REQUIRES <fixture_name>]
#     [FIXTURE_SETUP    <fixture_name>]
#     [LABELS <label...>]
#     [EXTRA_SOURCES <src...>]
#     [EXTRA_INCLUDES <dir...>]
#     [WORKING_DIRECTORY <dir>])
#
# Eliminates the ~300-line repetition of identical add_executable /
# target_link_libraries / add_test / set_tests_properties blocks.
function(vgre_register_test test_name source_file link_target)
    cmake_parse_arguments(_T
        ""
        "TIMEOUT;RESOURCE_LOCK;FIXTURE_REQUIRES;FIXTURE_SETUP;WORKING_DIRECTORY"
        "LABELS;EXTRA_SOURCES;EXTRA_INCLUDES"
        ${ARGN}
    )

    add_executable(${test_name} "${source_file}" ${_T_EXTRA_SOURCES})
    target_link_libraries(${test_name} PRIVATE ${link_target})

    if(_T_EXTRA_INCLUDES)
        target_include_directories(${test_name} PRIVATE ${_T_EXTRA_INCLUDES})
    endif()

    if(_T_WORKING_DIRECTORY)
        add_test(NAME ${test_name} COMMAND ${test_name}
            WORKING_DIRECTORY "${_T_WORKING_DIRECTORY}")
    else()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()

    set(_props "")
    if(_T_TIMEOUT)
        list(APPEND _props TIMEOUT ${_T_TIMEOUT})
    endif()
    if(_T_RESOURCE_LOCK)
        list(APPEND _props RESOURCE_LOCK ${_T_RESOURCE_LOCK})
    endif()
    if(_T_FIXTURE_REQUIRES)
        list(APPEND _props FIXTURES_REQUIRED ${_T_FIXTURE_REQUIRES})
    endif()
    if(_T_FIXTURE_SETUP)
        list(APPEND _props FIXTURES_SETUP ${_T_FIXTURE_SETUP})
    endif()
    if(_T_LABELS)
        list(APPEND _props LABELS "${_T_LABELS}")
    endif()
    if(_props)
        set_tests_properties(${test_name} PROPERTIES ${_props})
    endif()
endfunction()

# ──────────────────────────────────────────────────────────────────────────────
# §19  Tests & examples
# ──────────────────────────────────────────────────────────────────────────────
if(VGRE_BUILD_TESTS)
    enable_testing()
    vgre_add_subdirectory(tools/fuzzing)  # fuzz target cores (before tests, which link them)
    vgre_add_subdirectory(tests)
endif()

if(VGRE_BUILD_EXAMPLES)
    vgre_add_subdirectory(examples)
    vgre_add_subdirectory(tools)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §20  Flutter dashboard
# ──────────────────────────────────────────────────────────────────────────────
if(VGRE_BUILD_DASHBOARD)
    find_program(FLUTTER_EXECUTABLE flutter DOC "Flutter SDK 'flutter' binary")
    if(NOT FLUTTER_EXECUTABLE)
        message(WARNING
            "VGRE_BUILD_DASHBOARD=ON but 'flutter' not found in PATH. "
            "Dashboard build skipped. Install Flutter: https://flutter.dev/docs/get-started/install")
    elseif(NOT EXISTS "${PROJECT_SOURCE_DIR}/vgre_dashboard/pubspec.yaml")
        message(WARNING
            "Flutter dashboard source not found at "
            "'${PROJECT_SOURCE_DIR}/vgre_dashboard/pubspec.yaml'. "
            "Dashboard build skipped.")
    else()
        if(APPLE)
            set(_flt_platform macos)
            set(_flt_out "${PROJECT_SOURCE_DIR}/vgre_dashboard/build/macos/Build/Products/Release")
            set(_lib_pfx "lib")
            set(_lib_sfx ".dylib")
        elseif(WIN32)
            set(_flt_platform windows)
            set(_flt_out "${PROJECT_SOURCE_DIR}/vgre_dashboard/build/windows/x64/runner/Release")
            set(_lib_pfx "")
            set(_lib_sfx ".dll")
        else()
            set(_flt_platform linux)
            set(_flt_out "${PROJECT_SOURCE_DIR}/vgre_dashboard/build/linux/x64/release/bundle/lib")
            set(_lib_pfx "lib")
            set(_lib_sfx ".so")
        endif()

        add_custom_target(vgre_dashboard ALL
            COMMAND "${FLUTTER_EXECUTABLE}" build "${_flt_platform}" --release
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_flt_out}"
            COMMAND "${CMAKE_COMMAND}" -E copy
                "$<TARGET_FILE:vgre>"
                "${_flt_out}/${_lib_pfx}vgre${_lib_sfx}"
            COMMAND "${CMAKE_COMMAND}" -E copy
                "$<TARGET_FILE:vgre_cudart>"
                "${_flt_out}/${_lib_pfx}vgre_cudart${_lib_sfx}"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${PROJECT_SOURCE_DIR}/include/vgre"
                "${_flt_out}/../include/vgre"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/vgre_dashboard"
            DEPENDS vgre vgre_cudart
            COMMENT "Building VGRE Flutter dashboard and bundling native libraries"
            VERBATIM
        )

        if(UNIX AND NOT APPLE)
            set(_desktop_dst "${PROJECT_SOURCE_DIR}/vgre-dashboard.desktop")
            add_custom_command(TARGET vgre_dashboard POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E echo "[Desktop Entry]"              > "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo "Name=VGRE Dashboard"         >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo
                    "Comment=Virtual GPU Runtime Engine Dashboard"                >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo
                    "Exec=bash -ic \"$ENV{HOME}/.local/share/VGRE/vgre-launch.sh\"" >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo
                    "Icon=$ENV{HOME}/.local/share/VGRE/vgre_icon.png"            >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo "Terminal=false"              >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo "Type=Application"            >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E echo "Categories=Development;System;" >> "${_desktop_dst}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "$ENV{HOME}/.local/share/applications"
                COMMAND "${CMAKE_COMMAND}" -E copy
                    "${_desktop_dst}"
                    "$ENV{HOME}/.local/share/applications/vgre-dashboard.desktop"
                VERBATIM
            )
        endif()
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §21  SLURM GRES plugin
# ──────────────────────────────────────────────────────────────────────────────
if(VGRE_BUILD_SLURM_GRES)
    set(_slurm_src
        "${PROJECT_SOURCE_DIR}/src/deployment/slurm_gres/slurm_gres_vgpu.cpp")
    if(NOT EXISTS "${_slurm_src}")
        message(STATUS "SLURM GRES: source not present — skipping")
    else()
        find_path(SLURM_INCLUDE_DIR
            NAMES slurm/slurm.h
            PATHS /usr/include /usr/local/include /opt/slurm/include
            DOC "SLURM header directory")
        if(NOT SLURM_INCLUDE_DIR)
            message(STATUS
                "SLURM GRES: slurm/slurm.h not found — skipping. "
                "Install: sudo apt install libslurm-dev")
        else()
            add_library(vgre_slurm_gres SHARED "${_slurm_src}")
            target_include_directories(vgre_slurm_gres PRIVATE
                "${PROJECT_SOURCE_DIR}/src/deployment/slurm_gres"
                "${SLURM_INCLUDE_DIR}")
            set_target_properties(vgre_slurm_gres PROPERTIES
                VERSION ${PROJECT_VERSION}
                SOVERSION 0)
            message(STATUS "SLURM GRES: plugin enabled (${SLURM_INCLUDE_DIR})")
        endif()
    endif()
endif()

# ──────────────────────────────────────────────────────────────────────────────
# §21b  Kubernetes VGRE Operator  (Go, optional)
# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect Go toolchain for K8s operator build.
find_program(_go_autodetect go)
set(_k8s_operator_dir "${PROJECT_SOURCE_DIR}/src/deployment/vgre_operator")
if(_go_autodetect AND EXISTS "${_k8s_operator_dir}/go.mod")
    set(_VGRE_K8S_AUTO_DEFAULT ON)
else()
    set(_VGRE_K8S_AUTO_DEFAULT OFF)
endif()
option(VGRE_BUILD_K8S_OPERATOR "Build the Kubernetes VGRE Operator (requires Go 1.22+)"
       ${_VGRE_K8S_AUTO_DEFAULT})
set(_VGRE_K8S_OPERATOR_FOUND FALSE)
if(VGRE_BUILD_K8S_OPERATOR)
    find_program(GO_EXECUTABLE go DOC "Go toolchain 'go' binary")
    set(_operator_dir "${PROJECT_SOURCE_DIR}/src/deployment/vgre_operator")
    if(NOT GO_EXECUTABLE)
        message(WARNING
            "VGRE_BUILD_K8S_OPERATOR=ON but 'go' not found in PATH. "
            "Operator build skipped. Install Go: https://go.dev/dl/")
    elseif(NOT EXISTS "${_operator_dir}/go.mod")
        message(WARNING
            "VGRE_BUILD_K8S_OPERATOR=ON but go.mod not found at "
            "'${_operator_dir}/go.mod'. Operator build skipped.")
    else()
        set(_VGRE_K8S_OPERATOR_FOUND TRUE)
        add_custom_target(vgre_operator ALL
            COMMAND "${GO_EXECUTABLE}" build -o "${CMAKE_CURRENT_BINARY_DIR}/vgre-operator" main.go
            WORKING_DIRECTORY "${_operator_dir}"
            COMMENT "Building VGRE Kubernetes Operator (Go)"
            VERBATIM
        )
        message(STATUS "K8s Operator: enabled (${_operator_dir})")
    endif()
else()
    message(STATUS "K8s Operator: disabled (pass -DVGRE_BUILD_K8S_OPERATOR=ON to enable)")
endif()

# ── Footprint report (Track 20) ───────────────────────────────────────────────
# `cmake --build build --target vgre_footprint` prints the runtime library sizes
# (as built and stripped) plus the active optional-feature profile, so the lean
# VGRE_MINIMAL build can be measured against the full one.
add_custom_target(vgre_footprint
    COMMAND ${CMAKE_COMMAND} -E echo
        "Profile: minimal=${VGRE_MINIMAL} grpc=${VGRE_ENABLE_GRPC} rdma=${VGRE_ENABLE_RDMA} opencl=${VGRE_USE_OPENCL_BACKEND} fftw3=${VGRE_USE_FFTW3} zlib=${VGRE_ENABLE_ZLIB} lapack=${VGRE_USE_LAPACK}"
    COMMAND bash ${CMAKE_SOURCE_DIR}/tools/footprint_report.sh
        $<TARGET_FILE:vgre> $<TARGET_FILE:vgre_cudart>
    DEPENDS vgre vgre_cudart
    VERBATIM)

# ──────────────────────────────────────────────────────────────────────────────
# §22  Install rules  (GNUInstallDirs — no hardcoded lib/bin/include paths)
# ──────────────────────────────────────────────────────────────────────────────
install(TARGETS vgre vgre_cudart vgre-worker
    EXPORT   VGRETargets
    LIBRARY  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME  DESTINATION "${CMAKE_INSTALL_BINDIR}"
    ARCHIVE  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/vgre"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
if(TARGET vgre_slurm_gres)
    install(TARGETS vgre_slurm_gres
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

# CMake package config so downstream projects can use find_package(VGRE)
install(EXPORT VGRETargets
    FILE       VGRETargets.cmake
    NAMESPACE  VGRE::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VGRE"
)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/VGREConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/VGREConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VGRE"
    NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VGREConfigVersion.cmake"
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VGREConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VGREConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/VGRE"
)

# ──────────────────────────────────────────────────────────────────────────────
# §23  Configure summary matrix
# ──────────────────────────────────────────────────────────────────────────────
# Helper: print a single feature row with aligned ON/OFF indicator.
function(_vgre_row label flag)
    if(${flag})
        message(STATUS "  ${label}ON")
    else()
        message(STATUS "  ${label}OFF")
    endif()
endfunction()

string(REPEAT "─" 64 _SEP)
message(STATUS "${_SEP}")
message(STATUS "  VGRE ${PROJECT_VERSION}  ·  ${CMAKE_BUILD_TYPE}  ·  CMake ${CMAKE_VERSION}")
message(STATUS "${_SEP}")
message(STATUS "  Toolchain")
message(STATUS "    CXX compiler  : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "    Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "    Lib dir       : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "${_SEP}")
message(STATUS "  Feature                            Status")
message(STATUS "  ─────────────────────────────────────────")
_vgre_row("  Native SIMD (AVX2/512)      →   " VGRE_ENABLE_NATIVE_SIMD)
_vgre_row("  TLB cache (UVM fast-path)   →   " ENABLE_VGRE_TLB)
_vgre_row("  SPSC per-worker rings       →   " ENABLE_VGRE_SPSC)
_vgre_row("  Kernel fusion               →   " ENABLE_VGRE_KERNEL_FUSION)
_vgre_row("  Memory compression (LZ4)   →   " ENABLE_VGRE_MEMORY_COMPRESSION)
_vgre_row("  AVX-VNNI / AMX acceleration →  " ENABLE_VGRE_AMX)
_vgre_row("  LAPACK / BLAS               →   " VGRE_USE_LAPACK)
_vgre_row("  FFTW3 backend               →   " _VGRE_FFTW3_FOUND)
_vgre_row("  OpenCL backend              →   " _VGRE_OPENCL_FOUND)
_vgre_row("  RDMA / RoCE                 →   " _VGRE_RDMA_FOUND)
_vgre_row("  gRPC cluster transport      →   " _VGRE_GRPC_FOUND)
if(UNIX AND NOT APPLE)
_vgre_row("  TPM 2.0                     →   " _VGRE_TPM2_FOUND)
_vgre_row("  libsecret (GNOME Keyring)   →   " _VGRE_LIBSECRET_FOUND)
endif()
_vgre_row("  Tests                       →   " VGRE_BUILD_TESTS)
_vgre_row("  Examples                    →   " VGRE_BUILD_EXAMPLES)
_vgre_row("  Flutter dashboard           →   " VGRE_BUILD_DASHBOARD)
_vgre_row("  K8s VGRE Operator           →   " _VGRE_K8S_OPERATOR_FOUND)
_vgre_row("  Warnings-as-errors (CI)     →   " VGRE_WARNINGS_AS_ERRORS)
message(STATUS "  ─────────────────────────────────────────")
message(STATUS "  LLVM version : ${LLVM_PACKAGE_VERSION}")
if(VGRE_USE_LAPACK AND LAPACK_FOUND)
message(STATUS "  LAPACK       : ${VGRE_LAPACK_DESCRIPTION}")
endif()
if(VGRE_ENABLE_NATIVE_SIMD)
message(STATUS "  CPU ISA      : AVX512=${HOST_HAS_AVX512F}  AVX2=${HOST_HAS_AVX2}  "
               "SSE4=${HOST_HAS_SSE4}  AMX=${HOST_HAS_AMX}  AES-NI=${HOST_HAS_AES_NI}")
endif()
message(STATUS "${_SEP}")
