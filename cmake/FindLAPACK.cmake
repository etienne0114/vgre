# =============================================================================
# cmake/FindLAPACK.cmake — VGRE Cross-Platform LAPACK/BLAS Discovery
# =============================================================================
# Intentionally shadows CMake's built-in FindLAPACK to provide paths that
# the stock module does not cover: Ubuntu multiarch lib subdirs,
# macOS Accelerate, Windows Intel MKL/vcpkg OpenBLAS.
#
# Search order:
#   macOS   → Accelerate.framework → Homebrew OpenBLAS
#   Windows → MKLROOT env / oneAPI install → vcpkg OpenBLAS → MinGW BLAS
#   Linux   → pkg-config → OpenBLAS → Debian multiarch → MKL → generic
#
# Exports:
#   LAPACK_FOUND             - TRUE when a usable implementation was located
#   LAPACK_LIBRARIES         - Full list of library paths/flags to link
#   LAPACK_INCLUDE_DIRS      - Include directories (non-empty for MKL only)
#   VGRE_LAPACK_DESCRIPTION  - One-line description used in the configure log
#   LAPACK::LAPACK           - INTERFACE IMPORTED target (use this in
#                              target_link_libraries calls)
# =============================================================================

include(FindPackageHandleStandardArgs)

# Guard: avoid repeated work on re-configuration.
if(TARGET LAPACK::LAPACK)
    set(LAPACK_FOUND TRUE)
    return()
endif()

set(LAPACK_FOUND        FALSE)
set(LAPACK_LIBRARIES    "")
set(LAPACK_INCLUDE_DIRS "")
set(VGRE_LAPACK_DESCRIPTION "not found")

# Internal helper — creates the IMPORTED target and stamps result variables.
# Call this exactly once when a strategy succeeds.
function(_vgre_lapack_commit libs incs desc)
    if(NOT TARGET LAPACK::LAPACK)
        add_library(LAPACK::LAPACK INTERFACE IMPORTED GLOBAL)
    endif()
    target_link_libraries(LAPACK::LAPACK INTERFACE ${libs})
    if(incs)
        target_include_directories(LAPACK::LAPACK INTERFACE ${incs})
    endif()
    set(LAPACK_FOUND              TRUE    PARENT_SCOPE)
    set(LAPACK_LIBRARIES          "${libs}"  PARENT_SCOPE)
    set(LAPACK_INCLUDE_DIRS       "${incs}"  PARENT_SCOPE)
    set(VGRE_LAPACK_DESCRIPTION   "${desc}"  PARENT_SCOPE)
endfunction()

# =============================================================================
# macOS — Accelerate framework (ships with every macOS install)
# =============================================================================
if(APPLE AND NOT LAPACK_FOUND)
    find_library(_accel Accelerate)
    if(_accel)
        _vgre_lapack_commit("${_accel}" "" "macOS Accelerate.framework (vecLib BLAS + LAPACK)")
        message(STATUS "LAPACK: found Accelerate.framework (${_accel})")
    else()
        # Homebrew OpenBLAS fallback
        find_program(_brew brew)
        if(_brew)
            execute_process(COMMAND "${_brew}" --prefix openblas
                OUTPUT_VARIABLE _brew_openblas OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        endif()
        if(_brew_openblas AND EXISTS "${_brew_openblas}")
            find_library(_obl openblas HINTS "${_brew_openblas}/lib")
            if(_obl)
                _vgre_lapack_commit("${_obl}" "${_brew_openblas}/include"
                    "Homebrew OpenBLAS at ${_brew_openblas}")
                message(STATUS "LAPACK: found Homebrew OpenBLAS (${_obl})")
            endif()
        endif()
    endif()
endif()

# =============================================================================
# Windows — Intel MKL → vcpkg OpenBLAS → MinGW reference BLAS
# =============================================================================
if(WIN32 AND NOT LAPACK_FOUND)

    # ── Strategy 1: Intel MKL ─────────────────────────────────────────────
    set(_mkl_root "")
    if(DEFINED ENV{MKLROOT} AND EXISTS "$ENV{MKLROOT}")
        set(_mkl_root "$ENV{MKLROOT}")
    elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/mkl/latest")
        set(_mkl_root "$ENV{ONEAPI_ROOT}/mkl/latest")
    else()
        foreach(_yr IN ITEMS 2024 2023 2022 2021)
            foreach(_pfx IN ITEMS
                "$ENV{ProgramFiles}/Intel/oneAPI/mkl/${_yr}.0"
                "$ENV{ProgramFiles\(x86\)}/Intel/oneAPI/mkl/${_yr}.0")
                if(EXISTS "${_pfx}")
                    set(_mkl_root "${_pfx}")
                    break()
                endif()
            endforeach()
            if(_mkl_root)
                break()
            endif()
        endforeach()
    endif()

    if(_mkl_root)
        set(_mkl_libdir "${_mkl_root}/lib/intel64")
        find_library(_mkl_core   NAMES mkl_core           HINTS "${_mkl_libdir}" NO_DEFAULT_PATH)
        find_library(_mkl_lp64   NAMES mkl_intel_lp64     HINTS "${_mkl_libdir}" NO_DEFAULT_PATH)
        find_library(_mkl_thread NAMES mkl_intel_thread mkl_sequential
                                  HINTS "${_mkl_libdir}" NO_DEFAULT_PATH)
        if(_mkl_core AND _mkl_lp64 AND _mkl_thread)
            _vgre_lapack_commit(
                "${_mkl_lp64};${_mkl_thread};${_mkl_core}"
                "${_mkl_root}/include"
                "Intel oneAPI MKL LP64 at ${_mkl_root}")
            message(STATUS "LAPACK: found Intel MKL at ${_mkl_root}")
        endif()
    endif()

    # ── Strategy 2: vcpkg OpenBLAS ────────────────────────────────────────
    if(NOT LAPACK_FOUND)
        find_package(OpenBLAS CONFIG QUIET)
        if(OpenBLAS_FOUND AND TARGET OpenBLAS::OpenBLAS)
            if(NOT TARGET LAPACK::LAPACK)
                add_library(LAPACK::LAPACK INTERFACE IMPORTED GLOBAL)
            endif()
            target_link_libraries(LAPACK::LAPACK INTERFACE OpenBLAS::OpenBLAS)
            set(LAPACK_FOUND TRUE)
            set(VGRE_LAPACK_DESCRIPTION "OpenBLAS (vcpkg) ${OpenBLAS_VERSION}")
            message(STATUS "LAPACK: found vcpkg OpenBLAS ${OpenBLAS_VERSION}")
        endif()
    endif()

    # ── Strategy 3: MinGW / MSYS2 reference BLAS/LAPACK ──────────────────
    if(NOT LAPACK_FOUND)
        find_library(_ref_lapack NAMES lapack PATHS /mingw64/lib /usr/lib NO_DEFAULT_PATH)
        find_library(_ref_blas   NAMES blas   PATHS /mingw64/lib /usr/lib NO_DEFAULT_PATH)
        if(_ref_lapack AND _ref_blas)
            _vgre_lapack_commit("${_ref_lapack};${_ref_blas}" ""
                "MinGW reference LAPACK/BLAS at ${_ref_lapack}")
            message(STATUS "LAPACK: found MinGW reference LAPACK (${_ref_lapack})")
        endif()
    endif()
endif()

# =============================================================================
# Linux / Generic UNIX
# =============================================================================
if(UNIX AND NOT APPLE AND NOT LAPACK_FOUND)

    # ── Strategy 1: pkg-config ────────────────────────────────────────────
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(_pc_lapack QUIET lapack)
        pkg_check_modules(_pc_blas   QUIET blas)
        if(_pc_lapack_FOUND AND _pc_blas_FOUND)
            # Resolve -lfoo flags to absolute paths where possible
            set(_resolved "")
            foreach(_l IN LISTS _pc_lapack_LIBRARIES _pc_blas_LIBRARIES)
                find_library(_fl "${_l}"
                    HINTS ${_pc_lapack_LIBRARY_DIRS} ${_pc_blas_LIBRARY_DIRS})
                if(_fl)
                    list(APPEND _resolved "${_fl}")
                    unset(_fl CACHE)
                else()
                    list(APPEND _resolved "-l${_l}")
                endif()
            endforeach()
            _vgre_lapack_commit("${_resolved}" ""
                "pkg-config LAPACK ${_pc_lapack_VERSION} / BLAS ${_pc_blas_VERSION}")
            message(STATUS "LAPACK: found via pkg-config")
        endif()
    endif()

    # ── Strategy 2: OpenBLAS (single lib ships both BLAS and LAPACK) ─────
    if(NOT LAPACK_FOUND)
        find_library(_openblas NAMES openblas
            PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64
            PATH_SUFFIXES openblas)
        if(_openblas)
            _vgre_lapack_commit("${_openblas}" "" "OpenBLAS at ${_openblas}")
            message(STATUS "LAPACK: found OpenBLAS (${_openblas})")
        endif()
    endif()

    # ── Strategy 3: Debian/Ubuntu multiarch arch-specific directories ─────
    # Ubuntu stores versioned shared libraries in subdirs of
    # /usr/lib/<triple>/{lapack,blas}/ which CMake's default search skips.
    if(NOT LAPACK_FOUND)
        # Determine the Debian multiarch triple
        execute_process(
            COMMAND dpkg-architecture -q DEB_HOST_MULTIARCH
            OUTPUT_VARIABLE _dma OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(NOT _dma)
            # Derive from CMake's processor detection as a fallback
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
                set(_dma "x86_64-linux-gnu")
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                set(_dma "aarch64-linux-gnu")
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armhf")
                set(_dma "arm-linux-gnueabihf")
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
                set(_dma "riscv64-linux-gnu")
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|powerpc64le")
                set(_dma "powerpc64le-linux-gnu")
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
                set(_dma "s390x-linux-gnu")
            endif()
        endif()

        set(_arch_dirs "")
        if(_dma)
            list(APPEND _arch_dirs
                "/usr/lib/${_dma}/lapack"
                "/usr/lib/${_dma}/blas"
                "/usr/lib/${_dma}")
        endif()
        list(APPEND _arch_dirs
            /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64)

        # Debian/Ubuntu ships ONLY versioned symlinks (liblapack.so.3, no .so).
        # CMake's find_library doesn't try versioned suffixes automatically;
        # passing the full filename as a NAME bypasses the lib-prefix/suffix
        # normalisation and matches exactly against the filesystem.
        find_library(_sys_lapack
            NAMES
                lapack              # unversioned (present on Fedora/RHEL/openSUSE)
                liblapack.so.3      # Debian/Ubuntu Alternatives symlink
                liblapack.so.3.12.0 # exact versioned file
                liblapack.so.4      # future-proof
            HINTS ${_arch_dirs}
            NO_DEFAULT_PATH)

        find_library(_sys_blas
            NAMES
                blas
                libblas.so.3
                libblas.so.3.12.0
                libblas.so.4
            HINTS ${_arch_dirs}
            NO_DEFAULT_PATH)

        # Final fallback: include CMake's default search paths
        if(NOT _sys_lapack)
            find_library(_sys_lapack
                NAMES lapack liblapack.so.3 liblapack.so.4)
        endif()
        if(NOT _sys_blas)
            find_library(_sys_blas
                NAMES blas libblas.so.3 libblas.so.4)
        endif()

        if(_sys_lapack AND _sys_blas)
            _vgre_lapack_commit("${_sys_lapack};${_sys_blas}" ""
                "system LAPACK=${_sys_lapack}  BLAS=${_sys_blas}")
            message(STATUS "LAPACK: found system LAPACK (${_sys_lapack})")
            message(STATUS "LAPACK: found system BLAS  (${_sys_blas})")
        endif()
    endif()

    # ── Strategy 4: Intel MKL on Linux ────────────────────────────────────
    if(NOT LAPACK_FOUND)
        set(_mkl_root "")
        if(DEFINED ENV{MKLROOT} AND EXISTS "$ENV{MKLROOT}")
            set(_mkl_root "$ENV{MKLROOT}")
        elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/mkl/latest")
            set(_mkl_root "$ENV{ONEAPI_ROOT}/mkl/latest")
        else()
            foreach(_p IN ITEMS
                    /opt/intel/oneapi/mkl/latest
                    /opt/intel/mkl
                    /usr/local/intel/mkl)
                if(EXISTS "${_p}")
                    set(_mkl_root "${_p}")
                    break()
                endif()
            endforeach()
        endif()

        if(_mkl_root)
            foreach(_ldir IN ITEMS
                    "${_mkl_root}/lib/intel64"
                    "${_mkl_root}/lib")
                find_library(_mkl_core   NAMES mkl_core          HINTS "${_ldir}" NO_DEFAULT_PATH)
                find_library(_mkl_lp64   NAMES mkl_intel_lp64    HINTS "${_ldir}" NO_DEFAULT_PATH)
                find_library(_mkl_thread NAMES mkl_intel_thread mkl_sequential
                              HINTS "${_ldir}" NO_DEFAULT_PATH)
                if(_mkl_core AND _mkl_lp64 AND _mkl_thread)
                    break()
                endif()
            endforeach()
            if(_mkl_core AND _mkl_lp64 AND _mkl_thread)
                # Link against iomp5 if available alongside the MKL installation
                find_library(_mkl_iomp NAMES iomp5
                    HINTS
                        "${_mkl_root}/../compiler/latest/linux/lib"
                        "${_mkl_root}/../compiler/lib/intel64")
                set(_mkl_all "${_mkl_lp64};${_mkl_thread};${_mkl_core}")
                if(_mkl_iomp)
                    list(APPEND _mkl_all "${_mkl_iomp}")
                endif()
                _vgre_lapack_commit("${_mkl_all}" "${_mkl_root}/include"
                    "Intel oneAPI MKL LP64 at ${_mkl_root}")
                message(STATUS "LAPACK: found Intel MKL at ${_mkl_root}")
            endif()
        endif()
    endif()
endif()

# =============================================================================
# Standard result validation
# =============================================================================
if(NOT LAPACK_FOUND)
    # Emit a human-readable hint before the standard error fires.
    message(STATUS "LAPACK hint -- install one of:")
    message(STATUS "  Linux  : sudo apt install liblapack-dev libblas-dev")
    message(STATUS "           or: sudo apt install libopenblas-dev")
    message(STATUS "  macOS  : brew install openblas  (Accelerate tried first)")
    message(STATUS "  Windows: Intel oneAPI Base Toolkit or vcpkg openblas:x64-windows")
endif()
# DEFAULT_MSG emits the standard "Could not find LAPACK" message; the hints
# above give the user actionable install instructions without fighting CMake's
# argument parser with multi-line REASON_FAILURE_MESSAGE strings.
find_package_handle_standard_args(LAPACK DEFAULT_MSG LAPACK_FOUND)
