# VGREPlatform.cmake — dynamic cross-platform dependency discovery (no hardcoded prefixes)

# vgre_brew_prefix(FORMULA OUT_VAR)
# On macOS with Homebrew, sets OUT_VAR to `brew --prefix FORMULA` when the keg exists.
macro(vgre_brew_prefix FORMULA OUT_VAR)
    set(${OUT_VAR} "")
    if(APPLE)
        if(NOT _VGRE_BREW_EXE)
            find_program(_VGRE_BREW_EXE brew)
        endif()
        if(_VGRE_BREW_EXE)
            execute_process(
                COMMAND "${_VGRE_BREW_EXE}" --prefix ${FORMULA}
                OUTPUT_VARIABLE _vgre_brew_out
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(_vgre_brew_out AND EXISTS "${_vgre_brew_out}")
                set(${OUT_VAR} "${_vgre_brew_out}")
            endif()
        endif()
    endif()
endmacro()

# Pre-project hook: prefer Homebrew llvm@18 on macOS when the user did not pick a compiler.
# VGRE links LLVM 18 and JIT-compiles kernels with the same major toolchain family.
if(APPLE AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{VGRE_SKIP_MACOS_TOOLCHAIN})
    find_program(_VGRE_BREW_EARLY brew)
    if(_VGRE_BREW_EARLY)
        execute_process(
            COMMAND "${_VGRE_BREW_EARLY}" --prefix llvm@18
            OUTPUT_VARIABLE _VGRE_LLVM18_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_VGRE_LLVM18_PREFIX AND EXISTS "${_VGRE_LLVM18_PREFIX}/bin/clang++")
            set(CMAKE_C_COMPILER "${_VGRE_LLVM18_PREFIX}/bin/clang" CACHE FILEPATH "VGRE macOS toolchain" FORCE)
            set(CMAKE_CXX_COMPILER "${_VGRE_LLVM18_PREFIX}/bin/clang++" CACHE FILEPATH "VGRE macOS toolchain" FORCE)
        endif()
    endif()
endif()

function(vgre_apply_macos_link_flags)
    if(NOT APPLE)
        return()
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        execute_process(
            COMMAND sw_vers -productVersion
            OUTPUT_VARIABLE _VGRE_OSX_VER
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_VGRE_OSX_VER)
            string(REGEX REPLACE "\\.[0-9]+$" "" _VGRE_OSX_MAJOR_MINOR "${_VGRE_OSX_VER}")
            set(CMAKE_OSX_DEPLOYMENT_TARGET "${_VGRE_OSX_MAJOR_MINOR}"
                CACHE STRING "macOS deployment target (host version)" FORCE)
            message(STATUS "macOS: deployment target ${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()
    endif()
    vgre_brew_prefix(llvm@18 _llvm18)
    if(NOT _llvm18)
        return()
    endif()
    set(_libcxx "${_llvm18}/lib/c++")
    if(EXISTS "${_libcxx}")
        add_link_options(
            "LINKER:-L${_libcxx}"
            "LINKER:-rpath,${_libcxx}")
        message(STATUS "macOS: Homebrew llvm@18 libc++ rpath at ${_libcxx}")
    endif()
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE _VGRE_OSX_SDK
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_VGRE_OSX_SDK AND EXISTS "${_VGRE_OSX_SDK}")
        set(CMAKE_OSX_SYSROOT "${_VGRE_OSX_SDK}" CACHE PATH "macOS SDK for Homebrew clang" FORCE)
        message(STATUS "macOS: SDK ${_VGRE_OSX_SDK}")
    endif()
endfunction()

# CTest: both vars so Linux and macOS loaders find freshly-built libs in the build tree.
function(vgre_test_library_path_env OUT_VAR)
    set(${OUT_VAR}
        "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR};DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
        PARENT_SCOPE)
endfunction()
