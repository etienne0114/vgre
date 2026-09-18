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

# Windows has no LD_LIBRARY_PATH/DYLD_LIBRARY_PATH equivalent: its loader
# resolves DLLs via the launching exe's own directory, then PATH. CTest runs
# each test with its CWD set to the CMake subdirectory that defined it (e.g.
# build/tests), which holds neither vgre.dll nor vgre_nn.dll — so without a
# PATH entry the loader silently falls through to any same-named DLL already
# on the user's PATH (e.g. a stale prior install in %LOCALAPPDATA%\VGRE),
# producing STATUS_ENTRYPOINT_NOT_FOUND / access-violation crashes instead of
# a clean "DLL not found" error, and worse — a "passing" test could quietly be
# exercising old code. Call once per directory, after all add_test() calls in
# it, to prepend the real build output dirs ahead of everything already on PATH.
function(vgre_fixup_test_dll_search_path)
    if(NOT WIN32)
        return()
    endif()
    get_property(_vgre_tests DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY TESTS)
    foreach(_t IN LISTS _vgre_tests)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
            set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:${CMAKE_BINARY_DIR}"
                "PATH=path_list_prepend:${CMAKE_BINARY_DIR}/src/xla")
        else()
            # Fallback for CMake < 3.22 (no ENVIRONMENT_MODIFICATION): escape
            # the PATH value's semicolons so CTest's ";"-separated ENVIRONMENT
            # list treats the whole thing as one NAME=VALUE entry.
            set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT
                "PATH=${CMAKE_BINARY_DIR}\\;${CMAKE_BINARY_DIR}/src/xla\\;$ENV{PATH}")
        endif()
    endforeach()
endfunction()
