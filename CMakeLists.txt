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

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# ── Compiler flags ──────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
    # SIMD — enable at the highest level the host supports
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE4)
    if(COMPILER_SUPPORTS_AVX2)
        add_compile_options(-mavx2 -mfma)
        add_compile_definitions(VGRE_HAS_AVX2)
    elseif(COMPILER_SUPPORTS_SSE4)
        add_compile_options(-msse4.1)
        add_compile_definitions(VGRE_HAS_SSE4)
    endif()
endif()

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(LIBOMP_PATH "")
if(UNIX AND NOT APPLE)
    find_library(LIBOMP_PATH NAMES omp PATHS /usr/lib/llvm-18/lib /usr/lib/x86_64-linux-gnu)
    if(LIBOMP_PATH)
        set(LIBOMP_PATH "${LIBOMP_PATH}" CACHE FILEPATH "Path to LLVM OpenMP library")
        message(STATUS "Found LLVM OpenMP: ${LIBOMP_PATH}")
        link_libraries("-Wl,--no-as-needed" "${LIBOMP_PATH}" "-Wl,--as-needed")
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
    set_target_properties(vgre PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
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
    endif()
    
elseif(APPLE)
    # macOS: Security framework for Keychain
    find_library(SECURITY_FRAMEWORK Security)
    if(SECURITY_FRAMEWORK)
        target_link_libraries(vgre PUBLIC ${SECURITY_FRAMEWORK})
        message(STATUS "Found Security framework: ${SECURITY_FRAMEWORK}")
    endif()
elseif(WIN32)
    # Windows: Credential Manager (part of Advapi32)
    target_link_libraries(vgre PUBLIC Advapi32)
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
    src/api/cudart_shim.cpp
    src/api/cuda_driver_shim.cpp
    src/api/cuda_interceptor.cpp
    src/api/vgre_c_api.cpp
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
)
target_compile_definitions(vgre_cudart PRIVATE VGRE_CUDA_INTERCEPTOR_NO_EXPORTS)
target_link_libraries(vgre_cudart PRIVATE
    OpenMP::OpenMP_CXX
    ${OpenMP_CXX_LIBRARIES}
    Threads::Threads
    ${llvm_libs}
)
if(WIN32)
    target_link_libraries(vgre_cudart PRIVATE Advapi32)
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
# ── Install rules ───────────────────────────────────────────────────────────
install(TARGETS vgre vgre_cudart vgre-worker
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/vgre DESTINATION include)
