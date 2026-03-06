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
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs Core ExecutionEngine OrcJIT Support Target native)

# ── Global include directory ────────────────────────────────────────────────
include_directories(${PROJECT_SOURCE_DIR}/include)

# ── Sub-projects ────────────────────────────────────────────────────────────
add_subdirectory(src/core)
add_subdirectory(src/compiler)
add_subdirectory(src/api)
add_subdirectory(src/runtime)
add_subdirectory(src/advanced)

# ── Shared library ──────────────────────────────────────────────────────────
add_library(vgre SHARED
    $<TARGET_OBJECTS:vgre_core>
    $<TARGET_OBJECTS:vgre_compiler>
    $<TARGET_OBJECTS:vgre_api>
    $<TARGET_OBJECTS:vgre_runtime>
    $<TARGET_OBJECTS:vgre_advanced>
)
target_link_libraries(vgre PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
    ${llvm_libs}
)
set_target_properties(vgre PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# ── CUDART Shim (LD_PRELOAD wrapper) ────────────────────────────────────────
add_library(vgre_cudart SHARED
    src/api/cudart_shim.cpp
)
target_link_libraries(vgre_cudart PRIVATE vgre)
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
        add_custom_target(vgre_dashboard ALL
            COMMAND ${FLUTTER_EXECUTABLE} build linux --release
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/vgre_dashboard
            COMMENT "Building VGRE Flutter Dashboard (Release)"
            VERBATIM
        )
        add_dependencies(vgre_dashboard vgre)
    else()
        message(WARNING "flutter executable not found. Dashboard will not be built automatically.")
    endif()
endif()
# ── Install rules ───────────────────────────────────────────────────────────
install(TARGETS vgre LIBRARY DESTINATION lib)
install(DIRECTORY include/vgre DESTINATION include)
