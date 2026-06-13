# §  Fuzzing harness over VGRE's untrusted-input parsers (docs §1.4).
#
# The fuzz-target cores are compiled into a small static library shared by the
# deterministic CI smoke test (tests/fuzzing) and the optional libFuzzer
# binaries.  Build the real coverage-guided fuzzers with:
#     cmake -DVGRE_ENABLE_FUZZERS=ON ..    (requires clang)
#     ./vgre_fuzz_json   ./vgre_fuzz_jwt   ./vgre_fuzz_ptx
# For deep coverage, also configure the whole library with
#     -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link"

add_library(vgre_fuzz_targets STATIC fuzz_targets.cpp)
target_link_libraries(vgre_fuzz_targets PUBLIC vgre)
target_include_directories(vgre_fuzz_targets PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(vgre_fuzz_targets PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(VGRE_ENABLE_FUZZERS "Build libFuzzer fuzz binaries (requires clang)" OFF)
if(VGRE_ENABLE_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(WARNING "VGRE_ENABLE_FUZZERS requires Clang/libFuzzer; skipping fuzz binaries.")
    else()
        foreach(t json jwt ptx)
            add_executable(vgre_fuzz_${t} fuzz_${t}.cpp)
            target_link_libraries(vgre_fuzz_${t} PRIVATE vgre_fuzz_targets)
            target_compile_options(vgre_fuzz_${t} PRIVATE -fsanitize=fuzzer,address)
            target_link_options(vgre_fuzz_${t} PRIVATE -fsanitize=fuzzer,address)
        endforeach()
        # Instrument the target cores for coverage-guided mutation.
        target_compile_options(vgre_fuzz_targets PRIVATE -fsanitize=fuzzer-no-link)
    endif()
endif()
