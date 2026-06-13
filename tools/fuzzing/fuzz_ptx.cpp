// libFuzzer entry for the ptx target.  Build with -DVGRE_ENABLE_FUZZERS=ON
// (clang -fsanitize=fuzzer,address) then run: ./vgre_fuzz_ptx [corpus_dir]
#include "fuzz_targets.h"
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return vgre_fuzz_ptx(data, size);
}
