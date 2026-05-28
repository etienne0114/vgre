/**
 * Test: Cross-module PTX linking (cuLinkCreate / cuLinkAddData / cuLinkComplete)
 *
 * Verifies that cuLinkComplete correctly deduplicates .extern .func declarations
 * when two PTX modules cross-reference each other's symbols: module A forward-
 * declares helper() as extern and calls it; module B defines helper().
 * After linking, the merged PTX must contain exactly one definition of helper
 * and zero remaining .extern .func declarations for it.
 */

#include <iostream>
#include <cstring>
#include <string>

// ── Minimal CUDA Driver API types ────────────────────────────────────────────
extern "C" {

typedef int    CUresult;
typedef void*  CUlinkState;

static constexpr CUresult CUDA_SUCCESS              = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE  = 1;

// Options type (unused, but part of the signature)
enum CUjit_option { CU_JIT_MAX_REGISTERS = 0 };

CUresult cuLinkCreate(unsigned int numOptions, int *options, void **optionValues,
                      CUlinkState *stateOut);
CUresult cuLinkAddData(CUlinkState state, int type, void *data, size_t size,
                       const char *name, unsigned int numOptions,
                       int *options, void **optionValues);
CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
CUresult cuLinkDestroy(CUlinkState state);

} // extern "C"

static int g_pass = 0, g_total = 0;

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; std::cout << "PASS [" << g_total << "] " << name << "\n"; }
    else    { std::cerr << "FAIL [" << g_total << "] " << name << "\n"; }
}

// ── PTX snippets ──────────────────────────────────────────────────────────────
//
// Module A: declares helper as .extern .func, defines entry_a that calls it.
static const char kPtxA[] =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    "\n"
    ".extern .func helper (.param .u32 p);\n"
    "\n"
    ".visible .entry entry_a (.param .u32 val)\n"
    "{\n"
    "    .reg .u32 %r0;\n"
    "    ld.param.u32 %r0, [val];\n"
    "    { .param .u32 pval;\n"
    "      st.param.u32 [pval], %r0;\n"
    "      call.uni helper, (pval);\n"
    "    }\n"
    "    ret;\n"
    "}\n";

// Module B: defines helper(), forward-declares entry_a as .extern .func.
static const char kPtxB[] =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    "\n"
    ".extern .func entry_a (.param .u32 p);\n"
    "\n"
    ".visible .func helper (.param .u32 p)\n"
    "{\n"
    "    .reg .u32 %r1;\n"
    "    ld.param.u32 %r1, [p];\n"
    "    ret;\n"
    "}\n";

// ── Sub-tests ─────────────────────────────────────────────────────────────────

static void test_lifecycle() {
    CUlinkState state = nullptr;
    bool ok = (cuLinkCreate(0, nullptr, nullptr, &state) == CUDA_SUCCESS);
    ok &= (state != nullptr);
    ok &= (cuLinkDestroy(state) == CUDA_SUCCESS);
    ok &= (cuLinkCreate(0, nullptr, nullptr, nullptr) == CUDA_ERROR_INVALID_VALUE);
    check("cuLinkCreate / cuLinkDestroy lifecycle", ok);
}

static void test_null_rejection() {
    CUlinkState state = nullptr;
    cuLinkCreate(0, nullptr, nullptr, &state);
    // null data
    bool ok = (cuLinkAddData(state, 0, nullptr, 10, "a", 0, nullptr, nullptr)
               != CUDA_SUCCESS);
    // zero size
    char dummy[] = "dummy";
    ok &= (cuLinkAddData(state, 0, dummy, 0, "b", 0, nullptr, nullptr)
           != CUDA_SUCCESS);
    cuLinkDestroy(state);
    check("cuLinkAddData rejects null/zero inputs", ok);
}

static void test_cross_module_dedup() {
    CUlinkState state = nullptr;
    cuLinkCreate(0, nullptr, nullptr, &state);

    // Add module A first, then module B
    char ptxA[sizeof(kPtxA)];
    char ptxB[sizeof(kPtxB)];
    memcpy(ptxA, kPtxA, sizeof(kPtxA));
    memcpy(ptxB, kPtxB, sizeof(kPtxB));

    bool ok = (cuLinkAddData(state, 0, ptxA, strlen(ptxA), "moduleA",
                             0, nullptr, nullptr) == CUDA_SUCCESS);
    ok &= (cuLinkAddData(state, 0, ptxB, strlen(ptxB), "moduleB",
                         0, nullptr, nullptr) == CUDA_SUCCESS);

    void   *cubin  = nullptr;
    size_t  cusize = 0;
    ok &= (cuLinkComplete(state, &cubin, &cusize) == CUDA_SUCCESS);
    ok &= (cubin != nullptr && cusize > 0);

    if (ok) {
        std::string merged(static_cast<const char*>(cubin), cusize);

        // helper must appear as a definition (.func helper), not only as extern
        bool hasHelperDef = (merged.find(".func helper") != std::string::npos);

        // The .extern .func helper declaration should have been stripped
        // (still may appear as .func entry_a extern since entry_a IS defined)
        // Specifically: ".extern .func helper" should not exist after merge
        bool externHelperGone =
            (merged.find(".extern .func helper") == std::string::npos);

        ok = hasHelperDef && externHelperGone;
        if (!ok) {
            std::cerr << "  hasHelperDef=" << hasHelperDef
                      << " externHelperGone=" << externHelperGone << "\n";
        }
    }

    cuLinkDestroy(state);
    check("cuLinkComplete deduplicates .extern .func for cross-module symbols", ok);
}

static void test_merged_contains_both_entries() {
    CUlinkState state = nullptr;
    cuLinkCreate(0, nullptr, nullptr, &state);

    char ptxA[sizeof(kPtxA)];
    char ptxB[sizeof(kPtxB)];
    memcpy(ptxA, kPtxA, sizeof(kPtxA));
    memcpy(ptxB, kPtxB, sizeof(kPtxB));

    cuLinkAddData(state, 0, ptxA, strlen(ptxA), "A", 0, nullptr, nullptr);
    cuLinkAddData(state, 0, ptxB, strlen(ptxB), "B", 0, nullptr, nullptr);

    void *cubin = nullptr; size_t sz = 0;
    cuLinkComplete(state, &cubin, &sz);

    bool ok = false;
    if (cubin && sz > 0) {
        std::string merged(static_cast<const char*>(cubin), sz);
        // Both entry_a and helper must be present in merged output
        ok = (merged.find("entry_a") != std::string::npos) &&
             (merged.find("helper")  != std::string::npos);
    }

    cuLinkDestroy(state);
    check("merged PTX contains both entry_a and helper symbols", ok);
}

static void test_empty_link() {
    CUlinkState state = nullptr;
    cuLinkCreate(0, nullptr, nullptr, &state);
    void *cubin = nullptr; size_t sz = 999;
    bool ok = (cuLinkComplete(state, &cubin, &sz) == CUDA_SUCCESS);
    ok &= (sz == 0);
    cuLinkDestroy(state);
    check("cuLinkComplete on empty state returns empty output", ok);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "=== Test: Cross-module PTX Linking ===\n";

    test_lifecycle();
    test_null_rejection();
    test_cross_module_dedup();
    test_merged_contains_both_entries();
    test_empty_link();

    std::cout << "\nResult: " << g_pass << "/" << g_total << " passed\n";
    return (g_pass == g_total) ? 0 : 1;
}
