// Track 10 — cuSPARSE sparse-matrix format conversions.
//
// Verifies that the (previously stubbed) view/conversion path produces correct
// results for the SAME matrix in the target format:
//   COO → CSR (counting sort), CSR ↔ CSC (real transpose), CSR → COO.
//
// Reference matrix A (3×3):
//     [10  0 20]
//     [ 0 30  0]
//     [40  0 50]

#include "sparse_view.h"

#include <cstdio>
#include <vector>

using namespace vgre::sparse;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

template <typename V> static bool eq(const void *p, const V &expect) {
    const auto *a = static_cast<const typename V::value_type *>(p);
    for (size_t i = 0; i < expect.size(); ++i)
        if (a[i] != expect[i]) return false;
    return true;
}

int main() {
    printf("=== Sparse Matrix Conversions (Track 10) ===\n");

    // ── COO form of A (row-major nnz order) ──────────────────────────────────
    std::vector<int32_t> cooRow = {0, 0, 1, 2, 2};
    std::vector<int32_t> cooCol = {0, 2, 1, 0, 2};
    std::vector<float>   cooVal = {10, 20, 30, 40, 50};
    SparseMatrixView<float, int32_t> coo(cooRow.data(), cooCol.data(),
                                         cooVal.data(), 3, 3, 5, 1,
                                         SparseFormat::COO);

    // ── COO → CSR ────────────────────────────────────────────────────────────
    auto csr = coo.toView(SparseFormat::CSR);
    check("COO→CSR produced a matrix", csr.values != nullptr && !csr.isView);
    check("CSR rowPtr = [0,2,3,5]",
          eq(csr.rowOffsets, std::vector<int32_t>{0, 2, 3, 5}));
    check("CSR colInd = [0,2,1,0,2]",
          eq(csr.colIndices, std::vector<int32_t>{0, 2, 1, 0, 2}));
    check("CSR val = [10,20,30,40,50]",
          eq(csr.values, std::vector<float>{10, 20, 30, 40, 50}));

    // ── CSR → CSC (real transpose of the SAME matrix) ────────────────────────
    auto csc = csr.toView(SparseFormat::CSC);
    check("CSR→CSC produced a matrix", csc.values != nullptr && !csc.isView);
    // CSC of A: colPtr[cols+1], rowIdx[nnz], val[nnz].
    check("CSC colPtr = [0,2,3,5]",
          eq(csc.rowOffsets, std::vector<int32_t>{0, 2, 3, 5}));
    check("CSC rowIdx = [0,2,1,0,2]",
          eq(csc.colIndices, std::vector<int32_t>{0, 2, 1, 0, 2}));
    check("CSC val = [10,40,30,20,50] (column-major order)",
          eq(csc.values, std::vector<float>{10, 40, 30, 20, 50}));

    // ── CSC → CSR round-trips back to the original CSR ───────────────────────
    auto back = csc.toView(SparseFormat::CSR);
    check("CSC→CSR round-trip rowPtr matches",
          eq(back.rowOffsets, std::vector<int32_t>{0, 2, 3, 5}));
    check("CSC→CSR round-trip colInd matches",
          eq(back.colIndices, std::vector<int32_t>{0, 2, 1, 0, 2}));
    check("CSC→CSR round-trip val matches",
          eq(back.values, std::vector<float>{10, 20, 30, 40, 50}));

    // ── CSR → COO expands row pointers ───────────────────────────────────────
    auto coo2 = csr.toView(SparseFormat::COO);
    check("CSR→COO rowIdx = [0,0,1,2,2]",
          eq(coo2.rowOffsets, std::vector<int32_t>{0, 0, 1, 2, 2}));
    check("CSR→COO colIdx = [0,2,1,0,2]",
          eq(coo2.colIndices, std::vector<int32_t>{0, 2, 1, 0, 2}));

    // ── canConvertZeroCopy is now honest (CSR→CSC is NOT zero-copy) ──────────
    check("canConvertZeroCopy(CSR→CSC) == false (needs transpose)",
          !csr.canConvertZeroCopy(SparseFormat::CSC));
    check("canConvertZeroCopy(CSR→BSR,bs=1) == true",
          csr.canConvertZeroCopy(SparseFormat::BSR));

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
