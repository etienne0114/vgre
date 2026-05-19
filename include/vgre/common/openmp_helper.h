#pragma once

// Single-point OpenMP include guard — use this instead of duplicating
// #ifdef _OPENMP / #include <omp.h> / #endif in every source file.

#ifdef _OPENMP
#include <omp.h>
#endif

// MSVC ships OpenMP 2.0 which does not implement the collapse() clause
// (added in OpenMP 3.0).  Use OMP_COLLAPSE(n) inside #pragma omp directives
// so the clause is emitted on GCC/Clang but silently dropped on MSVC.
#ifdef _MSC_VER
#  define OMP_COLLAPSE(n)
#else
#  define OMP_COLLAPSE(n) collapse(n)
#endif
