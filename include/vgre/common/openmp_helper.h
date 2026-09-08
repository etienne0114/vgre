#pragma once

// Single-point OpenMP include guard — use this instead of duplicating
// #ifdef _OPENMP / #include <omp.h> / #endif in every source file.

#ifdef _OPENMP
#include <omp.h>
#else
// Built without OpenMP (no -fopenmp → _OPENMP undefined): provide serial
// fallbacks so the few omp_* runtime calls compile and behave single-threaded.
// #pragma omp directives are ignored by the compiler in this mode. This is what
// lets VGRE build with just a C++ compiler (VGRE_ENABLE_OPENMP=OFF).
static inline int  omp_get_thread_num()  { return 0; }
static inline int  omp_get_num_threads() { return 1; }
static inline int  omp_get_max_threads() { return 1; }
static inline void omp_set_num_threads(int) {}
#endif

// MSVC ships OpenMP 2.0 which does not implement the collapse() clause
// (added in OpenMP 3.0).  Use OMP_COLLAPSE(n) inside #pragma omp directives
// so the clause is emitted on GCC/Clang but silently dropped on MSVC.
#ifdef _MSC_VER
#  define OMP_COLLAPSE(n)
#else
#  define OMP_COLLAPSE(n) collapse(n)
#endif
