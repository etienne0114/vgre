#ifndef VGRE_COMMON_PLATFORM_H
#define VGRE_COMMON_PLATFORM_H

#if defined(_WIN32)
#  define VGRE_PUBLIC_API   __declspec(dllexport)
#else
#  define VGRE_PUBLIC_API   __attribute__((visibility("default")))
#endif
// C++11 thread_local is supported by MSVC 2015+, GCC 4.8+, and Apple Clang 5+.
// The GCC extension __thread is not valid C++ and warns on Apple Clang in
// C++ mode; thread_local is strictly correct.
#define VGRE_THREAD_LOCAL thread_local
#if defined(_MSC_VER)
#  include <intrin.h>
#  define VGRE_PREFETCH(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#else
#  define VGRE_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)
#endif

#endif // VGRE_COMMON_PLATFORM_H
