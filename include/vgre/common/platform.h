#ifndef VGRE_COMMON_PLATFORM_H
#define VGRE_COMMON_PLATFORM_H

#if defined(_WIN32)
// On Windows the shared libraries export their public symbols via CMake's
// WINDOWS_EXPORT_ALL_SYMBOLS (an auto-generated .def covering every non-static
// symbol), so no per-function __declspec(dllexport) is needed here. Emitting one
// is in fact harmful under clang-cl: unlike cl.exe, clang-cl makes it a hard
// ERROR to add dllexport on a definition when an earlier plain forward
// declaration was already seen ("cannot add 'dllexport' attribute") — and the
// vgre_jit_* / vgre_cdp_* runtime symbols are forward-declared plainly in a dozen
// device-lib headers and .cpp translation units. Leaving this empty removes that
// entire error class while WINDOWS_EXPORT_ALL_SYMBOLS keeps every symbol
// importable by the tests, tools and the JIT.
#  define VGRE_PUBLIC_API
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
