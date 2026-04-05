#ifndef VGRE_COMMON_PLATFORM_H
#define VGRE_COMMON_PLATFORM_H

#if defined(_WIN32)
#define VGRE_PUBLIC_API __declspec(dllexport)
#define VGRE_THREAD_LOCAL thread_local
#else
#define VGRE_PUBLIC_API __attribute__((visibility("default")))
#define VGRE_THREAD_LOCAL __thread
#endif

#endif // VGRE_COMMON_PLATFORM_H
