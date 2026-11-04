// math_functions.h from CUDA 13.2+ defines _NV_RSQRT_SPECIFIER.
// Clang does not include it, so we need to define it ourselves.
#if defined(__GNUC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 42)
#define _NV_RSQRT_SPECIFIER noexcept(true)
#endif
#endif

#ifndef _NV_RSQRT_SPECIFIER
#define _NV_RSQRT_SPECIFIER
#endif
