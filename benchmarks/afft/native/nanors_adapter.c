#include <stdint.h>

#include "deps/obl/oblas16.h"
#include "deps/obl/oblas16_afft.h"

static struct oblas16_impl field_impl;
static struct oblas16_afft_impl afft_impl;

void butterfly_fft_nanors_init(void)
{
#if defined(OBLAS_ARCH_X86) && !defined(_MSC_VER)
    __builtin_cpu_init();
#endif
    oblas16_afft_init();
    oblas16_get_impl(&field_impl);
    oblas16_afft_get_impl(&afft_impl);
}

const char *butterfly_fft_nanors_backend(void)
{
#if defined(OBLAS_ARCH_X86)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("gfni"))
        return "avx512-gfni";
    if (__builtin_cpu_supports("avx512f"))
        return "avx512";
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("gfni"))
        return "avx2-gfni";
    if (__builtin_cpu_supports("avx2"))
        return "avx2";
    if (__builtin_cpu_supports("ssse3") && __builtin_cpu_supports("gfni"))
        return "ssse3-gfni";
    if (__builtin_cpu_supports("ssse3"))
        return "ssse3";
    return "scalar";
#elif defined(OBLAS_ARCH_ARM)
    return "neon";
#elif defined(OBLAS_ARCH_RISCV) && defined(__riscv_vector)
    return "rvv";
#else
    return "scalar";
#endif
}

void butterfly_fft_nanors_forward(uint8_t *rows, unsigned log_points, unsigned row_len)
{
    oblas16_afft_fft(
        (uint16_t *)rows,
        (int)log_points,
        (int)(row_len / 2),
        0,
        0,
        &field_impl,
        &afft_impl);
}

void butterfly_fft_nanors_inverse(uint8_t *rows, unsigned log_points, unsigned row_len)
{
    oblas16_afft_ifft(
        (uint16_t *)rows,
        (int)log_points,
        (int)(row_len / 2),
        1 << log_points,
        0,
        &field_impl,
        &afft_impl);
}
