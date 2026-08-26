// Benchmark-only bridge to catid/leopard's translation-unit-local FF16 walkers.
// The pinned upstream implementation is included unchanged so these wrappers
// call the same static functions used by ReedSolomonEncode/Decode.
#include "LeopardCommon.cpp"
#include "LeopardFF16.cpp"

namespace leopard { namespace ff16 {

extern "C" bool butterfly_fft_leopard_init()
{
    InitializeCPUArch();
    return Initialize();
}

extern "C" const char* butterfly_fft_leopard_backend()
{
#if defined(LEO_TRY_AVX2)
    if (CpuHasAVX2)
        return "avx2";
#endif
    if (CpuHasSSSE3)
        return "ssse3";
    return "scalar";
}

extern "C" void butterfly_fft_leopard_forward(void** rows, unsigned points, uint64_t row_len)
{
    FFT_DIT(row_len, rows, points, points, FFTSkew - 1);
}

extern "C" void butterfly_fft_leopard_inverse(void** rows, unsigned points, uint64_t row_len)
{
    IFFT_DIT_Decoder(row_len, points, rows, points, FFTSkew - 1);
}

extern "C" void butterfly_fft_leopard_derivative(void** rows, unsigned points, uint64_t row_len)
{
    for (unsigned i = 1; i < points; ++i)
    {
        const unsigned width = ((i ^ (i - 1)) + 1) >> 1;
        if (width < 8)
            VectorXOR(row_len, width, rows + i - width, rows + i);
        else
            VectorXOR_Threads(row_len, width, rows + i - width, rows + i);
    }
}

extern "C" void butterfly_fft_leopard_derivative_exact(
    void** output,
    void* const* coefficients,
    unsigned points,
    uint64_t row_len)
{
    for (unsigned i = 0; i < points; ++i)
        memcpy(output[i], coefficients[i], row_len);

    butterfly_fft_leopard_derivative(output, points, row_len);

    // Leopard's in-place sweep computes c + D(c). Remove c to produce the
    // exact out-of-place derivative contract benchmarked by butterfly-fft.
    VectorXOR(row_len, points, output, const_cast<void**>(coefficients));
}

}} // namespace leopard::ff16
