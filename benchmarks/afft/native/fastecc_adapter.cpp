// Benchmark-only bridge to Bulat-Ziganshin/FastECC's NTT implementation.
// The pinned upstream implementation is included unchanged so these
// wrappers drive the same MFA_NTT used by its Reed-Solomon encoder.
#include <algorithm>
#include <math.h>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "GF(p).cpp"
#include "ntt.cpp"

namespace fastecc {

// FastECC's default field: GF(0xFFF00001), whose multiplicative group
// contains 2^20-th roots of unity.
constexpr uint32_t FIELD = 0xFFF00001;

struct FastEccNtt {
    uint32_t* data;
    uint32_t** blocks;
    size_t points;
    size_t lanes;
};

extern "C" FastEccNtt* butterfly_fft_fastecc_new(size_t points, size_t lanes)
{
    auto* state = new FastEccNtt;
    state->points = points;
    state->lanes = lanes;
    state->data = new uint32_t[points * lanes];
    state->blocks = new uint32_t*[points];
    for (size_t i = 0; i < points; i++)
        state->blocks[i] = state->data + i * lanes;
    return state;
}

extern "C" void butterfly_fft_fastecc_fill(FastEccNtt* state, uint32_t seed)
{
    uint32_t x = seed;
    uint32_t* end = state->data + state->points * state->lanes;
    for (uint32_t* p = state->data; p < end; p++) {
        x = x * 1664525u + 1013904223u;
        *p = x & 0xFFF00000u;  // every value stays below the field modulus
    }
}

extern "C" void butterfly_fft_fastecc_forward(FastEccNtt* state)
{
    MFA_NTT<uint32_t, FIELD>(state->blocks, state->points, state->lanes, false);
}

extern "C" void butterfly_fft_fastecc_inverse(FastEccNtt* state)
{
    MFA_NTT<uint32_t, FIELD>(state->blocks, state->points, state->lanes, true);
}

extern "C" const uint32_t* butterfly_fft_fastecc_data(const FastEccNtt* state)
{
    return state->data;
}

extern "C" void butterfly_fft_fastecc_free(FastEccNtt* state)
{
    delete[] state->data;
    delete[] state->blocks;
    delete state;
}

} // namespace fastecc
