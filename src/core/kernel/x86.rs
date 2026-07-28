//! `x86`/`x86_64` butterfly kernels for GF(2^8) and GF(2^16).
//!
//! Three tiers, selected at runtime by [`crate::core::kernel::backend`]:
//!
//! - **GFNI** (with AVX2): `GF2P8MULB` multiplies 32 GF(2^8) lanes in the
//!   AES polynomial directly. GF(2^16) uses the interleaved-component trick:
//!   multiplying `[a, b]` by `c0 + c1·u` is
//!   `[c0·a + Δ·c1·b, c1·a + (c0+c1)·b]`, computed as one multiply of the
//!   source and one of its adjacent-byte swap, with no planar conversion.
//! - **AVX2**: split-nibble `PSHUFB` tables, `c·x = lo[x&0xF] ^ hi[x>>4]`.
//!   GF(2^16) uses four base-field tables with the same swap trick.
//! - **SSSE3**: the AVX2 scheme over 16-byte lanes.
//!
//! All kernels are unaligned (`loadu`/`storeu`) and delegate sub-vector
//! tails to the portable scalar implementation.

#![allow(clippy::incompatible_msrv)]
// All vector accesses go through `loadu`/`storeu`, so pointer-to-vector
// casts never rely on alignment.
#![allow(clippy::cast_ptr_alignment)]

#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

use fff::{Gf8, Gf16, gf8, gf16};

use super::scalar;

/// Adjacent-byte swap mask for the interleaved GF(2^16) component trick.
const SWAP_ADJACENT: [u8; 32] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13,
    12, 15, 14,
];

/// Split-nibble multiply table for one GF(2^8) coefficient, both AVX2 lanes.
struct ScaleTable {
    low: [u8; 32],
    high: [u8; 32],
}

fn scale_table(coefficient: gf8::Elem) -> ScaleTable {
    let mut low = [0; 32];
    let mut high = [0; 32];
    for nibble in 0..16u8 {
        low[nibble as usize] = gf8::Elem(nibble).mul(coefficient).0;
        high[nibble as usize] = gf8::Elem(nibble << 4).mul(coefficient).0;
    }
    for nibble in 0..16 {
        low[16 + nibble] = low[nibble];
        high[16 + nibble] = high[nibble];
    }
    ScaleTable { low, high }
}

/// Broadcast pair of GF(2^8) coefficients for the GFNI GF(2^16) trick:
/// `same = [c0, c0+c1]`, `cross = [Δ·c1, c1]` in each 16-bit lane.
#[inline]
fn factor_words(coefficient: gf16::Elem) -> (i16, i16) {
    let (c0, c1) = coefficient.components();
    let delta_c1 = gf16::DELTA.mul(c1);
    let same = i16::from_le_bytes([c0.0, c0.add(c1).0]);
    let cross = i16::from_le_bytes([delta_c1.0, c1.0]);
    (same, cross)
}

/// The four base-field nibble tables for the AVX2/SSSE3 GF(2^16) trick:
/// `c0`, `c0+c1`, `Δ·c1`, `c1`.
fn factor_tables(coefficient: gf16::Elem) -> [ScaleTable; 4] {
    let (c0, c1) = coefficient.components();
    [
        scale_table(c0),
        scale_table(c0.add(c1)),
        scale_table(gf16::DELTA.mul(c1)),
        scale_table(c1),
    ]
}

#[target_feature(enable = "avx2")]
unsafe fn multiply_avx2(value: __m256i, table: &ScaleTable) -> __m256i {
    let low_nibbles = _mm256_and_si256(value, _mm256_set1_epi8(0x0f));
    let high_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(value), _mm256_set1_epi8(0x0f));
    // SAFETY: table arrays are 32 bytes; unaligned loads are allowed.
    let (low_table, high_table) = unsafe {
        (
            _mm256_loadu_si256(table.low.as_ptr().cast::<__m256i>()),
            _mm256_loadu_si256(table.high.as_ptr().cast::<__m256i>()),
        )
    };
    _mm256_xor_si256(
        _mm256_shuffle_epi8(low_table, low_nibbles),
        _mm256_shuffle_epi8(high_table, high_nibbles),
    )
}

#[target_feature(enable = "ssse3")]
unsafe fn multiply_ssse3(value: __m128i, table: &ScaleTable) -> __m128i {
    let low_nibbles = _mm_and_si128(value, _mm_set1_epi8(0x0f));
    let high_nibbles = _mm_and_si128(_mm_srli_epi16::<4>(value), _mm_set1_epi8(0x0f));
    // SAFETY: table arrays are at least 16 bytes; unaligned loads are allowed.
    let (low_table, high_table) = unsafe {
        (
            _mm_loadu_si128(table.low.as_ptr().cast::<__m128i>()),
            _mm_loadu_si128(table.high.as_ptr().cast::<__m128i>()),
        )
    };
    _mm_xor_si128(
        _mm_shuffle_epi8(low_table, low_nibbles),
        _mm_shuffle_epi8(high_table, high_nibbles),
    )
}

/// `source * coefficient` for interleaved GF(2^16) elements, GFNI form.
#[target_feature(enable = "avx2,gfni")]
unsafe fn scaled_vector_gfni(source: __m256i, same: i16, cross: i16) -> __m256i {
    // SAFETY: the mask constant is 32 bytes; unaligned load is allowed.
    let swap_mask = unsafe { _mm256_loadu_si256(SWAP_ADJACENT.as_ptr().cast::<__m256i>()) };
    let swapped = _mm256_shuffle_epi8(source, swap_mask);
    let direct = _mm256_gf2p8mul_epi8(source, _mm256_set1_epi16(same));
    let crossed = _mm256_gf2p8mul_epi8(swapped, _mm256_set1_epi16(cross));
    _mm256_xor_si256(direct, crossed)
}

/// `source * coefficient` for interleaved GF(2^16) elements, AVX2 form.
#[target_feature(enable = "avx2")]
unsafe fn scaled_vector_avx2(source: __m256i, tables: &[ScaleTable; 4]) -> __m256i {
    // SAFETY: the mask constant is 32 bytes; unaligned load is allowed.
    let swap_mask = unsafe { _mm256_loadu_si256(SWAP_ADJACENT.as_ptr().cast::<__m256i>()) };
    let swapped = _mm256_shuffle_epi8(source, swap_mask);
    let even_mask = _mm256_set1_epi16(0x00ff);
    // SAFETY: AVX2 is enabled by the enclosing target_feature.
    let (direct_even, direct_odd, cross_even, cross_odd) = unsafe {
        (
            multiply_avx2(source, &tables[0]),
            multiply_avx2(source, &tables[1]),
            multiply_avx2(swapped, &tables[2]),
            multiply_avx2(swapped, &tables[3]),
        )
    };
    let direct = _mm256_xor_si256(
        _mm256_and_si256(direct_even, even_mask),
        _mm256_andnot_si256(even_mask, direct_odd),
    );
    let crossed = _mm256_xor_si256(
        _mm256_and_si256(cross_even, even_mask),
        _mm256_andnot_si256(even_mask, cross_odd),
    );
    _mm256_xor_si256(direct, crossed)
}

/// `source * coefficient` for interleaved GF(2^16) elements, SSSE3 form.
#[target_feature(enable = "ssse3")]
unsafe fn scaled_vector_ssse3(source: __m128i, tables: &[ScaleTable; 4]) -> __m128i {
    // SAFETY: the mask constant is at least 16 bytes; unaligned load allowed.
    let swap_mask = unsafe { _mm_loadu_si128(SWAP_ADJACENT.as_ptr().cast::<__m128i>()) };
    let swapped = _mm_shuffle_epi8(source, swap_mask);
    let even_mask = _mm_set1_epi16(0x00ff);
    // SAFETY: SSSE3 is enabled by the enclosing target_feature.
    let (direct_even, direct_odd, cross_even, cross_odd) = unsafe {
        (
            multiply_ssse3(source, &tables[0]),
            multiply_ssse3(source, &tables[1]),
            multiply_ssse3(swapped, &tables[2]),
            multiply_ssse3(swapped, &tables[3]),
        )
    };
    let direct = _mm_xor_si128(
        _mm_and_si128(direct_even, even_mask),
        _mm_andnot_si128(even_mask, direct_odd),
    );
    let crossed = _mm_xor_si128(
        _mm_and_si128(cross_even, even_mask),
        _mm_andnot_si128(even_mask, cross_odd),
    );
    _mm_xor_si128(direct, crossed)
}

// ---------------------------------------------------------------------------
// GF(2^8) fused butterflies
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,gfni")]
pub(super) unsafe fn gf8_fused_forward_gfni(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let coeff = _mm256_set1_epi8(coefficient.0.cast_signed());
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let scaled = _mm256_gf2p8mul_epi8(h, coeff);
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_forward::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "avx2,gfni")]
pub(super) unsafe fn gf8_fused_inverse_gfni(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let coeff = _mm256_set1_epi8(coefficient.0.cast_signed());
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let new_high = _mm256_xor_si256(h, l);
        let scaled = _mm256_gf2p8mul_epi8(new_high, coeff);
        let new_low = _mm256_xor_si256(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_inverse::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn gf8_fused_forward_avx2(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let table = scale_table(coefficient);
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        // SAFETY: AVX2 is enabled by the enclosing target_feature.
        let scaled = unsafe { multiply_avx2(h, &table) };
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_forward::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn gf8_fused_inverse_avx2(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let table = scale_table(coefficient);
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let new_high = _mm256_xor_si256(h, l);
        // SAFETY: AVX2 is enabled by the enclosing target_feature.
        let scaled = unsafe { multiply_avx2(new_high, &table) };
        let new_low = _mm256_xor_si256(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_inverse::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gf8_fused_forward_ssse3(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let table = scale_table(coefficient);
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm_loadu_si128(low.as_ptr().add(offset).cast::<__m128i>()),
                _mm_loadu_si128(high.as_ptr().add(offset).cast::<__m128i>()),
            )
        };
        // SAFETY: SSSE3 is enabled by the enclosing target_feature.
        let scaled = unsafe { multiply_ssse3(h, &table) };
        let new_low = _mm_xor_si128(l, scaled);
        let new_high = _mm_xor_si128(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm_storeu_si128(low.as_mut_ptr().add(offset).cast::<__m128i>(), new_low);
            _mm_storeu_si128(high.as_mut_ptr().add(offset).cast::<__m128i>(), new_high);
        }
        offset += 16;
    }
    scalar::fused_forward::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gf8_fused_inverse_ssse3(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let table = scale_table(coefficient);
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm_loadu_si128(low.as_ptr().add(offset).cast::<__m128i>()),
                _mm_loadu_si128(high.as_ptr().add(offset).cast::<__m128i>()),
            )
        };
        let new_high = _mm_xor_si128(h, l);
        // SAFETY: SSSE3 is enabled by the enclosing target_feature.
        let scaled = unsafe { multiply_ssse3(new_high, &table) };
        let new_low = _mm_xor_si128(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm_storeu_si128(low.as_mut_ptr().add(offset).cast::<__m128i>(), new_low);
            _mm_storeu_si128(high.as_mut_ptr().add(offset).cast::<__m128i>(), new_high);
        }
        offset += 16;
    }
    scalar::fused_inverse::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

// ---------------------------------------------------------------------------
// GF(2^16) fused butterflies
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,gfni")]
pub(super) unsafe fn gf16_fused_forward_gfni(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let (same, cross) = factor_words(coefficient);
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        // SAFETY: AVX2+GFNI are enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_gfni(h, same, cross) };
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_forward::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "avx2,gfni")]
pub(super) unsafe fn gf16_fused_inverse_gfni(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let (same, cross) = factor_words(coefficient);
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let new_high = _mm256_xor_si256(h, l);
        // SAFETY: AVX2+GFNI are enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_gfni(new_high, same, cross) };
        let new_low = _mm256_xor_si256(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_inverse::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn gf16_fused_forward_avx2(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        // SAFETY: AVX2 is enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_avx2(h, &tables) };
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_forward::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn gf16_fused_inverse_avx2(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let vector_len = low.len() / 32 * 32;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 32 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm256_loadu_si256(low.as_ptr().add(offset).cast::<__m256i>()),
                _mm256_loadu_si256(high.as_ptr().add(offset).cast::<__m256i>()),
            )
        };
        let new_high = _mm256_xor_si256(h, l);
        // SAFETY: AVX2 is enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_avx2(new_high, &tables) };
        let new_low = _mm256_xor_si256(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm256_storeu_si256(low.as_mut_ptr().add(offset).cast::<__m256i>(), new_low);
            _mm256_storeu_si256(high.as_mut_ptr().add(offset).cast::<__m256i>(), new_high);
        }
        offset += 32;
    }
    scalar::fused_inverse::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gf16_fused_forward_ssse3(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm_loadu_si128(low.as_ptr().add(offset).cast::<__m128i>()),
                _mm_loadu_si128(high.as_ptr().add(offset).cast::<__m128i>()),
            )
        };
        // SAFETY: SSSE3 is enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_ssse3(h, &tables) };
        let new_low = _mm_xor_si128(l, scaled);
        let new_high = _mm_xor_si128(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm_storeu_si128(low.as_mut_ptr().add(offset).cast::<__m128i>(), new_low);
            _mm_storeu_si128(high.as_mut_ptr().add(offset).cast::<__m128i>(), new_high);
        }
        offset += 16;
    }
    scalar::fused_forward::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn gf16_fused_inverse_ssse3(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                _mm_loadu_si128(low.as_ptr().add(offset).cast::<__m128i>()),
                _mm_loadu_si128(high.as_ptr().add(offset).cast::<__m128i>()),
            )
        };
        let new_high = _mm_xor_si128(h, l);
        // SAFETY: SSSE3 is enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_ssse3(new_high, &tables) };
        let new_low = _mm_xor_si128(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            _mm_storeu_si128(low.as_mut_ptr().add(offset).cast::<__m128i>(), new_low);
            _mm_storeu_si128(high.as_mut_ptr().add(offset).cast::<__m128i>(), new_high);
        }
        offset += 16;
    }
    scalar::fused_inverse::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}
