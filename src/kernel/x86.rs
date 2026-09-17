//! `x86`/`x86_64` butterfly kernels for GF(2^8) and GF(2^16).
//!
//! Three tiers, selected at runtime by [`crate::kernel::backend`]:
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
//! Every kernel is a safe [`archmage`] capability-token function taking the
//! exact token its instructions require (`X64V3Token` for AVX2, `X64V2Token`
//! for SSSE3), walking the halves with reference-based loads and stores over
//! chunk arrays; the lane builders are `#[rite]` helpers. Sub-lane tails go
//! to the portable scalar kernels on an element boundary.
//!
//! [`archmage`]: https://docs.rs/archmage

#![allow(clippy::incompatible_msrv)]

#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

use fgf::{Gf8B, Gf16, gf8b, gf16};

use super::scalar;

/// Adjacent-byte swap mask for the interleaved GF(2^16) component trick.
///
/// Sixteen bytes; the 32-byte lanes broadcast it to both halves.
const SWAP_ADJACENT: [u8; 16] = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14];

/// Split-nibble multiply table for one GF(2^8) coefficient.
///
/// Sixteen entries per half; AVX2 lanes use each half broadcast to both
/// 128-bit lanes of the 32-byte registers.
struct ScaleTable {
    low: [u8; 16],
    high: [u8; 16],
}

fn scale_table(coefficient: gf8b::Elem) -> ScaleTable {
    let mut low = [0; 16];
    let mut high = [0; 16];
    for nibble in 0..16u8 {
        low[nibble as usize] = gf8b::Elem::from_raw(nibble).mul(coefficient).to_raw();
        high[nibble as usize] = gf8b::Elem::from_raw(nibble << 4).mul(coefficient).to_raw();
    }
    ScaleTable { low, high }
}

/// Broadcast pair of GF(2^8) coefficients for the GFNI GF(2^16) trick:
/// `same = [c0, c0+c1]`, `cross = [Δ·c1, c1]` in each 16-bit lane.
#[inline]
fn factor_words(coefficient: gf16::Elem) -> (i16, i16) {
    let (c0, c1) = coefficient.to_components();
    let delta_c1 = gf16::DELTA.mul(c1);
    let same = i16::from_le_bytes([c0.to_raw(), c0.add(c1).to_raw()]);
    let cross = i16::from_le_bytes([delta_c1.to_raw(), c1.to_raw()]);
    (same, cross)
}

/// The four base-field nibble tables for the AVX2/SSSE3 GF(2^16) trick:
/// `c0`, `c0+c1`, `Δ·c1`, `c1`.
fn factor_tables(coefficient: gf16::Elem) -> [ScaleTable; 4] {
    let (c0, c1) = coefficient.to_components();
    [
        scale_table(c0),
        scale_table(c0.add(c1)),
        scale_table(gf16::DELTA.mul(c1)),
        scale_table(c1),
    ]
}

/// `value * coefficient` per GF(2^8) lane via the split-nibble tables, AVX2
/// form.
#[archmage::rite(v3, import_intrinsics)]
fn multiply_avx2(value: __m256i, table: &ScaleTable) -> __m256i {
    let low_nibbles = _mm256_and_si256(value, _mm256_set1_epi8(0x0f));
    let high_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(value), _mm256_set1_epi8(0x0f));
    let (low_table, high_table) = (
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.low)),
        _mm256_broadcastsi128_si256(_mm_loadu_si128(&table.high)),
    );
    _mm256_xor_si256(
        _mm256_shuffle_epi8(low_table, low_nibbles),
        _mm256_shuffle_epi8(high_table, high_nibbles),
    )
}

/// `value * coefficient` per GF(2^8) lane via the split-nibble tables, SSSE3
/// form.
#[archmage::rite(v2, import_intrinsics)]
fn multiply_ssse3(value: __m128i, table: &ScaleTable) -> __m128i {
    let low_nibbles = _mm_and_si128(value, _mm_set1_epi8(0x0f));
    let high_nibbles = _mm_and_si128(_mm_srli_epi16::<4>(value), _mm_set1_epi8(0x0f));
    let (low_table, high_table) = (_mm_loadu_si128(&table.low), _mm_loadu_si128(&table.high));
    _mm_xor_si128(
        _mm_shuffle_epi8(low_table, low_nibbles),
        _mm_shuffle_epi8(high_table, high_nibbles),
    )
}

/// `source * coefficient` for interleaved GF(2^16) elements, AVX2 form.
#[archmage::rite(v3, import_intrinsics)]
fn scaled_vector_avx2(source: __m256i, tables: &[ScaleTable; 4]) -> __m256i {
    let swap_mask = _mm256_broadcastsi128_si256(_mm_loadu_si128(&SWAP_ADJACENT));
    let swapped = _mm256_shuffle_epi8(source, swap_mask);
    let even_mask = _mm256_set1_epi16(0x00ff);
    let (direct_even, direct_odd, cross_even, cross_odd) = (
        multiply_avx2(source, &tables[0]),
        multiply_avx2(source, &tables[1]),
        multiply_avx2(swapped, &tables[2]),
        multiply_avx2(swapped, &tables[3]),
    );
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
#[archmage::rite(v2, import_intrinsics)]
fn scaled_vector_ssse3(source: __m128i, tables: &[ScaleTable; 4]) -> __m128i {
    let swap_mask = _mm_loadu_si128(&SWAP_ADJACENT);
    let swapped = _mm_shuffle_epi8(source, swap_mask);
    let even_mask = _mm_set1_epi16(0x00ff);
    let (direct_even, direct_odd, cross_even, cross_odd) = (
        multiply_ssse3(source, &tables[0]),
        multiply_ssse3(source, &tables[1]),
        multiply_ssse3(swapped, &tables[2]),
        multiply_ssse3(swapped, &tables[3]),
    );
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

/// Fused forward butterfly over 32-byte AVX2 lanes, GF(2^8) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf8_fused_forward_avx2(
    _token: archmage::X64V3Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8b::Elem,
) {
    let table = scale_table(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let scaled = multiply_avx2(h, &table);
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_forward::<Gf8B>(low_tail, high_tail, coefficient);
}

/// Fused inverse butterfly over 32-byte AVX2 lanes, GF(2^8) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf8_fused_inverse_avx2(
    _token: archmage::X64V3Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8b::Elem,
) {
    let table = scale_table(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let new_high = _mm256_xor_si256(h, l);
        let scaled = multiply_avx2(new_high, &table);
        let new_low = _mm256_xor_si256(l, scaled);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_inverse::<Gf8B>(low_tail, high_tail, coefficient);
}

/// Fused forward butterfly over 16-byte SSSE3 lanes, GF(2^8) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf8_fused_forward_ssse3(
    _token: archmage::X64V2Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8b::Elem,
) {
    let table = scale_table(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<16>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<16>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm_loadu_si128(&*low_lane);
        let h = _mm_loadu_si128(&*high_lane);
        let scaled = multiply_ssse3(h, &table);
        let new_low = _mm_xor_si128(l, scaled);
        let new_high = _mm_xor_si128(h, new_low);
        _mm_storeu_si128(low_lane, new_low);
        _mm_storeu_si128(high_lane, new_high);
    }
    scalar::fused_forward::<Gf8B>(low_tail, high_tail, coefficient);
}

/// Fused inverse butterfly over 16-byte SSSE3 lanes, GF(2^8) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf8_fused_inverse_ssse3(
    _token: archmage::X64V2Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8b::Elem,
) {
    let table = scale_table(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<16>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<16>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm_loadu_si128(&*low_lane);
        let h = _mm_loadu_si128(&*high_lane);
        let new_high = _mm_xor_si128(h, l);
        let scaled = multiply_ssse3(new_high, &table);
        let new_low = _mm_xor_si128(l, scaled);
        _mm_storeu_si128(low_lane, new_low);
        _mm_storeu_si128(high_lane, new_high);
    }
    scalar::fused_inverse::<Gf8B>(low_tail, high_tail, coefficient);
}

// ---------------------------------------------------------------------------
// GF(2^16) fused butterflies
// ---------------------------------------------------------------------------

/// Fused forward butterfly over 32-byte AVX2 lanes, GF(2^16) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf16_fused_forward_avx2(
    _token: archmage::X64V3Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let scaled = scaled_vector_avx2(h, &tables);
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_forward::<Gf16>(low_tail, high_tail, coefficient);
}

/// Fused inverse butterfly over 32-byte AVX2 lanes, GF(2^16) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf16_fused_inverse_avx2(
    _token: archmage::X64V3Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let new_high = _mm256_xor_si256(h, l);
        let scaled = scaled_vector_avx2(new_high, &tables);
        let new_low = _mm256_xor_si256(l, scaled);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_inverse::<Gf16>(low_tail, high_tail, coefficient);
}

/// Fused forward butterfly over 16-byte SSSE3 lanes, GF(2^16) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf16_fused_forward_ssse3(
    _token: archmage::X64V2Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<16>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<16>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm_loadu_si128(&*low_lane);
        let h = _mm_loadu_si128(&*high_lane);
        let scaled = scaled_vector_ssse3(h, &tables);
        let new_low = _mm_xor_si128(l, scaled);
        let new_high = _mm_xor_si128(h, new_low);
        _mm_storeu_si128(low_lane, new_low);
        _mm_storeu_si128(high_lane, new_high);
    }
    scalar::fused_forward::<Gf16>(low_tail, high_tail, coefficient);
}

/// Fused inverse butterfly over 16-byte SSSE3 lanes, GF(2^16) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub(super) fn gf16_fused_inverse_ssse3(
    _token: archmage::X64V2Token,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let tables = factor_tables(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<16>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<16>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm_loadu_si128(&*low_lane);
        let h = _mm_loadu_si128(&*high_lane);
        let new_high = _mm_xor_si128(h, l);
        let scaled = scaled_vector_ssse3(new_high, &tables);
        let new_low = _mm_xor_si128(l, scaled);
        _mm_storeu_si128(low_lane, new_low);
        _mm_storeu_si128(high_lane, new_high);
    }
    scalar::fused_inverse::<Gf16>(low_tail, high_tail, coefficient);
}

mod gfni;
pub(super) use gfni::{
    gf8_fused_forward_gfni, gf8_fused_inverse_gfni, gf16_fused_forward_gfni,
    gf16_fused_inverse_gfni,
};
