//! AArch64 NEON butterfly kernels for GF(2^8) and GF(2^16).
//!
//! NEON is baseline on AArch64, so there is a single tier, selected at
//! runtime by [`crate::core::kernel::backend`] like every other backend:
//!
//! - **GF(2^8)**: split-nibble `TBL` tables, `c·x = lo[x&0xF] ^ hi[x>>4]`.
//! - **GF(2^16)**: bit-serial base-field multiply with the
//!   interleaved-component trick: multiplying `[a, b]` by `c0 + c1·u` is
//!   `[c0·a + Δ·c1·b, c1·a + (c0+c1)·b]`, computed as one multiply of the
//!   source and one of its adjacent-byte swap (`VREV16`), with no planar
//!   conversion.
//!
//! All kernels are unaligned (`vld1q`/`vst1q`) and delegate sub-vector
//! tails to the portable scalar implementation.

#![allow(clippy::incompatible_msrv)]

use ::core::arch::aarch64::*;

use fgf::{Gf8, Gf16, gf8, gf16};

use super::scalar;

/// Split-nibble multiply table for one GF(2^8) coefficient.
struct ScaleTable {
    low: [u8; 16],
    high: [u8; 16],
}

fn scale_table(coefficient: gf8::Elem) -> ScaleTable {
    let mut low = [0; 16];
    let mut high = [0; 16];
    for nibble in 0..16u8 {
        low[nibble as usize] = gf8::Elem(nibble).mul(coefficient).0;
        high[nibble as usize] = gf8::Elem(nibble << 4).mul(coefficient).0;
    }
    ScaleTable { low, high }
}

/// Broadcast pair of GF(2^8) coefficients for the GF(2^16) trick:
/// `same = [c0, c0+c1]`, `cross = [Δ·c1, c1]` in each 16-bit lane.
#[inline]
fn factor_words(coefficient: gf16::Elem) -> (u16, u16) {
    let (c0, c1) = coefficient.components();
    let delta_c1 = gf16::DELTA.mul(c1);
    let same = u16::from_le_bytes([c0.0, c0.add(c1).0]);
    let cross = u16::from_le_bytes([delta_c1.0, c1.0]);
    (same, cross)
}

/// `value * coefficient` per GF(2^8) lane via the split-nibble tables.
#[target_feature(enable = "neon")]
unsafe fn multiply_neon(
    value: uint8x16_t,
    low_table: uint8x16_t,
    high_table: uint8x16_t,
) -> uint8x16_t {
    let nibble_mask = vdupq_n_u8(0x0f);
    let low_nibbles = vandq_u8(value, nibble_mask);
    let high_nibbles = vandq_u8(vshrq_n_u8::<4>(value), nibble_mask);
    veorq_u8(
        vqtbl1q_u8(low_table, low_nibbles),
        vqtbl1q_u8(high_table, high_nibbles),
    )
}

/// Bit-serial GF(2^8) multiply `value * factor` per lane: eight rounds of
/// mask/add/shift/reduce with the AES polynomial (`0x1B`).
#[target_feature(enable = "neon")]
unsafe fn multiply_base_vector(mut value: uint8x16_t, mut factor: uint8x16_t) -> uint8x16_t {
    let mut product = vdupq_n_u8(0);
    let one = vdupq_n_u8(1);
    let high_threshold = vdupq_n_u8(0x7f);
    let reduction = vdupq_n_u8(0x1b);
    for _ in 0..8 {
        let active = vceqq_u8(vandq_u8(factor, one), one);
        product = veorq_u8(product, vandq_u8(value, active));
        let high = vcgtq_u8(value, high_threshold);
        value = veorq_u8(vshlq_n_u8::<1>(value), vandq_u8(high, reduction));
        factor = vshrq_n_u8::<1>(factor);
    }
    product
}

/// `source * coefficient` for interleaved GF(2^16) elements.
#[target_feature(enable = "neon")]
unsafe fn scaled_vector_neon(
    source: uint8x16_t,
    same: uint8x16_t,
    cross: uint8x16_t,
) -> uint8x16_t {
    // SAFETY: NEON is enabled by the enclosing target_feature.
    let (direct, crossed) = unsafe {
        (
            multiply_base_vector(source, same),
            multiply_base_vector(vrev16q_u8(source), cross),
        )
    };
    veorq_u8(direct, crossed)
}

// ---------------------------------------------------------------------------
// GF(2^8) fused butterflies
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
pub(super) unsafe fn gf8_fused_forward_neon(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let table = scale_table(coefficient);
    // SAFETY: the table arrays are 16 bytes; unaligned loads are allowed.
    let (low_table, high_table) =
        unsafe { (vld1q_u8(table.low.as_ptr()), vld1q_u8(table.high.as_ptr())) };
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                vld1q_u8(low.as_ptr().add(offset)),
                vld1q_u8(high.as_ptr().add(offset)),
            )
        };
        // SAFETY: NEON is enabled by the enclosing target_feature.
        let scaled = unsafe { multiply_neon(h, low_table, high_table) };
        let new_low = veorq_u8(l, scaled);
        let new_high = veorq_u8(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_u8(low.as_mut_ptr().add(offset), new_low);
            vst1q_u8(high.as_mut_ptr().add(offset), new_high);
        }
        offset += 16;
    }
    scalar::fused_forward::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gf8_fused_inverse_neon(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8::Elem,
) {
    let table = scale_table(coefficient);
    // SAFETY: the table arrays are 16 bytes; unaligned loads are allowed.
    let (low_table, high_table) =
        unsafe { (vld1q_u8(table.low.as_ptr()), vld1q_u8(table.high.as_ptr())) };
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                vld1q_u8(low.as_ptr().add(offset)),
                vld1q_u8(high.as_ptr().add(offset)),
            )
        };
        let new_high = veorq_u8(h, l);
        // SAFETY: NEON is enabled by the enclosing target_feature.
        let scaled = unsafe { multiply_neon(new_high, low_table, high_table) };
        let new_low = veorq_u8(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_u8(low.as_mut_ptr().add(offset), new_low);
            vst1q_u8(high.as_mut_ptr().add(offset), new_high);
        }
        offset += 16;
    }
    scalar::fused_inverse::<Gf8>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

// ---------------------------------------------------------------------------
// GF(2^16) fused butterflies
// ---------------------------------------------------------------------------

#[target_feature(enable = "neon")]
pub(super) unsafe fn gf16_fused_forward_neon(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let (same_word, cross_word) = factor_words(coefficient);
    let same = vreinterpretq_u8_u16(vdupq_n_u16(same_word));
    let cross = vreinterpretq_u8_u16(vdupq_n_u16(cross_word));
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                vld1q_u8(low.as_ptr().add(offset)),
                vld1q_u8(high.as_ptr().add(offset)),
            )
        };
        // SAFETY: NEON is enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_neon(h, same, cross) };
        let new_low = veorq_u8(l, scaled);
        let new_high = veorq_u8(h, new_low);
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_u8(low.as_mut_ptr().add(offset), new_low);
            vst1q_u8(high.as_mut_ptr().add(offset), new_high);
        }
        offset += 16;
    }
    scalar::fused_forward::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gf16_fused_inverse_neon(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let (same_word, cross_word) = factor_words(coefficient);
    let same = vreinterpretq_u8_u16(vdupq_n_u16(same_word));
    let cross = vreinterpretq_u8_u16(vdupq_n_u16(cross_word));
    let vector_len = low.len() / 16 * 16;
    let mut offset = 0;
    while offset < vector_len {
        // SAFETY: `offset + 16 <= low.len() == high.len()`; unaligned allowed.
        let (l, h) = unsafe {
            (
                vld1q_u8(low.as_ptr().add(offset)),
                vld1q_u8(high.as_ptr().add(offset)),
            )
        };
        let new_high = veorq_u8(h, l);
        // SAFETY: NEON is enabled by the enclosing target_feature.
        let scaled = unsafe { scaled_vector_neon(new_high, same, cross) };
        let new_low = veorq_u8(l, scaled);
        // SAFETY: same bounds as the loads above.
        unsafe {
            vst1q_u8(low.as_mut_ptr().add(offset), new_low);
            vst1q_u8(high.as_mut_ptr().add(offset), new_high);
        }
        offset += 16;
    }
    scalar::fused_inverse::<Gf16>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}
