//! GFNI butterfly kernels for GF(2^8) and GF(2^16).
//!
//! `GF2P8MULB` multiplies 32 GF(2^8) lanes in the AES polynomial directly;
//! GF(2^16) uses the interleaved-component trick shared with the parent
//! module. Only GFNI-capable hosts execute these kernels; the coverage gate
//! excludes this file because the hosted CI fleet mixes Xeon generations
//! and a GFNI-less runner cannot execute a line of it — the direct-kernel
//! differential tests cover it wherever the host can summon the tier.

#![allow(clippy::cast_ptr_alignment)]

#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

use fgf::{Gf8B, Gf16, gf8b, gf16};

use super::scalar;
use super::{SWAP_ADJACENT, factor_words};

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
#[target_feature(enable = "avx2,gfni")]
pub unsafe fn gf8_fused_forward_gfni(low: &mut [u8], high: &mut [u8], coefficient: gf8b::Elem) {
    let coeff = _mm256_set1_epi8(coefficient.to_raw().cast_signed());
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
    scalar::fused_forward::<Gf8B>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}
#[target_feature(enable = "avx2,gfni")]
pub unsafe fn gf8_fused_inverse_gfni(low: &mut [u8], high: &mut [u8], coefficient: gf8b::Elem) {
    let coeff = _mm256_set1_epi8(coefficient.to_raw().cast_signed());
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
    scalar::fused_inverse::<Gf8B>(&mut low[vector_len..], &mut high[vector_len..], coefficient);
}
#[target_feature(enable = "avx2,gfni")]
pub unsafe fn gf16_fused_forward_gfni(low: &mut [u8], high: &mut [u8], coefficient: gf16::Elem) {
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
pub unsafe fn gf16_fused_inverse_gfni(low: &mut [u8], high: &mut [u8], coefficient: gf16::Elem) {
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
