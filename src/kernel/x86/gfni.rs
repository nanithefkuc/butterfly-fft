//! GFNI butterfly kernels for GF(2^8) and GF(2^16).
//!
//! `GF2P8MULB` multiplies 32 GF(2^8) lanes in the AES polynomial directly;
//! GF(2^16) uses the interleaved-component trick shared with the parent
//! module. Every kernel is a safe [`archmage`] capability-token function
//! taking the exact token its instructions require
//! (`X64V3GfniCryptoToken`); sub-lane tails go to the portable scalar
//! kernels on an element boundary. Only GFNI-capable hosts execute these
//! kernels; the coverage gate excludes this file because the hosted CI fleet
//! mixes Xeon generations and a GFNI-less runner cannot execute a line of it
//! — the direct-kernel differential tests cover it wherever the host can
//! summon the tier.
//!
//! [`archmage`]: https://docs.rs/archmage

#[cfg(target_arch = "x86")]
use ::core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use ::core::arch::x86_64::*;

use fgf::{Gf8B, Gf16, gf8b, gf16};

use super::scalar;
use super::{SWAP_ADJACENT, factor_words};

/// `source * coefficient` for interleaved GF(2^16) elements, GFNI form.
#[archmage::rite(v3_gfni_crypto, import_intrinsics)]
fn scaled_vector_gfni(source: __m256i, same: i16, cross: i16) -> __m256i {
    let swap_mask = _mm256_broadcastsi128_si256(_mm_loadu_si128(&SWAP_ADJACENT));
    let swapped = _mm256_shuffle_epi8(source, swap_mask);
    let direct = _mm256_gf2p8mul_epi8(source, _mm256_set1_epi16(same));
    let crossed = _mm256_gf2p8mul_epi8(swapped, _mm256_set1_epi16(cross));
    _mm256_xor_si256(direct, crossed)
}

/// Fused forward butterfly over 32-byte GFNI lanes, GF(2^8) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn gf8_fused_forward_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8b::Elem,
) {
    let coeff = _mm256_set1_epi8(coefficient.to_raw().cast_signed());
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let new_low = _mm256_xor_si256(l, _mm256_gf2p8mul_epi8(h, coeff));
        let new_high = _mm256_xor_si256(h, new_low);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_forward::<Gf8B>(low_tail, high_tail, coefficient);
}

/// Fused inverse butterfly over 32-byte GFNI lanes, GF(2^8) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn gf8_fused_inverse_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf8b::Elem,
) {
    let coeff = _mm256_set1_epi8(coefficient.to_raw().cast_signed());
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let new_high = _mm256_xor_si256(h, l);
        let new_low = _mm256_xor_si256(l, _mm256_gf2p8mul_epi8(new_high, coeff));
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_inverse::<Gf8B>(low_tail, high_tail, coefficient);
}

/// Fused forward butterfly over 32-byte GFNI lanes, GF(2^16) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn gf16_fused_forward_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let (same, cross) = factor_words(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let scaled = scaled_vector_gfni(h, same, cross);
        let new_low = _mm256_xor_si256(l, scaled);
        let new_high = _mm256_xor_si256(h, new_low);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_forward::<Gf16>(low_tail, high_tail, coefficient);
}

/// Fused inverse butterfly over 32-byte GFNI lanes, GF(2^16) coefficients.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn gf16_fused_inverse_gfni(
    _token: archmage::X64V3GfniCryptoToken,
    low: &mut [u8],
    high: &mut [u8],
    coefficient: gf16::Elem,
) {
    let (same, cross) = factor_words(coefficient);
    let (low_lanes, low_tail) = low.as_chunks_mut::<32>();
    let (high_lanes, high_tail) = high.as_chunks_mut::<32>();
    for (low_lane, high_lane) in low_lanes.iter_mut().zip(high_lanes) {
        let l = _mm256_loadu_si256(&*low_lane);
        let h = _mm256_loadu_si256(&*high_lane);
        let new_high = _mm256_xor_si256(h, l);
        let scaled = scaled_vector_gfni(new_high, same, cross);
        let new_low = _mm256_xor_si256(l, scaled);
        _mm256_storeu_si256(low_lane, new_low);
        _mm256_storeu_si256(high_lane, new_high);
    }
    scalar::fused_inverse::<Gf16>(low_tail, high_tail, coefficient);
}
