//! Per-field butterfly kernel wiring for `fgf`'s binary fields.
//!
//! GF(2^8) and GF(2^16) have dedicated SIMD butterflies, so their
//! [`TierButterflies`] entries forward the capability token and the halves to
//! the architecture kernels; every other field takes the trait's scalar
//! defaults.

use fgf::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf8B, Gf8D, Gf16, Gf32, Gf64};

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use super::aarch64;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::x86;
use super::{BUTTERFLY_FFT_TIERS, Backend, ButterflyKernels, TierButterflies};

impl super::private::Sealed for Gf8B {}
impl super::private::Sealed for Gf8D {}
impl super::private::Sealed for Gf16 {}
impl super::private::Sealed for Gf32 {}
impl super::private::Sealed for Gf64 {}
impl super::private::Sealed for FanPaar8 {}
impl super::private::Sealed for FanPaar16 {}
impl super::private::Sealed for FanPaar32 {}
impl super::private::Sealed for FanPaar64 {}

impl ButterflyKernels for Gf8B {
    const BUTTERFLY_TIERS: &'static [Backend] = BUTTERFLY_FFT_TIERS;
}

impl TierButterflies for Gf8B {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_gfni(
        token: archmage::X64V3GfniCryptoToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf8_fused_forward_gfni(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_gfni(
        token: archmage::X64V3GfniCryptoToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf8_fused_inverse_gfni(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_avx2(
        token: archmage::X64V3Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf8_fused_forward_avx2(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_avx2(
        token: archmage::X64V3Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf8_fused_inverse_avx2(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_ssse3(
        token: archmage::X64V2Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf8_fused_forward_ssse3(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_ssse3(
        token: archmage::X64V2Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf8_fused_inverse_ssse3(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    fn fused_forward_neon(
        token: archmage::NeonToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        aarch64::gf8_fused_forward_neon(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    fn fused_inverse_neon(
        token: archmage::NeonToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        aarch64::gf8_fused_inverse_neon(token, low, high, coefficient);
    }
}

impl ButterflyKernels for Gf16 {
    const BUTTERFLY_TIERS: &'static [Backend] = BUTTERFLY_FFT_TIERS;
}

impl TierButterflies for Gf16 {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_gfni(
        token: archmage::X64V3GfniCryptoToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf16_fused_forward_gfni(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_gfni(
        token: archmage::X64V3GfniCryptoToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf16_fused_inverse_gfni(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_avx2(
        token: archmage::X64V3Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf16_fused_forward_avx2(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_avx2(
        token: archmage::X64V3Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf16_fused_inverse_avx2(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_ssse3(
        token: archmage::X64V2Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf16_fused_forward_ssse3(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_ssse3(
        token: archmage::X64V2Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        x86::gf16_fused_inverse_ssse3(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    fn fused_forward_neon(
        token: archmage::NeonToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        aarch64::gf16_fused_forward_neon(token, low, high, coefficient);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    fn fused_inverse_neon(
        token: archmage::NeonToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        aarch64::gf16_fused_inverse_neon(token, low, high, coefficient);
    }
}

impl ButterflyKernels for Gf32 {}
impl TierButterflies for Gf32 {}
impl ButterflyKernels for Gf64 {}
impl TierButterflies for Gf64 {}
impl ButterflyKernels for FanPaar8 {}
impl TierButterflies for FanPaar8 {}
impl ButterflyKernels for FanPaar16 {}
impl TierButterflies for FanPaar16 {}
impl ButterflyKernels for FanPaar32 {}
impl TierButterflies for FanPaar32 {}
impl ButterflyKernels for FanPaar64 {}
impl TierButterflies for FanPaar64 {}

impl ButterflyKernels for Gf8D {}
impl TierButterflies for Gf8D {}
