//! Fused butterfly kernels with runtime SIMD dispatch.
//!
//! One-pass fused butterflies over interleaved byte halves
//! (`low' = low ⊕ c·high`, `high' = low' ⊕ high`, and the inverse pairing),
//! with the zero-coefficient XOR-coupling fast path. The backend is selected
//! once per process; every SIMD backend is differentially tested against the
//! portable scalar implementation, which also serves vector tails.
//!
//! ## Backends
//!
//! cafft supports a subset of [`fff::kernel::Backend`]: AVX-512 hosts run the
//! GFNI kernels (AVX-512 detection implies GFNI), and WebAssembly runs scalar
//! until dedicated `simd128` butterflies exist. GF(2^8) and GF(2^16) have
//! dedicated kernels on `x86` (GFNI/AVX2/SSSE3) and `AArch64` (NEON); wider
//! fields always report [`Backend::Scalar`].
//!
//! The process backend may be downgraded at startup via the `CAFFT_BACKEND`
//! environment variable (`gfni`, `avx2`, `ssse3`, `neon`, `scalar`), applied
//! after — and independently of — fff's own `FFF_BACKEND`. Requests for a
//! backend the host cannot run are ignored: running vector code without the
//! instruction set is undefined behaviour, not a configuration choice.

// Unsafe is expected and confined here: this module owns every intrinsic in
// the crate, all behind runtime feature detection. The rest of the crate
// keeps `#![warn(unsafe_code)]`.
#![allow(unsafe_code)]

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64;
mod scalar;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86;

use ::core::marker::PhantomData;

use fff::field::Elem;
use fff::kernel::FieldKernels;
use fff::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, Gf8, Gf16, Gf32, Gf64};

pub use fff::kernel::Backend;

#[cfg(feature = "std")]
static BACKEND: ::std::sync::LazyLock<Backend> = ::std::sync::LazyLock::new(resolve_backend);

/// Map fff's backend onto the set cafft implements.
const fn cap(backend: Backend) -> Backend {
    match backend {
        // cafft uses its 32-byte AVX2+GFNI kernels on AVX-512 hosts. Host
        // support for both target features is checked separately below.
        Backend::Avx512 => Backend::Gfni,
        // No wasm butterflies yet.
        Backend::Simd128 => Backend::Scalar,
        other => other,
    }
}

/// Whether every target feature required by `backend` is available now.
#[cfg(feature = "std")]
fn supported_on_host(backend: Backend) -> bool {
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Gfni => {
            ::std::arch::is_x86_feature_detected!("avx2")
                && ::std::arch::is_x86_feature_detected!("gfni")
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => ::std::arch::is_x86_feature_detected!("avx2"),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Ssse3 => ::std::arch::is_x86_feature_detected!("ssse3"),
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => ::std::arch::is_aarch64_feature_detected!("neon"),
        Backend::Scalar => true,
        _ => false,
    }
}

#[cfg(feature = "std")]
fn resolve_backend() -> Backend {
    // fff's resolution already honored FFF_BACKEND; cap to cafft's set,
    // then independently prove every feature required by our target-feature
    // kernels. Scalar is the safe fallback for unusual feature masks.
    let detected = cap(fff::kernel::backend());
    let detected = if supported_on_host(detected) {
        detected
    } else {
        Backend::Scalar
    };
    match ::std::env::var("CAFFT_BACKEND") {
        Ok(name) => match Backend::from_name(name.trim()) {
            // Downgrade-only: requests must be weaker than the detected set.
            Some(requested) if supported_on_host(requested) && requested >= detected => requested,
            _ => detected,
        },
        Err(_) => detected,
    }
}

/// The backend the butterfly kernels run on, detected once per process.
///
/// May be downgraded at startup via `CAFFT_BACKEND`; see the module docs.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    #[cfg(feature = "std")]
    {
        *BACKEND
    }
    #[cfg(not(feature = "std"))]
    {
        Backend::Scalar
    }
}

/// The backend used for field `F`: the process backend capped to what the
/// field supports. Wider fields always report [`Backend::Scalar`].
#[inline]
#[must_use]
pub fn backend_for<F: ButterflyKernels>() -> Backend {
    let field_cap = cap(fff::kernel::backend_for::<F>());
    let ours = backend();
    // fff::kernel::Backend orders weaker backends greater; take the weaker.
    if ours > field_cap { ours } else { field_cap }
}

mod private {
    pub trait Sealed {}
}

impl private::Sealed for Gf8 {}
impl private::Sealed for Gf16 {}
impl private::Sealed for Gf32 {}
impl private::Sealed for Gf64 {}
impl private::Sealed for FanPaar8 {}
impl private::Sealed for FanPaar16 {}
impl private::Sealed for FanPaar32 {}
impl private::Sealed for FanPaar64 {}

/// The per-field butterfly kernel contract.
///
/// Sealed: cafft implements this for fff's fields, and the set of fields
/// with dedicated SIMD kernels (currently GF(2^8) and GF(2^16)) is fixed by
/// the implementation. Fields without dedicated kernels inherit the portable
/// scalar defaults, which is why every transform works over every fff field.
///
/// Callers should use the safe wrappers ([`fused_forward`], [`fused_inverse`]
/// and, in-crate, the backend structs plus the `dispatch_butterfly!`
/// monomorphized dispatch) rather than these methods directly.
///
/// # Safety
///
/// Each `unsafe` method requires that the instruction set in its name was
/// runtime-detected (see [`backend`]); the dispatch layer upholds this by
/// construction. The scalar defaults are always safe to call.
pub trait ButterflyKernels: FieldKernels + private::Sealed {
    /// Fused forward butterfly, AVX2+GFNI kernel.
    ///
    /// # Safety
    /// Requires AVX2 and GFNI; the byte slices must be equal-length with a
    /// whole number of elements.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_gfni(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, AVX2+GFNI kernel.
    ///
    /// # Safety
    /// Same contract as [`ButterflyKernels::fused_forward_gfni`].
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_gfni(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }

    /// Fused forward butterfly, AVX2 kernel.
    ///
    /// # Safety
    /// Requires AVX2; the byte slices must be equal-length with a whole
    /// number of elements.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_avx2(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, AVX2 kernel.
    ///
    /// # Safety
    /// Same contract as [`ButterflyKernels::fused_forward_avx2`].
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_avx2(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }

    /// Fused forward butterfly, SSSE3 kernel.
    ///
    /// # Safety
    /// Requires SSSE3; the byte slices must be equal-length with a whole
    /// number of elements.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_ssse3(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, SSSE3 kernel.
    ///
    /// # Safety
    /// Same contract as [`ButterflyKernels::fused_forward_ssse3`].
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_ssse3(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }

    /// Fused forward butterfly, NEON kernel.
    ///
    /// # Safety
    /// Requires NEON (baseline on AArch64); the byte slices must be
    /// equal-length with a whole number of elements.
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn fused_forward_neon(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, NEON kernel.
    ///
    /// # Safety
    /// Same contract as [`ButterflyKernels::fused_forward_neon`].
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn fused_inverse_neon(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }
}

impl ButterflyKernels for Gf8 {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_gfni(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects GFNI only after detection.
        unsafe { x86::gf8_fused_forward_gfni(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_gfni(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects GFNI only after detection.
        unsafe { x86::gf8_fused_inverse_gfni(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_avx2(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects AVX2 only after detection.
        unsafe { x86::gf8_fused_forward_avx2(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_avx2(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects AVX2 only after detection.
        unsafe { x86::gf8_fused_inverse_avx2(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_ssse3(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects SSSE3 only after detection.
        unsafe { x86::gf8_fused_forward_ssse3(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_ssse3(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects SSSE3 only after detection.
        unsafe { x86::gf8_fused_inverse_ssse3(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn fused_forward_neon(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { aarch64::gf8_fused_forward_neon(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn fused_inverse_neon(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { aarch64::gf8_fused_inverse_neon(low, high, coefficient) }
    }
}

impl ButterflyKernels for Gf16 {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_gfni(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects GFNI only after detection.
        unsafe { x86::gf16_fused_forward_gfni(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_gfni(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects GFNI only after detection.
        unsafe { x86::gf16_fused_inverse_gfni(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_avx2(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects AVX2 only after detection.
        unsafe { x86::gf16_fused_forward_avx2(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_avx2(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects AVX2 only after detection.
        unsafe { x86::gf16_fused_inverse_avx2(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_forward_ssse3(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects SSSE3 only after detection.
        unsafe { x86::gf16_fused_forward_ssse3(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    unsafe fn fused_inverse_ssse3(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: the dispatch layer selects SSSE3 only after detection.
        unsafe { x86::gf16_fused_inverse_ssse3(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn fused_forward_neon(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { aarch64::gf16_fused_forward_neon(low, high, coefficient) }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    unsafe fn fused_inverse_neon(low: &mut [u8], high: &mut [u8], coefficient: Self::Elem) {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { aarch64::gf16_fused_inverse_neon(low, high, coefficient) }
    }
}

impl ButterflyKernels for Gf32 {}
impl ButterflyKernels for Gf64 {}
impl ButterflyKernels for FanPaar8 {}
impl ButterflyKernels for FanPaar16 {}
impl ButterflyKernels for FanPaar32 {}
impl ButterflyKernels for FanPaar64 {}

/// A backend tag type: selects which kernel set executes the butterflies.
///
/// The transform walkers are generic over `B: ButterflyBackend<F>`, so the
/// whole recursion monomorphizes onto one backend — dispatch happens once
/// per transform call, never per butterfly. Instantiated only through the
/// `dispatch_butterfly!` macro; the trait methods are static.
pub(crate) trait ButterflyBackend<F: ButterflyKernels> {
    /// `low' = low ⊕ c·high`, `high' = low' ⊕ high` for nonzero `c`.
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem);
    /// `high' = high ⊕ low`, `low' = low ⊕ c·high'` for nonzero `c`.
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem);
}

pub(crate) struct ScalarBackend<F>(PhantomData<fn() -> F>);

impl<F: ButterflyKernels> ButterflyBackend<F> for ScalarBackend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        scalar::fused_forward::<F>(low, high, coefficient);
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        scalar::fused_inverse::<F>(low, high, coefficient);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) struct GfniBackend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<F: ButterflyKernels> ButterflyBackend<F> for GfniBackend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: this tag is selected only after AVX2+GFNI detection.
        unsafe { F::fused_forward_gfni(low, high, coefficient) }
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: this tag is selected only after AVX2+GFNI detection.
        unsafe { F::fused_inverse_gfni(low, high, coefficient) }
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) struct Avx2Backend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<F: ButterflyKernels> ButterflyBackend<F> for Avx2Backend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: this tag is selected only after AVX2 detection.
        unsafe { F::fused_forward_avx2(low, high, coefficient) }
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: this tag is selected only after AVX2 detection.
        unsafe { F::fused_inverse_avx2(low, high, coefficient) }
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) struct Ssse3Backend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<F: ButterflyKernels> ButterflyBackend<F> for Ssse3Backend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: this tag is selected only after SSSE3 detection.
        unsafe { F::fused_forward_ssse3(low, high, coefficient) }
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: this tag is selected only after SSSE3 detection.
        unsafe { F::fused_inverse_ssse3(low, high, coefficient) }
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) struct NeonBackend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
impl<F: ButterflyKernels> ButterflyBackend<F> for NeonBackend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { F::fused_forward_neon(low, high, coefficient) }
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { F::fused_inverse_neon(low, high, coefficient) }
    }
}

/// Dispatch a kernel-generic walker onto the process backend for field `F`.
///
/// Expands `$function::<F, B>(args…)` with `B` the selected backend tag, so
/// the entire walker monomorphizes onto one backend per call.
macro_rules! dispatch_butterfly {
    ($field:ty, $function:ident ($($argument:expr),* $(,)?)) => {
        match $crate::core::kernel::backend_for::<$field>() {
            $crate::core::kernel::Backend::Scalar => {
                $function::<$field, $crate::core::kernel::ScalarBackend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            $crate::core::kernel::Backend::Gfni => {
                $function::<$field, $crate::core::kernel::GfniBackend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            $crate::core::kernel::Backend::Avx2 => {
                $function::<$field, $crate::core::kernel::Avx2Backend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            $crate::core::kernel::Backend::Ssse3 => {
                $function::<$field, $crate::core::kernel::Ssse3Backend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            $crate::core::kernel::Backend::Neon => {
                $function::<$field, $crate::core::kernel::NeonBackend<$field>>($($argument),*)
            }
            // backend_for never yields a backend outside the cfg'd set; the
            // wildcard only covers `Backend`'s non-exhaustiveness.
            #[allow(unreachable_patterns)]
            _ => $function::<$field, $crate::core::kernel::ScalarBackend<$field>>($($argument),*),
        }
    };
}

// Walkers in `core::transform` call this through the path re-export.
pub(crate) use dispatch_butterfly;

/// Fused forward butterfly on backend `B`, with the zero-coefficient
/// XOR-coupling fast path (`high ^= low`, leaving `low` untouched).
#[inline]
pub(crate) fn fused_forward_with<F: ButterflyKernels, B: ButterflyBackend<F>>(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: F::Elem,
) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % F::BYTES, 0);
    if coefficient.is_zero() {
        fff::ops::add_assign::<F>(high, low);
    } else {
        B::forward_nonzero(low, high, coefficient);
    }
}

/// Fused inverse butterfly on backend `B`, with the zero-coefficient
/// XOR-coupling fast path.
#[inline]
pub(crate) fn fused_inverse_with<F: ButterflyKernels, B: ButterflyBackend<F>>(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: F::Elem,
) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % F::BYTES, 0);
    if coefficient.is_zero() {
        fff::ops::add_assign::<F>(high, low);
    } else {
        B::inverse_nonzero(low, high, coefficient);
    }
}

/// Fused forward butterfly over two equal-length interleaved byte halves:
/// `low' = low ⊕ c·high`, `high' = low' ⊕ high`.
///
/// Each half is read and written once. Dispatches to the best backend for
/// `F`; for repeated calls over the same backend (transform walkers), the
/// in-crate `dispatch_butterfly!` hoists dispatch out of the recursion.
///
/// # Panics
/// Panics if `low` and `high` differ in length or hold a partial trailing
/// element.
#[inline]
pub fn fused_forward<F: ButterflyKernels>(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
    assert_eq!(
        low.len(),
        high.len(),
        "butterfly halves must have equal length"
    );
    assert_eq!(low.len() % F::BYTES, 0, "partial trailing element");
    crate::core::kernel::dispatch_butterfly!(F, fused_forward_with(low, high, coefficient));
}

/// Fused inverse butterfly: `high' = high ⊕ low`, `low' = low ⊕ c·high'`.
///
/// Undoes [`fused_forward`] with the same coefficient.
///
/// # Panics
/// Panics if `low` and `high` differ in length or hold a partial trailing
/// element.
#[inline]
pub fn fused_inverse<F: ButterflyKernels>(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
    assert_eq!(
        low.len(),
        high.len(),
        "butterfly halves must have equal length"
    );
    assert_eq!(low.len() % F::BYTES, 0, "partial trailing element");
    crate::core::kernel::dispatch_butterfly!(F, fused_inverse_with(low, high, coefficient));
}

/// XOR `coefficient * src` into `dst`, element by element.
///
/// Thin wrapper over [`fff::ops::mul_add`] with the zero/one-coefficient
/// fast paths; the workhorse for coefficient-scaled row accumulation outside
/// the butterfly recursion.
///
/// # Panics
/// Panics if `dst` and `src` differ in length or hold a partial trailing
/// element.
#[inline]
pub fn xor_scaled_bytes<F: ButterflyKernels>(dst: &mut [u8], coefficient: F::Elem, src: &[u8]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "scaled buffers must have equal length"
    );
    assert_eq!(src.len() % F::BYTES, 0, "partial trailing element");
    if coefficient.is_zero() {
        return;
    }
    if coefficient.is_one() {
        fff::ops::add_assign::<F>(dst, src);
        return;
    }
    fff::ops::mul_add::<F>(dst, coefficient, src);
}

/// XOR one source row into every destination row with a distinct coefficient.
///
/// `destinations` contains `coefficients.len()` contiguous rows of
/// `row_len` bytes. Row `j` becomes
/// `row_j ⊕ coefficients[j]·source`, element by element.
///
/// # Panics
/// Panics if `row_len` is zero or holds a partial trailing element, if
/// `source.len() != row_len`, if the destination geometry does not match, or
/// if its complete byte length is not representable by [`usize`].
#[inline]
pub fn xor_scaled_bytes_rows<F: ButterflyKernels>(
    destinations: &mut [u8],
    row_len: usize,
    coefficients: &[F::Elem],
    source: &[u8],
) {
    assert_ne!(row_len, 0, "row length must be nonzero");
    assert_eq!(row_len % F::BYTES, 0, "partial trailing element");
    assert_eq!(source.len(), row_len, "source length must equal row length");
    let expected = coefficients
        .len()
        .checked_mul(row_len)
        .expect("destination byte length overflow");
    assert_eq!(
        destinations.len(),
        expected,
        "destination rows do not match coefficients"
    );
    for (row, &coefficient) in destinations.chunks_exact_mut(row_len).zip(coefficients) {
        xor_scaled_bytes::<F>(row, coefficient, source);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::alloc::vec::Vec;
    use fff::field::{Elem, Field};

    #[cfg(feature = "std")]
    #[test]
    fn resolved_backend_has_all_required_target_features() {
        let resolved = backend();
        assert!(supported_on_host(resolved));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if resolved == Backend::Gfni {
            assert!(::std::arch::is_x86_feature_detected!("avx2"));
            assert!(::std::arch::is_x86_feature_detected!("gfni"));
        }
    }

    fn pattern(seed: u8, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len);
        let mut x = seed;
        for _ in 0..len {
            bytes.push(x);
            x = x.wrapping_mul(29).wrapping_add(0xa5);
        }
        bytes
    }

    /// Byte lengths covering empty, sub-lane, exact-lane, and tailed shapes.
    fn lengths(bytes_per_elem: usize) -> Vec<usize> {
        let elems = [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 65];
        elems.map(|e| e * bytes_per_elem).into_iter().collect()
    }

    fn scalar_roundtrip<F: ButterflyKernels>(coefficients: &[F::Elem]) {
        for &coefficient in coefficients {
            for len in lengths(F::BYTES) {
                let low = pattern(0x11, len);
                let high = pattern(0x97, len);
                let (original_low, original_high) = (low.clone(), high.clone());
                let (mut low, mut high) = (low, high);
                scalar::fused_forward::<F>(&mut low, &mut high, coefficient);
                scalar::fused_inverse::<F>(&mut low, &mut high, coefficient);
                assert_eq!(low, original_low, "low not restored at len {len}");
                assert_eq!(high, original_high, "high not restored at len {len}");
            }
        }
    }

    #[test]
    fn scalar_roundtrip_gf8() {
        let coefficients = [0x00, 0x01, 0x02, 0x03, 0x53, 0xff].map(fff::gf8::Elem);
        scalar_roundtrip::<Gf8>(&coefficients);
    }

    #[test]
    fn scalar_roundtrip_gf16() {
        let coefficients = [0x0000, 0x0001, 0x0108, 0x9b37, 0xffff].map(fff::gf16::Elem);
        scalar_roundtrip::<Gf16>(&coefficients);
    }

    #[test]
    fn scalar_forward_matches_element_math_gf8() {
        for coefficient in [0x00, 0x01, 0x03, 0xff].map(fff::gf8::Elem) {
            for len in lengths(1) {
                let low = pattern(0x3d, len);
                let high = pattern(0xc2, len);
                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                for (l, h) in expected_low.iter_mut().zip(&mut expected_high) {
                    let lo = fff::gf8::Elem(*l);
                    let hi = fff::gf8::Elem(*h);
                    let new_low = lo.add(coefficient.mul(hi));
                    *l = new_low.0;
                    *h = hi.add(new_low).0;
                }
                let (mut low, mut high) = (low, high);
                scalar::fused_forward::<Gf8>(&mut low, &mut high, coefficient);
                assert_eq!(low, expected_low);
                assert_eq!(high, expected_high);
            }
        }
    }

    #[test]
    fn scalar_forward_matches_element_math_gf16() {
        for coefficient in [0x0000, 0x0001, 0x0108, 0xbeef].map(fff::gf16::Elem) {
            for len in lengths(2) {
                let low = pattern(0x3d, len);
                let high = pattern(0xc2, len);
                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                for (l, h) in expected_low
                    .chunks_exact_mut(2)
                    .zip(expected_high.chunks_exact_mut(2))
                {
                    let lo = Gf16::read(l);
                    let hi = Gf16::read(h);
                    let new_low = lo.add(coefficient.mul(hi));
                    Gf16::write(l, new_low);
                    Gf16::write(h, hi.add(new_low));
                }
                let (mut low, mut high) = (low, high);
                scalar::fused_forward::<Gf16>(&mut low, &mut high, coefficient);
                assert_eq!(low, expected_low);
                assert_eq!(high, expected_high);
            }
        }
    }

    #[test]
    fn public_butterflies_roundtrip() {
        fn check<F: ButterflyKernels>(coefficients: &[F::Elem]) {
            for &coefficient in coefficients {
                for len in lengths(F::BYTES) {
                    let low = pattern(0x5b, len);
                    let high = pattern(0xe4, len);
                    let (original_low, original_high) = (low.clone(), high.clone());
                    let (mut low, mut high) = (low, high);
                    fused_forward::<F>(&mut low, &mut high, coefficient);
                    fused_inverse::<F>(&mut low, &mut high, coefficient);
                    assert_eq!(low, original_low);
                    assert_eq!(high, original_high);
                }
            }
        }
        check::<Gf8>(&[0x00, 0x01, 0x53, 0xff].map(fff::gf8::Elem));
        check::<Gf16>(&[0x0000, 0x0001, 0x0108, 0x9b37].map(fff::gf16::Elem));
    }

    #[test]
    fn public_forward_zero_coefficient_is_xor_coupling() {
        fn check<F: ButterflyKernels>() {
            for len in lengths(F::BYTES) {
                let low = pattern(0x77, len);
                let high = pattern(0x08, len);
                let expected_high: Vec<u8> = high.iter().zip(&low).map(|(h, l)| h ^ l).collect();
                let original_low = low.clone();
                let (mut low, mut high) = (low, high);
                fused_forward::<F>(&mut low, &mut high, F::Elem::ZERO);
                assert_eq!(low, original_low, "low must be untouched");
                assert_eq!(high, expected_high);
            }
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    #[test]
    fn xor_scaled_matches_element_math() {
        fn check<F: ButterflyKernels>(coefficient: F::Elem) {
            for len in lengths(F::BYTES) {
                let src = pattern(0x19, len);
                let mut dst = pattern(0xb3, len);
                let mut expected = dst.clone();
                for (out, input) in expected
                    .chunks_exact_mut(F::BYTES)
                    .zip(src.chunks_exact(F::BYTES))
                {
                    let product = F::read(input).mul(coefficient);
                    let acc = F::read(out).add(product);
                    F::write(out, acc);
                }
                xor_scaled_bytes::<F>(&mut dst, coefficient, &src);
                assert_eq!(dst, expected);
            }
        }
        check::<Gf8>(fff::gf8::Elem(0x53));
        check::<Gf8>(fff::gf8::Elem(0x00));
        check::<Gf8>(fff::gf8::Elem(0x01));
        check::<Gf16>(fff::gf16::Elem(0x9b37));
        check::<Gf16>(fff::gf16::Elem(0x0000));
        check::<Gf16>(fff::gf16::Elem(0x0001));
    }

    #[test]
    fn xor_scaled_rows_match_element_math() {
        fn check<F: ButterflyKernels>(coefficients: &[F::Elem]) {
            let row_len = 3 * F::BYTES;
            let source = pattern(0x31, row_len);
            let mut destinations = pattern(0xc7, coefficients.len() * row_len);
            let mut expected = destinations.clone();
            for (row, &coefficient) in expected.chunks_exact_mut(row_len).zip(coefficients) {
                for (out, input) in row
                    .chunks_exact_mut(F::BYTES)
                    .zip(source.chunks_exact(F::BYTES))
                {
                    let value = F::read(out).add(coefficient.mul(F::read(input)));
                    F::write(out, value);
                }
            }
            xor_scaled_bytes_rows::<F>(&mut destinations, row_len, coefficients, &source);
            assert_eq!(destinations, expected);
        }

        check::<Gf8>(&[0x00, 0x01, 0x53, 0xff].map(fff::gf8::Elem));
        check::<Gf16>(&[0x0000, 0x0001, 0x9b37, 0xffff].map(fff::gf16::Elem));
    }

    /// Compare one concrete SIMD backend against the scalar reference in
    /// both directions independently. A round trip is insufficient because
    /// paired forward/inverse errors can cancel.
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn differential_backend<F: ButterflyKernels, B: ButterflyBackend<F>>(
        label: &str,
        coefficients: &[F::Elem],
    ) {
        for &coefficient in coefficients {
            for len in lengths(F::BYTES) {
                let low = pattern(0x2e, len);
                let high = pattern(0xd1, len);

                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                scalar::fused_forward::<F>(&mut expected_low, &mut expected_high, coefficient);
                let mut actual_low = low.clone();
                let mut actual_high = high.clone();
                fused_forward_with::<F, B>(&mut actual_low, &mut actual_high, coefficient);
                assert_eq!(
                    actual_low, expected_low,
                    "{label} forward low diverged at len {len}"
                );
                assert_eq!(
                    actual_high, expected_high,
                    "{label} forward high diverged at len {len}"
                );

                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                scalar::fused_inverse::<F>(&mut expected_low, &mut expected_high, coefficient);
                let mut actual_low = low;
                let mut actual_high = high;
                fused_inverse_with::<F, B>(&mut actual_low, &mut actual_high, coefficient);
                assert_eq!(
                    actual_low, expected_low,
                    "{label} inverse low diverged at len {len}"
                );
                assert_eq!(
                    actual_high, expected_high,
                    "{label} inverse high diverged at len {len}"
                );
            }
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn differential_x86<F: ButterflyKernels>(coefficients: &[F::Elem]) {
        if ::std::arch::is_x86_feature_detected!("avx2")
            && ::std::arch::is_x86_feature_detected!("gfni")
        {
            differential_backend::<F, GfniBackend<F>>("gfni", coefficients);
        }
        if ::std::arch::is_x86_feature_detected!("avx2") {
            differential_backend::<F, Avx2Backend<F>>("avx2", coefficients);
        }
        if ::std::arch::is_x86_feature_detected!("ssse3") {
            differential_backend::<F, Ssse3Backend<F>>("ssse3", coefficients);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_backends_match_scalar_in_both_directions() {
        differential_x86::<Gf8>(&[0x00, 0x01, 0x02, 0x53, 0xff].map(fff::gf8::Elem));
        differential_x86::<Gf16>(&[0x0000, 0x0001, 0x0108, 0x9b37, 0xffff].map(fff::gf16::Elem));
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn neon_backends_match_scalar_in_both_directions() {
        differential_backend::<Gf8, NeonBackend<Gf8>>(
            "neon",
            &[0x00, 0x01, 0x02, 0x53, 0xff].map(fff::gf8::Elem),
        );
        differential_backend::<Gf16, NeonBackend<Gf16>>(
            "neon",
            &[0x0000, 0x0001, 0x0108, 0x9b37, 0xffff].map(fff::gf16::Elem),
        );
    }
}
