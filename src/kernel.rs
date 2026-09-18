//! Fused butterfly kernels with runtime SIMD dispatch.
//!
//! One-pass fused butterflies over interleaved byte halves
//! (`low' = low ⊕ c·high`, `high' = low' ⊕ high`, and the inverse pairing),
//! with the zero-coefficient XOR-coupling fast path. The backend is selected
//! once per process; every SIMD backend is differentially tested against the
//! portable scalar implementation, which also serves vector tails.
//!
//! # Backends
//!
//! `butterfly-fft` supports a subset of [`Backend`] (a re-export of
//! [`simdispatch::Backend`](https://docs.rs/simdispatch)): the
//! `v3_gfni_crypto` / `v3` / `v2` tiers on x86 and `neon` / `neon_aes` on
//! `AArch64` GF(2^8) and GF(2^16) butterflies. WebAssembly runs scalar (no
//! dedicated `wasm128` butterflies), and wider fields always report
//! [`Backend::Scalar`]. Detection, ordering, and the downgrade-only override
//! are single-source: `Selection` resolves over [`BUTTERFLY_FFT_TIERS`].
//!
//! The process backend may be downgraded at startup via the one stack-wide
//! `SIMD_BACKEND` environment variable (`v3_gfni_crypto`, `v3`, `v2`,
//! `neon_aes`, `neon`, `scalar`), owned by `simdispatch` and shared with fgf.
//! Requests for a backend the host cannot run are ignored: running vector code
//! without the instruction set is undefined behaviour, not a configuration
//! choice.
//!
//! On an `AArch64` `PMULL` host this resolves to [`Backend::NeonAes`] (the
//! tier's `aes` feature proves `PMULL`) even though the butterflies run the
//! same NEON kernels as the plain [`Backend::Neon`] tier; only the reported
//! tier label differs.

// Every SIMD kernel is a safe `archmage` capability-token function; the crate
// is `#![deny(unsafe_code)]` and this module owns no unsafe surface.

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64;
mod fields;
mod scalar;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86;

use ::core::marker::PhantomData;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;

// The backend ladder and the downgrade-only override are owned by
// `simdispatch` (Level 0); fgf re-exports the same ladder, so `Backend`
// here and `fgf::kernel::Backend` are the same type.
pub use fgf::kernel::Backend;

// Only the SIMD resolve path consults the environment: a `simd`-less build
// reports `Scalar`.
#[cfg(feature = "simd")]
use simdispatch::Selection;
// `summon()` is the `SimdToken` trait's capability probe.
#[cfg(all(
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
use archmage::SimdToken;

/// The butterfly tiers compiled into this build, in detection-preference
/// order. SIMD tiers require `simd` and their target architecture; scalar is
/// always available. `AArch64` keeps both `NeonAes` and `Neon` so `PMULL`
/// hosts resolve to the crypto tier rather than silently reporting `Neon`.
pub const BUTTERFLY_FFT_TIERS: &[Backend] = &[
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    Backend::V3GfniCrypto,
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    Backend::V3,
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    Backend::V2,
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    Backend::NeonAes,
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    Backend::Neon,
    Backend::Scalar,
];

/// The backend the butterfly kernels run on, resolved once per process.
///
/// `simdispatch::Selection` over [`BUTTERFLY_FFT_TIERS`], adjusted by the one
/// stack-wide `SIMD_BACKEND` downgrade-only override; detected via the single
/// `archmage` `summon()` probe.
///
/// The `LazyLock` caches the resolve so dispatch never touches the
/// environment per call — a memoized `simdispatch` selection, not a second
/// resolver.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    #[cfg(feature = "simd")]
    {
        *BACKEND
    }
    #[cfg(not(feature = "simd"))]
    {
        Backend::Scalar
    }
}

/// Memoized [`Selection`] over [`BUTTERFLY_FFT_TIERS`].
#[cfg(feature = "simd")]
static BACKEND: ::std::sync::LazyLock<Backend> = ::std::sync::LazyLock::new(|| {
    Selection::new("SIMD_BACKEND")
        .supports(BUTTERFLY_FFT_TIERS)
        .resolve()
});

/// Capability tokens for the x86 butterfly tiers, summoned once per process.
///
/// Host-capability facts, deliberately separate from the policy
/// [`Selection`] in [`BACKEND`]: `SIMD_BACKEND` chooses *which* tier's
/// kernels run, while these prove the host *can* run each tier's kernels —
/// the direct-kernel differential tests exercise tiers the process override
/// does not select. The weaker tokens narrow from the strongest one the host
/// summons, so one probe covers the ladder.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
static X86_TOKENS: ::std::sync::LazyLock<X86Tokens> = ::std::sync::LazyLock::new(X86Tokens::summon);

/// The summoned x86 capability tokens; `None` where the host lacks the tier.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
struct X86Tokens {
    v3_gfni_crypto: Option<archmage::X64V3GfniCryptoToken>,
    v3: Option<archmage::X64V3Token>,
    v2: Option<archmage::X64V2Token>,
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl X86Tokens {
    fn summon() -> Self {
        let v3_gfni_crypto = archmage::X64V3GfniCryptoToken::summon();
        let v3 = v3_gfni_crypto
            .map(archmage::X64V3GfniCryptoToken::v3)
            .or_else(archmage::X64V3Token::summon);
        let v2 = v3
            .map(archmage::X64V3Token::v2)
            .or_else(archmage::X64V2Token::summon);
        Self {
            v3_gfni_crypto,
            v3,
            v2,
        }
    }
}

/// The NEON capability token, summoned once per process.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
static NEON_TOKEN: ::std::sync::LazyLock<Option<archmage::NeonToken>> =
    ::std::sync::LazyLock::new(|| archmage::NeonToken::summon());

/// The GFNI proof for the GFNI butterfly kernels.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn v3_gfni_crypto_token() -> archmage::X64V3GfniCryptoToken {
    X86_TOKENS
        .v3_gfni_crypto
        .expect("GFNI butterflies reached on a host without the GFNI token")
}

/// The AVX2 proof for the AVX2 butterfly kernels.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn v3_token() -> archmage::X64V3Token {
    X86_TOKENS
        .v3
        .expect("AVX2 butterflies reached on a host without the V3 token")
}

/// The SSSE3 proof for the SSSE3 butterfly kernels.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn v2_token() -> archmage::X64V2Token {
    X86_TOKENS
        .v2
        .expect("SSSE3 butterflies reached on a host without the V2 token")
}

/// The NEON proof for the NEON butterfly kernels.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn neon_token() -> archmage::NeonToken {
    NEON_TOKEN.expect("NEON butterflies reached on a host without the NEON token")
}

/// The backend used for field `F`: the cached process backend ([`backend()`],
/// already resolved and override-adjusted over [`BUTTERFLY_FFT_TIERS`]) narrowed
/// to the tiers that field's butterfly kernels implement
/// ([`ButterflyKernels::BUTTERFLY_TIERS`]). Wider fields implement none and
/// always report [`Backend::Scalar`].
///
/// The narrowing reads the cached backend rather than re-running `Selection`
/// on every call, which would touch the environment inside the
/// steady-state path.
#[inline]
#[must_use]
pub fn backend_for<F: ButterflyKernels>() -> Backend {
    let resolved = backend();
    if F::BUTTERFLY_TIERS.contains(&resolved) {
        resolved
    } else {
        Backend::Scalar
    }
}

mod private {
    pub trait Sealed {}
}

/// The per-field butterfly kernel contract: which tiers this field's
/// butterflies implement.
///
/// Sealed: `butterfly-fft` implements this for `fgf`'s binary fields, and
/// the set of fields with dedicated SIMD kernels — GF(2^8) and GF(2^16) —
/// is fixed by the implementation. Fields without dedicated
/// kernels inherit the portable scalar backend, which is why every
/// transform works over every implementor. External code holds this bound
/// but cannot name the private `TierButterflies` supertrait the per-tier
/// kernel entries live on.
#[allow(private_bounds)]
pub trait ButterflyKernels: FieldKernels + private::Sealed + TierButterflies {
    /// The tiers this field's butterfly kernels implement, used to narrow
    /// [`backend_for`]. Only GF(2^8) and GF(2^16) vectorize, so the default
    /// is scalar-only.
    const BUTTERFLY_TIERS: &'static [Backend] = &[Backend::Scalar];
}

/// The per-tier kernel entries, reachable only through the private
/// supertrait bound: each entry takes the genuine `archmage` capability
/// token its kernel's instructions require, summoned once and cached by the
/// dispatch layer ([`X86_TOKENS`], [`NEON_TOKEN`]).
///
/// # Preconditions
///
/// The byte slices must be equal-length with a whole number of elements.
pub(crate) trait TierButterflies: Field {
    /// Fused forward butterfly, AVX2+GFNI kernel.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_gfni(
        _token: archmage::X64V3GfniCryptoToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, AVX2+GFNI kernel.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_gfni(
        _token: archmage::X64V3GfniCryptoToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }

    /// Fused forward butterfly, AVX2 kernel.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_avx2(
        _token: archmage::X64V3Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, AVX2 kernel.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_avx2(
        _token: archmage::X64V3Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }

    /// Fused forward butterfly, SSSE3 kernel.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_forward_ssse3(
        _token: archmage::X64V2Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, SSSE3 kernel.
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn fused_inverse_ssse3(
        _token: archmage::X64V2Token,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }

    /// Fused forward butterfly, NEON kernel.
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    fn fused_forward_neon(
        _token: archmage::NeonToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_forward::<Self>(low, high, coefficient);
    }

    /// Fused inverse butterfly, NEON kernel.
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    fn fused_inverse_neon(
        _token: archmage::NeonToken,
        low: &mut [u8],
        high: &mut [u8],
        coefficient: Self::Elem,
    ) {
        scalar::fused_inverse::<Self>(low, high, coefficient);
    }
}

/// A backend tag type: selects which kernel set executes the butterflies.
///
/// The transform walkers are generic over `B: ButterflyBackend<F>`, so the
/// whole recursion monomorphizes onto one backend — dispatch happens once
/// per transform call, never per butterfly. Instantiated only through the
/// `dispatch_butterfly!` macro; the trait methods are static.
pub(crate) trait ButterflyBackend<F: TierButterflies> {
    /// `low' = low ⊕ c·high`, `high' = low' ⊕ high` for nonzero `c`.
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem);
    /// `high' = high ⊕ low`, `low' = low ⊕ c·high'` for nonzero `c`.
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem);
}

pub(crate) struct ScalarBackend<F>(PhantomData<fn() -> F>);

impl<F: TierButterflies> ButterflyBackend<F> for ScalarBackend<F> {
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
impl<F: TierButterflies> ButterflyBackend<F> for GfniBackend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_forward_gfni(v3_gfni_crypto_token(), low, high, coefficient);
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_inverse_gfni(v3_gfni_crypto_token(), low, high, coefficient);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) struct Avx2Backend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<F: TierButterflies> ButterflyBackend<F> for Avx2Backend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_forward_avx2(v3_token(), low, high, coefficient);
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_inverse_avx2(v3_token(), low, high, coefficient);
    }
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) struct Ssse3Backend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
impl<F: TierButterflies> ButterflyBackend<F> for Ssse3Backend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_forward_ssse3(v2_token(), low, high, coefficient);
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_inverse_ssse3(v2_token(), low, high, coefficient);
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) struct NeonBackend<F>(PhantomData<fn() -> F>);

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
impl<F: TierButterflies> ButterflyBackend<F> for NeonBackend<F> {
    #[inline]
    fn forward_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_forward_neon(neon_token(), low, high, coefficient);
    }

    #[inline]
    fn inverse_nonzero(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
        F::fused_inverse_neon(neon_token(), low, high, coefficient);
    }
}

/// Dispatch a kernel-generic walker onto the process backend for field `F`.
///
/// Expands `$function::<F, B>(args…)` with `B` the selected backend tag, so
/// the entire walker monomorphizes onto one backend per call.
macro_rules! dispatch_butterfly {
    ($field:ty, $function:ident ($($argument:expr),* $(,)?)) => {
        match $crate::kernel::backend_for::<$field>() {
            $crate::kernel::Backend::Scalar => {
                $function::<$field, $crate::kernel::ScalarBackend<$field>>($($argument),*)
            }
            // Both `NeonAes` (AArch64 crypto, PMULL hosts) and `Neon` run the
            // same NEON butterflies; only the resolved tier label differs.
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            $crate::kernel::Backend::NeonAes | $crate::kernel::Backend::Neon => {
                $function::<$field, $crate::kernel::NeonBackend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            $crate::kernel::Backend::V3GfniCrypto => {
                $function::<$field, $crate::kernel::GfniBackend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            $crate::kernel::Backend::V3 => {
                $function::<$field, $crate::kernel::Avx2Backend<$field>>($($argument),*)
            }
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            $crate::kernel::Backend::V2 => {
                $function::<$field, $crate::kernel::Ssse3Backend<$field>>($($argument),*)
            }
            // backend_for never yields a backend outside the cfg'd set; the
            // wildcard only covers `Backend`'s non-exhaustiveness.
            #[allow(unreachable_patterns)]
            _ => $function::<$field, $crate::kernel::ScalarBackend<$field>>($($argument),*),
        }
    };
}

// Walkers in `transform` call this through the path re-export.
pub(crate) use dispatch_butterfly;

/// Fused forward butterfly on backend `B`, with fast paths for the trivial
/// coefficients: zero couples the halves with one XOR (`high ^= low`, `low`
/// untouched) and one swaps their roles through two XORs (`low' = low ⊕
/// high`, `high' = low`).
#[inline]
pub(crate) fn fused_forward_backend<F: ButterflyKernels, B: ButterflyBackend<F>>(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: F::Elem,
) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % F::BYTES, 0);
    if coefficient.is_zero() {
        fgf::ops::add_assign::<F>(high, low);
    } else if coefficient.is_one() {
        fgf::ops::add_assign::<F>(low, high);
        fgf::ops::add_assign::<F>(high, low);
    } else {
        B::forward_nonzero(low, high, coefficient);
    }
}

/// Fused inverse butterfly on backend `B`, with the same trivial-coefficient
/// fast paths: zero couples the halves with one XOR and one makes both
/// outputs equal to `low ⊕ high`.
#[inline]
pub(crate) fn fused_inverse_backend<F: ButterflyKernels, B: ButterflyBackend<F>>(
    low: &mut [u8],
    high: &mut [u8],
    coefficient: F::Elem,
) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % F::BYTES, 0);
    if coefficient.is_zero() {
        fgf::ops::add_assign::<F>(high, low);
    } else if coefficient.is_one() {
        fgf::ops::add_assign::<F>(high, low);
        fgf::ops::add_assign::<F>(low, high);
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
    crate::kernel::dispatch_butterfly!(F, fused_forward_backend(low, high, coefficient));
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
    crate::kernel::dispatch_butterfly!(F, fused_inverse_backend(low, high, coefficient));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::alloc::vec::Vec;
    use fgf::field::{Elem, Field};
    use fgf::{FanPaar64, Gf8B, Gf16, Gf32};

    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn direct_kernel_available<F: Field>(backend: Backend, available: bool) -> bool {
        use ::std::io::Write;

        // Write directly: libtest captures print macros on passing tests,
        // which would hide unsupported tiers during an ordinary sweep.
        writeln!(
            ::std::io::stderr().lock(),
            "INFO direct kernel check backend={} field={} status={} reason={}",
            backend.name(),
            ::core::any::type_name::<F>(),
            if available { "running" } else { "skipped" },
            if available {
                "host_supported"
            } else {
                "host_unsupported"
            },
        )
        .expect("write direct-kernel capability report");
        available
    }

    #[test]
    // `backend()` and `backend_for::<Gf8B/Gf16>` always land on a tier
    // the crate implements (or Scalar), never a backend without kernels.
    fn resolved_backend_is_in_supported_tiers() {
        assert!(BUTTERFLY_FFT_TIERS.contains(&backend()));
        assert!(Gf8B::BUTTERFLY_TIERS.contains(&backend_for::<Gf8B>()));
        assert!(Gf16::BUTTERFLY_TIERS.contains(&backend_for::<Gf16>()));
        // Wider fields have no butterfly kernels: always Scalar.
        assert_eq!(backend_for::<Gf32>(), Backend::Scalar);
        assert_eq!(backend_for::<FanPaar64>(), Backend::Scalar);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_gfni_resolution_implies_host_features() {
        // The summon() proof is simdispatch's; here we only sanity-check the
        // self-consistency a resolved GFNI tier implies the host really has
        // AVX2+GFNI (garble guard for the selection wiring).
        if backend() == Backend::V3GfniCrypto {
            assert!(X86_TOKENS.v3_gfni_crypto.is_some());
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
        let coefficients = [0x00, 0x01, 0x02, 0x03, 0x53, 0xff].map(fgf::gf8b::Elem::from_raw);
        scalar_roundtrip::<Gf8B>(&coefficients);
    }

    #[test]
    fn scalar_roundtrip_gf16() {
        let coefficients = [0x0000, 0x0001, 0x0108, 0x9b37, 0xffff].map(fgf::gf16::Elem::from_raw);
        scalar_roundtrip::<Gf16>(&coefficients);
    }

    #[test]
    fn scalar_forward_matches_element_math_gf8() {
        for coefficient in [0x00, 0x01, 0x03, 0xff].map(fgf::gf8b::Elem::from_raw) {
            for len in lengths(1) {
                let low = pattern(0x3d, len);
                let high = pattern(0xc2, len);
                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                for (l, h) in expected_low.iter_mut().zip(&mut expected_high) {
                    let lo = fgf::gf8b::Elem::from_raw(*l);
                    let hi = fgf::gf8b::Elem::from_raw(*h);
                    let new_low = lo.add(coefficient.mul(hi));
                    *l = new_low.to_raw();
                    *h = hi.add(new_low).to_raw();
                }
                let (mut low, mut high) = (low, high);
                scalar::fused_forward::<Gf8B>(&mut low, &mut high, coefficient);
                assert_eq!(low, expected_low);
                assert_eq!(high, expected_high);
            }
        }
    }

    #[test]
    fn scalar_forward_matches_element_math_gf16() {
        for coefficient in [0x0000, 0x0001, 0x0108, 0xbeef].map(fgf::gf16::Elem::from_raw) {
            for len in lengths(2) {
                let low = pattern(0x3d, len);
                let high = pattern(0xc2, len);
                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                for start in (0..expected_low.len()).step_by(2) {
                    let (l, h) = (
                        &mut expected_low[start..start + 2],
                        &mut expected_high[start..start + 2],
                    );
                    let lo = Gf16::decode(l);
                    let hi = Gf16::decode(h);
                    let new_low = lo.add(coefficient.mul(hi));
                    Gf16::encode(l, new_low);
                    Gf16::encode(h, hi.add(new_low));
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
        check::<Gf8B>(&[0x00, 0x01, 0x53, 0xff].map(fgf::gf8b::Elem::from_raw));
        check::<Gf16>(&[0x0000, 0x0001, 0x0108, 0x9b37].map(fgf::gf16::Elem::from_raw));
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
        check::<Gf8B>();
        check::<Gf16>();
    }

    /// The unit-coefficient fast paths must equal the general butterfly with
    /// `c = 1`: forward maps `(l, h)` to `(l⊕h, l)`, inverse to `(h, h⊕l)`,
    /// and the pair undoes itself.
    #[test]
    fn one_coefficient_fast_paths_match_xor_semantics() {
        fn check<F: ButterflyKernels>() {
            for len in lengths(F::BYTES) {
                let low = pattern(0x5a, len);
                let high = pattern(0xc3, len);
                let expected: Vec<u8> = low.iter().zip(&high).map(|(l, h)| l ^ h).collect();

                let mut forward = (low.clone(), high.clone());
                fused_forward::<F>(&mut forward.0, &mut forward.1, F::Elem::ONE);
                assert_eq!(forward.0, expected, "forward-one low");
                assert_eq!(forward.1, low, "forward-one high");

                let mut inverse = (low.clone(), high.clone());
                fused_inverse::<F>(&mut inverse.0, &mut inverse.1, F::Elem::ONE);
                assert_eq!(inverse.0, high, "inverse-one low");
                assert_eq!(inverse.1, expected, "inverse-one high");

                // The fast paths must still undo each other.
                fused_inverse::<F>(&mut forward.0, &mut forward.1, F::Elem::ONE);
                assert_eq!(forward.0, low);
                assert_eq!(forward.1, high);

                // And agree with the scalar reference at the same coefficient.
                let (mut reference_low, mut reference_high) = (low.clone(), high.clone());
                scalar::fused_forward::<F>(&mut reference_low, &mut reference_high, F::Elem::ONE);
                let (mut fast_low, mut fast_high) = (low.clone(), high.clone());
                fused_forward::<F>(&mut fast_low, &mut fast_high, F::Elem::ONE);
                assert_eq!(fast_low, reference_low);
                assert_eq!(fast_high, reference_high);
            }
        }
        check::<Gf8B>();
        check::<Gf16>();
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
                fused_forward_backend::<F, B>(&mut actual_low, &mut actual_high, coefficient);
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
                fused_inverse_backend::<F, B>(&mut actual_low, &mut actual_high, coefficient);
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

    /// The trivial coefficients must also reach the SIMD kernels past the
    /// public fast paths: `fused_forward_backend` intercepts zero and one
    /// before the backend runs, so the ordinary differential loop never
    /// exercises the kernels with them.
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn differential_backend_trivial_coefficients<F: ButterflyKernels, B: ButterflyBackend<F>>(
        label: &str,
    ) {
        for coefficient in [F::Elem::ZERO, F::Elem::ONE] {
            for len in lengths(F::BYTES) {
                let low = pattern(0x2e, len);
                let high = pattern(0xd1, len);

                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                scalar::fused_forward::<F>(&mut expected_low, &mut expected_high, coefficient);
                let mut actual_low = low;
                let mut actual_high = high;
                B::forward_nonzero(&mut actual_low, &mut actual_high, coefficient);
                assert_eq!(
                    actual_low, expected_low,
                    "{label} forward-zero/one low diverged at len {len}"
                );
                assert_eq!(
                    actual_high, expected_high,
                    "{label} forward-zero/one high diverged at len {len}"
                );

                let mut low = pattern(0x2e, len);
                let mut high = pattern(0xd1, len);
                let mut expected_low = low.clone();
                let mut expected_high = high.clone();
                scalar::fused_inverse::<F>(&mut expected_low, &mut expected_high, coefficient);
                B::inverse_nonzero(&mut low, &mut high, coefficient);
                assert_eq!(
                    low, expected_low,
                    "{label} inverse-zero/one low at len {len}"
                );
                assert_eq!(
                    high, expected_high,
                    "{label} inverse-zero/one high at len {len}"
                );
            }
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    fn differential_x86<F: ButterflyKernels>(coefficients: &[F::Elem]) {
        if direct_kernel_available::<F>(Backend::V3GfniCrypto, X86_TOKENS.v3_gfni_crypto.is_some())
        {
            differential_backend::<F, GfniBackend<F>>("gfni", coefficients);
            differential_backend_trivial_coefficients::<F, GfniBackend<F>>("gfni");
        }
        if direct_kernel_available::<F>(Backend::V3, X86_TOKENS.v3.is_some()) {
            differential_backend::<F, Avx2Backend<F>>("avx2", coefficients);
            differential_backend_trivial_coefficients::<F, Avx2Backend<F>>("avx2");
        }
        if direct_kernel_available::<F>(Backend::V2, X86_TOKENS.v2.is_some()) {
            differential_backend::<F, Ssse3Backend<F>>("ssse3", coefficients);
            differential_backend_trivial_coefficients::<F, Ssse3Backend<F>>("ssse3");
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_backends_match_scalar_in_both_directions() {
        differential_x86::<Gf8B>(&[0x00, 0x01, 0x02, 0x53, 0xff].map(fgf::gf8b::Elem::from_raw));
        differential_x86::<Gf16>(
            &[0x0000, 0x0001, 0x0108, 0x9b37, 0xffff].map(fgf::gf16::Elem::from_raw),
        );
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn neon_backends_match_scalar_in_both_directions() {
        if !direct_kernel_available::<Gf8B>(Backend::Neon, NEON_TOKEN.is_some()) {
            return;
        }
        differential_backend::<Gf8B, NeonBackend<Gf8B>>(
            "neon",
            &[0x00, 0x01, 0x02, 0x53, 0xff].map(fgf::gf8b::Elem::from_raw),
        );
        differential_backend_trivial_coefficients::<Gf8B, NeonBackend<Gf8B>>("neon");
        direct_kernel_available::<Gf16>(Backend::Neon, true);
        differential_backend::<Gf16, NeonBackend<Gf16>>(
            "neon",
            &[0x0000, 0x0001, 0x0108, 0x9b37, 0xffff].map(fgf::gf16::Elem::from_raw),
        );
        differential_backend_trivial_coefficients::<Gf16, NeonBackend<Gf16>>("neon");
    }

    /// The tier entries of a field without dedicated butterflies inherit the
    /// portable scalar kernels: every default body must agree with `scalar`
    /// in both directions. Cached capability tokens make unsupported entries
    /// skip explicitly rather than panic, independently of `SIMD_BACKEND`.
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn kernelless_field_tier_entries_inherit_scalar() {
        fn entries_match_scalar<F: TierButterflies>(
            label: &str,
            coefficient: F::Elem,
            mut forward: impl FnMut(&mut [u8], &mut [u8], F::Elem),
            mut inverse: impl FnMut(&mut [u8], &mut [u8], F::Elem),
        ) {
            for len in [0usize, 1, 15, 16, 33] {
                let mut low = pattern(0x91, len * F::BYTES);
                let mut high = pattern(0x5c, len * F::BYTES);
                let (mut want_low, mut want_high) = (low.clone(), high.clone());
                scalar::fused_forward::<F>(&mut want_low, &mut want_high, coefficient);
                forward(&mut low, &mut high, coefficient);
                assert_eq!(low, want_low, "{label} forward low at len {len}");
                assert_eq!(high, want_high, "{label} forward high at len {len}");

                let mut low = pattern(0x91, len * F::BYTES);
                let mut high = pattern(0x5c, len * F::BYTES);
                let (mut want_low, mut want_high) = (low.clone(), high.clone());
                scalar::fused_inverse::<F>(&mut want_low, &mut want_high, coefficient);
                inverse(&mut low, &mut high, coefficient);
                assert_eq!(low, want_low, "{label} inverse low at len {len}");
                assert_eq!(high, want_high, "{label} inverse high at len {len}");
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if direct_kernel_available::<Gf32>(
                Backend::V3GfniCrypto,
                X86_TOKENS.v3_gfni_crypto.is_some(),
            ) {
                let token = v3_gfni_crypto_token();
                entries_match_scalar::<Gf32>(
                    "gfni defaults",
                    <Gf32 as Field>::Elem::ONE,
                    |low, high, c| Gf32::fused_forward_gfni(token, low, high, c),
                    |low, high, c| Gf32::fused_inverse_gfni(token, low, high, c),
                );
            }
            if direct_kernel_available::<Gf32>(Backend::V3, X86_TOKENS.v3.is_some()) {
                let token = v3_token();
                entries_match_scalar::<Gf32>(
                    "avx2 defaults",
                    <Gf32 as Field>::Elem::ZERO,
                    |low, high, c| Gf32::fused_forward_avx2(token, low, high, c),
                    |low, high, c| Gf32::fused_inverse_avx2(token, low, high, c),
                );
            }
            if direct_kernel_available::<Gf32>(Backend::V2, X86_TOKENS.v2.is_some()) {
                let token = v2_token();
                entries_match_scalar::<Gf32>(
                    "ssse3 defaults",
                    <Gf32 as Field>::Elem::ONE,
                    |low, high, c| Gf32::fused_forward_ssse3(token, low, high, c),
                    |low, high, c| Gf32::fused_inverse_ssse3(token, low, high, c),
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if direct_kernel_available::<Gf32>(Backend::Neon, NEON_TOKEN.is_some()) {
                let token = neon_token();
                entries_match_scalar::<Gf32>(
                    "neon defaults",
                    <Gf32 as Field>::Elem::ONE,
                    |low, high, c| Gf32::fused_forward_neon(token, low, high, c),
                    |low, high, c| Gf32::fused_inverse_neon(token, low, high, c),
                );
            }
        }
    }
}
