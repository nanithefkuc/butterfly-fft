//! The engine: twiddle tables, butterfly kernels, execution models.
//!
//! Generic over [`fgf::field::Field`]. GF(2^8) and GF(2^16) carry dedicated
//! SIMD butterfly backends; wider fields use the portable scalar backend.
//!
//! - `factors` — normalized subspace polynomials and the per-node twiddle
//!   and derivative tables (unstable inspection under feature `internals`).
//! - [`kernel`] — fused one-pass butterflies over byte rows, dispatched once
//!   per process to the best backend the host supports.
//! - [`transform`] — the reusable plan and the recursive in-place execution
//!   models (full, selected, range, truncated, high-coset, derivative).

#[cfg(feature = "internals")]
pub mod factors;
#[cfg(not(feature = "internals"))]
pub(crate) mod factors;
pub mod kernel;
pub mod transform;
