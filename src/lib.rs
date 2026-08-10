//! # Common Additive Fast Fourier Transform
//!
//! Shared additive-FFT engine over binary fields: subspace twiddle tables,
//! SIMD-batched butterfly kernels, and in-place transform execution models,
//! plus the extended basis/shifted/truncated APIs its codec consumers need.
//!
//! Field arithmetic and byte-buffer vector primitives come from [`fgf`];
//! this crate never re-implements field arithmetic. Consumers own wire
//! formats, codec shells, and evaluation-point ↔ wire-index maps.
//!
//! ## Layout
//!
//! - [`core`] — the engine: [`core::factors`] subspace twiddle/derivative
//!   tables, [`core::kernel`] fused butterfly kernels with runtime SIMD
//!   dispatch, [`core::transform`] plans and in-place execution models.
//! - [`basis`] — ordered field bases (bit, Cantor) and monomial ↔ novel
//!   coefficient-basis conversion.
//! - [`shifted`] — transforms over affine cosets `α + V`.
//! - [`rs`] — RS-facing erasure algebra (feature `rs`).
//!
//! ## Features
//!
//! - `std` (default) — runtime CPU detection and shared plan caches.
//! - `simd` (default, implies `std`) — vector butterfly backends.
//! - `rs` — RS erasure helpers.
//! - `internals` — unstable APIs, exempt from compatibility guarantees.
//!
//! ## Naming note
//!
//! This crate has a module named [`core`]. Inside the crate, sysroot paths
//! must be written absolutely (`::core::…`, `::std::…`); a relative
//! `core::…` resolves to the local module.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // Arch intrinsics are imported wholesale by universal convention; naming
    // each one would be unmaintainable and duplicated across cfg arms.
    clippy::wildcard_imports,
    clippy::inline_always,
    clippy::module_name_repetitions
)]

// Tables and plans allocate; the `std`-less configuration still needs `Vec`
// and `Arc`. Written `::alloc::…` at use sites, like every other sysroot
// path in this crate (see the naming note above).
extern crate alloc;

pub mod basis;
pub mod core;
pub mod error;
#[cfg(feature = "internals")]
pub mod internals;
#[cfg(feature = "rs")]
pub mod rs;
pub mod shifted;
