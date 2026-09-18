#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]
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
// path in this crate.
extern crate alloc;
mod tuning;

pub mod basis;
pub mod error;
pub mod kernel;
pub mod ntt;
pub mod transform;

#[cfg(feature = "internals")]
pub mod internals;

pub use error::{NttError, PlanError, TransformError};
pub use ntt::NttPlan;
pub use transform::TransformPlan;
