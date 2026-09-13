//! Multiplicative (number-theoretic) transforms.
//!
//! The rest of this crate evaluates over *additive* subspaces of a binary
//! field. This module is the second transform family over the same
//! mathematical object: a radix-two Cooley–Tukey transform over the
//! *multiplicative* group of a field that contains a primitive `size`-th
//! root of unity. Field arithmetic still comes from [`fgf`]; nothing here
//! re-implements a field loop, and SIMD selection stays where it already
//! lives.
//!
//! ## Convention
//!
//! For a plan of size `n` with root `ω` ([`NttPlan::root`]), the forward
//! transform maps input `x` to
//!
//! ```text
//! Y[k] = Σ_{j=0}^{n-1} x[j] · ω^(j·k),   k = 0 … n-1
//! ```
//!
//! Input is consumed in natural order and output is produced in **natural
//! order**: the implementation applies the bit-reversal permutation first
//! and then runs decimation-in-time butterflies. The inverse runs the same
//! loop with `ω^-1` and finishes by scaling every lane by `n^-1`, so
//! `inverse(forward(x)) == x` exactly. `size == 1` is the identity.
//!
//! ## Field reach
//!
//! A plan exists exactly when `size` divides the order of the
//! multiplicative group, `|F| - 1`, and `size <= 1 << MAX_LOG_SIZE`:
//!
//! - **Goldilocks** (`p = 2^64 - 2^32 + 1`): `p - 1 = 2^32·(2^32 - 1)`, so
//!   `2^32 | p - 1` and every size up to `2^20` is available.
//! - **`QuadMersenne31`** (`|F| = p²`, `p = 2^31 - 1`): `p² - 1 =
//!   (p-1)(p+1)` and `p + 1 = 2^31`, so `2^32 | p² - 1` and every size up
//!   to `2^20` is available.
//! - **Base `Mersenne31`**: `p - 1 = 2·(2^30 - 1)` with an odd cofactor, so
//!   the only nontrivial radix-two size is `2`. That is why convolution
//!   over `Mersenne31` embeds into `QuadMersenne31` instead of using this
//!   field's own multiplicative group.
//! - **Binary fields** (`|F| = 2^m`): `|F| - 1` is odd, so no even size
//!   divides it and only `size == 1`, the identity, is accepted. There is
//!   no special case for them in the code; the divisibility rule already
//!   says so.
//!
//! ## Buffers
//!
//! Execution is byte-oriented: `rows` holds `size` contiguous rows of
//! `row_len` bytes, and a row is a vector of independent lanes that all
//! undergo the same transform. One lane per row is the scalar case. The
//! plan owns every twiddle and the `n^-1` scale, prepared once at
//! construction, so [`NttPlan::forward_bytes`] and
//! [`NttPlan::inverse_bytes`] allocate nothing and prepare nothing.
//!
//! Consumers own padding and truncation. A plan transforms exactly `size`
//! points; zero-extending a shorter input and cutting the result back down
//! — including choosing a transform length large enough that a cyclic
//! convolution does not wrap — is the caller's responsibility.

use ::alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::ops::{self, Coeff};

use crate::core::transform::MAX_LOG_SIZE;

/// Error returned by multiplicative-transform plan construction and
/// execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttError {
    /// Transform size was zero or not a power of two.
    InvalidSize {
        /// The offending size.
        size: usize,
    },
    /// The field has no primitive `size`-th root of unity, or `size`
    /// exceeds the table-size cap `MAX_LOG_SIZE`. A power-of-two `size`
    /// above one needs `size` to divide `field_order - 1`, which no binary
    /// field satisfies.
    UnsupportedSize {
        /// The requested size.
        size: usize,
        /// Number of elements in the field.
        field_order: u128,
    },
    /// A size, offset, or byte length did not fit in `usize`/`u64`.
    GeometryOverflow,
    /// A table or scratch allocation failed.
    AllocationFailed,
    /// The row buffer length does not match `size * row_len`.
    BufferLength {
        /// Length the plan requires, in bytes.
        expected: usize,
        /// Length the caller supplied, in bytes.
        actual: usize,
    },
    /// A row length is not a whole number of field elements, or the
    /// supplied scratch was built for a field of a different element width.
    InvalidRowLength {
        /// The offending row length, in bytes.
        row_len: usize,
        /// Width of one field element, in bytes.
        element_bytes: usize,
    },
    /// The supplied scratch row temporary is shorter than the requested
    /// row length.
    ScratchTooSmall {
        /// Row bytes required.
        required: usize,
        /// Row bytes the scratch was built for.
        available: usize,
    },
}

impl ::core::fmt::Display for NttError {
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            Self::InvalidSize { size } => {
                write!(
                    formatter,
                    "invalid transform size {size}: not a power of two"
                )
            }
            Self::UnsupportedSize { size, field_order } => {
                write!(
                    formatter,
                    "no primitive root of unity of order {size} in a field of order {field_order}"
                )
            }
            Self::GeometryOverflow => {
                write!(formatter, "transform geometry overflowed")
            }
            Self::AllocationFailed => {
                write!(formatter, "transform table allocation failed")
            }
            Self::BufferLength { expected, actual } => {
                write!(
                    formatter,
                    "wrong row buffer length: expected {expected} bytes, got {actual}"
                )
            }
            Self::InvalidRowLength {
                row_len,
                element_bytes,
            } => {
                write!(
                    formatter,
                    "row length {row_len} is not a whole number of {element_bytes}-byte elements"
                )
            }
            Self::ScratchTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "scratch too small: need {required} row bytes, have {available}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for NttError {}

/// Reusable row temporary for [`NttPlan::forward_bytes`] and
/// [`NttPlan::inverse_bytes`].
///
/// Holds exactly one row and remembers the element width and row length it
/// was built for, so reuse against a different geometry is rejected instead
/// of silently transforming the wrong shape. Not generic: the recorded
/// element width is what ties it to a field.
#[derive(Clone, Debug)]
pub struct NttScratch {
    /// One row of butterfly temporaries, `row_len` bytes.
    row: Vec<u8>,
    /// Element width of the field this scratch was built for.
    element_bytes: usize,
    /// Row length this scratch was built for, in bytes.
    row_len: usize,
}

/// Reusable radix-two multiplicative transform over a power-of-two domain.
///
/// Construction resolves the root of unity, the bit-reversal permutation,
/// every stage's twiddle powers in both directions, and the `n^-1` inverse
/// scale — all as prepared [`Coeff`] values. Execution is then pure
/// butterflies: no allocation, no coefficient preparation.
///
/// See the [module documentation][self] for the transform convention and
/// the fields that admit a domain.
#[derive(Clone, Debug)]
pub struct NttPlan<F: FieldKernels> {
    /// Number of transform points.
    size: usize,
    /// `log2(size)`; also the number of butterfly stages.
    log_size: usize,
    /// Primitive `size`-th root of unity used by the forward transform.
    root: F::Elem,
    /// `permutation[i]` is `i` with its `log_size` low bits reversed.
    permutation: Vec<usize>,
    /// Start of each stage's twiddle run inside the flat tables.
    stage_offsets: Vec<usize>,
    /// Flat `size - 1` powers of `root`, stage by stage.
    forward_twiddles: Vec<Coeff<F>>,
    /// The same table built from `root^-1`.
    inverse_twiddles: Vec<Coeff<F>>,
    /// Prepared `size^-1`, applied by the inverse transform.
    inverse_scale: Coeff<F>,
}

/// Embed the integer `value` into `F` by double-and-add of `ONE`.
///
/// A raw byte encoding is not an integer embedding in an extension field,
/// so the characteristic-agnostic route is the only correct one.
fn embed<F: Field>(value: u64) -> F::Elem {
    let mut remaining = value;
    let mut term = F::Elem::ONE;
    let mut total = F::Elem::ZERO;
    while remaining != 0 {
        if remaining & 1 != 0 {
            total = total.add(term);
        }
        term = term.add(term);
        remaining >>= 1;
    }
    total
}

/// Reserve `additional` slots, mapping failure onto [`NttError`].
fn reserve<T>(target: &mut Vec<T>, additional: usize) -> Result<(), NttError> {
    target
        .try_reserve_exact(additional)
        .map_err(|_| NttError::AllocationFailed)
}

impl<F: FieldKernels> NttPlan<F> {
    /// Construct a plan for a power-of-two `size` over `F`.
    ///
    /// The root is `GENERATOR^((|F| - 1) / size)`, verified to have order
    /// exactly `size` before any table is built.
    ///
    /// # Errors
    /// [`NttError::InvalidSize`] for a zero or non-power-of-two size;
    /// [`NttError::UnsupportedSize`] when `size` exceeds the table cap or
    /// the field has no primitive `size`-th root of unity;
    /// [`NttError::GeometryOverflow`] for a size or table offset that does
    /// not fit; [`NttError::AllocationFailed`] if a table cannot be
    /// allocated.
    pub fn new(size: usize) -> Result<Self, NttError> {
        if size == 0 || !size.is_power_of_two() {
            return Err(NttError::InvalidSize { size });
        }
        let log_size = size.trailing_zeros() as usize;
        let unsupported = NttError::UnsupportedSize {
            size,
            field_order: F::ORDER,
        };
        if log_size > MAX_LOG_SIZE {
            return Err(unsupported);
        }
        let size_u64 = u64::try_from(size).map_err(|_| NttError::GeometryOverflow)?;
        let group_order = F::ORDER - 1;
        if group_order % u128::from(size_u64) != 0 {
            return Err(unsupported);
        }
        let exponent = u64::try_from(group_order / u128::from(size_u64))
            .map_err(|_| NttError::GeometryOverflow)?;
        let root = F::GENERATOR.pow(exponent);
        if root.pow(size_u64) != F::Elem::ONE {
            return Err(unsupported);
        }
        if size > 1 && root.pow(size_u64 / 2) == F::Elem::ONE {
            return Err(unsupported);
        }

        let permutation = bit_reversal(size, log_size)?;
        let stage_offsets = stage_offsets(log_size)?;
        let twiddle_count = size - 1;
        let forward_twiddles = twiddle_table::<F>(size, log_size, root, twiddle_count)?;
        let inverse_twiddles = twiddle_table::<F>(size, log_size, root.inv(), twiddle_count)?;
        let inverse_scale = Coeff::new(embed::<F>(size_u64).inv());

        Ok(Self {
            size,
            log_size,
            root,
            permutation,
            stage_offsets,
            forward_twiddles,
            inverse_twiddles,
            inverse_scale,
        })
    }

    /// Number of transform points.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The primitive `size`-th root of unity this plan transforms with.
    ///
    /// Output index `k` is `Σ_j x[j] · root^(j·k)`.
    #[inline]
    #[must_use]
    pub fn root(&self) -> F::Elem {
        self.root
    }

    /// Allocate the row temporary for rows of `row_len` bytes.
    ///
    /// The result is reusable for any execution whose row length is at most
    /// `row_len` over a field of the same element width.
    ///
    /// # Errors
    /// [`NttError::InvalidRowLength`] if `row_len` is not a whole number of
    /// elements; [`NttError::AllocationFailed`] if the row cannot be
    /// allocated.
    pub fn scratch(&self, row_len: usize) -> Result<NttScratch, NttError> {
        if !row_len.is_multiple_of(F::BYTES) {
            return Err(NttError::InvalidRowLength {
                row_len,
                element_bytes: F::BYTES,
            });
        }
        let mut row = Vec::new();
        reserve(&mut row, row_len)?;
        row.resize(row_len, 0);
        Ok(NttScratch {
            row,
            element_bytes: F::BYTES,
            row_len,
        })
    }

    /// Forward transform, in place, over `size` rows of `row_len` bytes.
    ///
    /// Lanes within a row are independent; every lane sees the same
    /// transform. Input is natural order, output is natural order. Nothing
    /// is written unless every structural check below passes.
    ///
    /// # Errors
    /// [`NttError::InvalidRowLength`], [`NttError::BufferLength`],
    /// [`NttError::ScratchTooSmall`], or [`NttError::GeometryOverflow`].
    pub fn forward_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        if row_len == 0 {
            return Ok(());
        }
        canonicalize::<F>(rows);
        self.butterflies(rows, row_len, &self.forward_twiddles, scratch);
        Ok(())
    }

    /// Inverse transform, in place, over `size` rows of `row_len` bytes.
    ///
    /// Runs the forward loop with `root^-1` and then scales every lane by
    /// `size^-1`, so it inverts [`NttPlan::forward_bytes`] exactly.
    ///
    /// # Errors
    /// As [`NttPlan::forward_bytes`].
    pub fn inverse_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        if row_len == 0 {
            return Ok(());
        }
        canonicalize::<F>(rows);
        self.butterflies(rows, row_len, &self.inverse_twiddles, scratch);
        ops::mul_assign_with(rows, &self.inverse_scale);
        Ok(())
    }

    /// Check every dimension before any mutation happens.
    fn validate(
        &self,
        buffer_bytes: usize,
        row_len: usize,
        scratch: &NttScratch,
    ) -> Result<(), NttError> {
        if !row_len.is_multiple_of(F::BYTES) {
            return Err(NttError::InvalidRowLength {
                row_len,
                element_bytes: F::BYTES,
            });
        }
        let expected = self
            .size
            .checked_mul(row_len)
            .ok_or(NttError::GeometryOverflow)?;
        if buffer_bytes != expected {
            return Err(NttError::BufferLength {
                expected,
                actual: buffer_bytes,
            });
        }
        if scratch.element_bytes != F::BYTES {
            return Err(NttError::InvalidRowLength {
                row_len,
                element_bytes: F::BYTES,
            });
        }
        if scratch.row_len < row_len {
            return Err(NttError::ScratchTooSmall {
                required: row_len,
                available: scratch.row_len,
            });
        }
        Ok(())
    }

    /// Bit-reversal permutation followed by `log_size` butterfly stages.
    ///
    /// `row_len` is nonzero and a whole number of elements, `rows` is
    /// `size * row_len` bytes, and `scratch` holds at least one row: all
    /// established by [`NttPlan::validate`].
    fn butterflies(
        &self,
        rows: &mut [u8],
        row_len: usize,
        twiddles: &[Coeff<F>],
        scratch: &mut NttScratch,
    ) {
        for (index, &target) in self.permutation.iter().enumerate() {
            if index < target {
                let (head, tail) = rows.split_at_mut(target * row_len);
                head[index * row_len..(index + 1) * row_len].swap_with_slice(&mut tail[..row_len]);
            }
        }

        let temp = &mut scratch.row[..row_len];
        for stage in 0..self.log_size {
            let half = 1usize << stage;
            let base = self.stage_offsets[stage];
            let half_bytes = half * row_len;
            for block in rows.chunks_exact_mut(half_bytes * 2) {
                let (lows, highs) = block.split_at_mut(half_bytes);
                let pairs = lows
                    .chunks_exact_mut(row_len)
                    .zip(highs.chunks_exact_mut(row_len));
                for (offset, (low, high)) in pairs.enumerate() {
                    let twiddle = &twiddles[base + offset];
                    ops::mul_into_with(&mut *temp, twiddle, high);
                    high.copy_from_slice(low);
                    ops::sub_assign::<F>(high, temp);
                    ops::add_assign::<F>(low, temp);
                }
            }
        }
    }
}

/// Bring every lane into canonical form once, before any packed kernel
/// runs: fgf's prime-field kernels are defined on canonical lanes, while
/// `pack` and `from_raw` preserve whatever representative the caller had.
/// Characteristic two has a unique representative per element, so the pass
/// is skipped there entirely.
// `as_chunks_mut::<F::BYTES>()` would need a const generic argument that
// depends on `F`, which stable Rust does not accept.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn canonicalize<F: FieldKernels>(rows: &mut [u8]) {
    if F::CHARACTERISTIC == 2 {
        return;
    }
    for slot in rows.chunks_exact_mut(F::BYTES) {
        F::write(slot, F::read(slot).add(F::Elem::ZERO));
    }
}

/// `permutation[i]` is `i` with its low `log_size` bits reversed, built by
/// incrementing a reversed counter so no width-dependent shift is needed.
fn bit_reversal(size: usize, log_size: usize) -> Result<Vec<usize>, NttError> {
    let mut permutation = Vec::new();
    reserve(&mut permutation, size)?;
    debug_assert_eq!(size, 1usize << log_size);
    let mut reversed = 0usize;
    for _ in 0..size {
        permutation.push(reversed);
        let mut bit = size >> 1;
        while bit != 0 && reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed |= bit;
    }
    Ok(permutation)
}

/// Start of each stage's run in the flat twiddle table: stage `s` holds
/// `2^s` entries, so it begins at `2^s - 1`.
fn stage_offsets(log_size: usize) -> Result<Vec<usize>, NttError> {
    let mut offsets = Vec::new();
    reserve(&mut offsets, log_size)?;
    let mut offset = 0usize;
    for stage in 0..log_size {
        offsets.push(offset);
        let count = 1usize
            .checked_shl(u32::try_from(stage).map_err(|_| NttError::GeometryOverflow)?)
            .ok_or(NttError::GeometryOverflow)?;
        offset = offset
            .checked_add(count)
            .ok_or(NttError::GeometryOverflow)?;
    }
    Ok(offsets)
}

/// Flat per-stage powers of `root`, prepared for the host backend.
///
/// Stage `s` (transform length `2^(s+1)`) stores `w^0 … w^(2^s - 1)` for
/// `w = root^(size / 2^(s+1))`, so the whole table is `size - 1` entries.
fn twiddle_table<F: FieldKernels>(
    size: usize,
    log_size: usize,
    root: F::Elem,
    capacity: usize,
) -> Result<Vec<Coeff<F>>, NttError> {
    let mut table = Vec::new();
    reserve(&mut table, capacity)?;
    for stage in 0..log_size {
        let half = 1usize << stage;
        let step =
            root.pow(u64::try_from(size / (half * 2)).map_err(|_| NttError::GeometryOverflow)?);
        let mut power = F::Elem::ONE;
        for _ in 0..half {
            table.push(Coeff::new(power));
            power = power.mul(step);
        }
    }
    Ok(table)
}
