//! Multiplicative (number-theoretic) transforms.
//!
//! Radix-two Cooley–Tukey transforms over the multiplicative group of a
//! field that contains a primitive `size`-th root of unity — the second
//! transform family over the same field objects, beside the additive
//! subspaces the rest of this crate evaluates over. Field arithmetic
//! comes from [`fgf`]; nothing here re-implements a field loop.
//!
//! For a plan of size `n` with root `ω` ([`NttPlan::root`]), the forward
//! transform maps input `x` to
//!
//! ```text
//! Y[k] = Σ_{j=0}^{n-1} x[j] · ω^(j·k),   k = 0 … n-1
//! ```
//!
//! Input is consumed in natural order and output is produced in natural
//! order: the bit-reversal permutation runs first, then
//! decimation-in-time butterflies. The inverse runs the same loop with
//! `ω^-1` and finishes by scaling every lane by `n^-1`. `size == 1` is
//! the identity.
//!
//! A plan exists exactly when `size` divides the order of the
//! multiplicative group, `|F| - 1`, and `size <= 1 << MAX_LOG_SIZE`.
//! Goldilocks (`2^32 | p - 1`) and `QuadMersenne31` (`2^32 | p² - 1`)
//! admit every size up to the cap. Base `Mersenne31`'s group has a single
//! factor of two, so size `2` is its largest radix-two transform — which
//! is why convolution over `Mersenne31` embeds into `QuadMersenne31`.
//! The binary fields have odd group order, so only the identity
//! `size == 1` is accepted; the divisibility rule says so with no special
//! case in the code.
//!
//! # Layout
//!
//! `rows` holds `size` contiguous rows of `row_len` bytes, and a row is a
//! vector of independent lanes that all undergo the same transform; one
//! lane per row is the scalar case. The plan owns every twiddle and the
//! `n^-1` scale, prepared once at construction, so
//! [`NttPlan::forward_bytes_scratch`] and [`NttPlan::inverse_bytes_scratch`]
//! allocate nothing and prepare nothing. Consumers own padding and
//! truncation: a plan transforms exactly `size` points, and choosing a
//! transform length large enough that a cyclic convolution does not wrap
//! is the caller's responsibility.
//!
//! # Totality and canonicalization
//!
//! Execution over the prime fields canonicalizes every input lane before
//! the first butterfly, so different raw encodings of one field value
//! transform to the same result; characteristic two has a unique
//! representative per value and skips the pass entirely. Identities hold
//! in field values: `inverse(forward(x))` equals `x` in every lane, and
//! the encoded bytes return unchanged whenever the input lanes were
//! canonical.

use ::alloc::boxed::Box;
use ::alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::ops::{self, Coeff};

pub use crate::error::NttError;

use crate::transform::MAX_LOG_SIZE;

/// Row length in bytes below which butterfly stages over fields with vector
/// elementwise kernels run in the batched region form; at and above it they
/// run the packed op-call form, whose broadcast multiply reads no twiddle
/// stream and wins on wide rows. A pure function of the row geometry and
/// the resolved backend, set by the interleaved campaign behind the
/// butterfly-schedule record in `BENCHMARKS.md`.
const BATCHED_ROW_MAX: usize = 128;

/// Region bytes — lane-tiled twiddles and their product with the high
/// rows — that one batch of the batched schedule may occupy; a wider
/// single row is a batch on its own.
const BATCH_BYTES: usize = 64 * 1024;

/// Logarithm of the smallest transform the four-step tuning schedule
/// decomposes; smaller plans carry no decomposition and the four-step entry
/// runs the production schedule over them. A pure placeholder until the
/// campaign behind the four-step record in `BENCHMARKS.md` measures the
/// crossover against the one-buffer schedules.
const FOURSTEP_MIN_LOG: usize = 10;

/// Reusable row temporary for [`NttPlan::forward_bytes_scratch`] and
/// [`NttPlan::inverse_bytes_scratch`].
///
/// Holds one row, the two region buffers the batched tuning schedule
/// batches into — lane-tiled twiddles and their product with the high
/// rows, reused by the four-step tuning schedule for its lane-tiled
/// twiddle products — the transpose visited flags the four-step schedule
/// walks its cycles with, and records the element width and row capacity
/// it was built for. Each region buffer covers the widest batch any
/// execution over this plan at a row length up to the built one can run,
/// and never exceeds `max(BATCH_BYTES, row_len)` bytes; the flag buffer
/// carries one bit per transform point of the building plan. Reuse over
/// the same field at any row length up to the built capacity is accepted;
/// a different element width or a longer row is rejected by the execution
/// methods, the batched schedule additionally rejects batch regions
/// smaller than the executing plan fills, and the four-step schedule
/// additionally rejects transpose flags smaller than the executing plan
/// fills. Not generic: the recorded element width is what ties it to a
/// field.
#[derive(Clone, Debug)]
pub struct NttScratch {
    /// One row of butterfly temporaries, `row_len` bytes.
    row: Vec<u8>,
    /// Region product `twiddle ⊙ high` over one batch of the batched
    /// schedule.
    batch_temp: Vec<u8>,
    /// Lane-tiled twiddles over one batch of the batched schedule.
    batch_twiddles: Vec<u8>,
    /// Visited flags for the four-step schedule's in-place transposes, one
    /// bit per transform point of the building plan; every transpose
    /// clears the range it walks before following any cycle.
    transpose_bits: Vec<u8>,
    /// Element width of the field this scratch was built for.
    element_bytes: usize,
    /// Row length this scratch was built for, in bytes.
    row_len: usize,
}

/// Reusable radix-two multiplicative transform over a power-of-two domain.
///
/// Construction resolves the root of unity, the bit-reversal permutation,
/// every stage's twiddle powers in both directions, and the `n^-1` inverse
/// butterflies: no allocation, no coefficient preparation. Plans at or
/// above the four-step activation size also carry the sub-plans of the
/// cache-blocked decomposition.
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
    /// Cache-blocked decomposition of this plan, carried only at or above
    /// the four-step activation size.
    fourstep: Option<Box<FourStepSchedule<F>>>,
}

/// One plan's cache-blocked decomposition: the transform factored over a
/// `second.size()`-by-`first.size()` slot matrix, with the twiddle pass
/// between the two phases and a transpose before, between, and after them.
///
/// The sub-plan roots are the powers of the parent root the decomposition
/// needs — `root^(first.size())` and `root^(second.size())` — because
/// construction derives every root from the same generator exponent, so
/// the phases compose to the parent transform exactly. Sub-plans never
/// decompose: the schedule stays one level deep, and each phase is a
/// plain butterfly walk.
#[derive(Clone, Debug)]
struct FourStepSchedule<F: FieldKernels> {
    /// Transform for the phase along the short axis of the slot matrix.
    first: Box<NttPlan<F>>,
    /// Transform for the phase along the long axis.
    second: Box<NttPlan<F>>,
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
        Self::build(size, FOURSTEP_MIN_LOG)
    }

    /// Shared constructor; `fourstep_from` is the smallest log size that
    /// builds a decomposition, and `usize::MAX` never builds one, which is
    /// how the sub-plans stay shallow.
    fn build(size: usize, fourstep_from: usize) -> Result<Self, NttError> {
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
        let fourstep = if log_size >= fourstep_from {
            let short = 1usize << (log_size / 2);
            let long = 1usize << log_size.div_ceil(2);
            Some(Box::new(FourStepSchedule {
                first: Box::new(Self::build(short, usize::MAX)?),
                second: Box::new(Self::build(long, usize::MAX)?),
            }))
        } else {
            None
        };

        Ok(Self {
            size,
            log_size,
            root,
            permutation,
            stage_offsets,
            forward_twiddles,
            inverse_twiddles,
            inverse_scale,
            fourstep,
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

    /// Allocate the row temporary, the batched-schedule region buffers,
    /// and the four-step transpose flags for rows of `row_len` bytes.
    ///
    /// The result is reusable for any execution whose row length is at most
    /// `row_len` over a field of the same element width. The region buffers
    /// are sized for this plan's widest batch, and the transpose flags for
    /// this plan's size, so the batched and four-step schedules reject them
    /// when a larger plan executes.
    ///
    /// # Errors
    /// [`NttError::InvalidRowLength`] if `row_len` is zero or not a whole
    /// number of elements; [`NttError::AllocationFailed`] if any buffer
    /// cannot be allocated.
    pub fn scratch(&self, row_len: usize) -> Result<NttScratch, NttError> {
        if row_len == 0 || !row_len.is_multiple_of(F::BYTES) {
            return Err(NttError::InvalidRowLength {
                row_len,
                element_bytes: F::BYTES,
            });
        }
        // `batch_rows` is a sawtooth in `row_len`, so a scratch built for
        // one row length must cover the widest batch every shorter row
        // length over this plan can ask for: the whole widest half capped
        // by one batch.
        let widest_half = (self.size >> 1)
            .checked_mul(row_len)
            .ok_or(NttError::GeometryOverflow)?;
        let batch_run = widest_half.min(BATCH_BYTES.max(row_len));
        let mut row = Vec::new();
        reserve(&mut row, row_len)?;
        row.resize(row_len, 0);
        let mut batch_temp = Vec::new();
        reserve(&mut batch_temp, batch_run)?;
        batch_temp.resize(batch_run, 0);
        let mut batch_twiddles = Vec::new();
        reserve(&mut batch_twiddles, batch_run)?;
        batch_twiddles.resize(batch_run, 0);
        let transpose_flags = if self.fourstep.is_some() {
            self.size.div_ceil(8)
        } else {
            0
        };
        let mut transpose_bits = Vec::new();
        reserve(&mut transpose_bits, transpose_flags)?;
        transpose_bits.resize(transpose_flags, 0);
        Ok(NttScratch {
            row,
            batch_temp,
            batch_twiddles,
            transpose_bits,
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
    /// [`NttError::InvalidRowLength`] for a zero or partial-element row
    /// length; [`NttError::BufferLength`] for a wrong buffer length;
    /// [`NttError::ScratchFieldMismatch`] when the scratch was built for a
    /// different element width; [`NttError::ScratchTooSmall`] when its row
    /// capacity is below `row_len`; [`NttError::GeometryOverflow`] when
    /// the total length does not fit [`usize`].
    pub fn forward_bytes_scratch(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        canonicalize::<F>(rows);
        self.butterflies(rows, row_len, &self.forward_twiddles, scratch);
        Ok(())
    }

    /// Inverse transform, in place, over `size` rows of `row_len` bytes.
    ///
    /// Runs the forward loop with `root^-1` and then scales every lane by
    /// `size^-1`, so it inverts [`NttPlan::forward_bytes_scratch`] exactly.
    ///
    /// # Errors
    /// As [`NttPlan::forward_bytes_scratch`].
    pub fn inverse_bytes_scratch(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
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
        if row_len == 0 || !row_len.is_multiple_of(F::BYTES) {
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
            return Err(NttError::ScratchFieldMismatch {
                scratch_element_bytes: scratch.element_bytes,
                field_element_bytes: F::BYTES,
            });
        }
        if scratch.row_len < row_len {
            return Err(NttError::ScratchTooSmall {
                required: row_len,
                available: scratch.row_len,
            });
        }
        // The batched schedule's region buffers must cover this plan's
        // widest batch: scratch reused from a smaller plan passes the row
        // check above but cannot hold the run.
        let needed = batch_rows(row_len, self.size >> 1)
            .checked_mul(row_len)
            .ok_or(NttError::GeometryOverflow)?;
        if scratch.batch_temp.len() < needed {
            return Err(NttError::ScratchTooSmall {
                required: needed,
                available: scratch.batch_temp.len(),
            });
        }
        Ok(())
    }

    /// Bit-reversal permutation followed by `log_size` butterfly stages
    /// through the selected schedule.
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
        self.permute(rows, row_len);
        // The schedule is a pure function of the row geometry and the
        // resolved backend. Fields without vector elementwise kernels run
        // the fused register form at every row width — on scalar hosts the
        // region passes would ride the same scalar body through four
        // dispatches per run. Fields with vector kernels run the batched
        // region form below `BATCHED_ROW_MAX`, where the campaign shows it
        // ahead of both other schedules at every measured row width,
        // including sub-vector rows; at and above the bound the packed
        // op-call form keeps the wide rows, where its broadcast multiply
        // reads no twiddle stream and wins.
        if !F::has_vector_elementwise() {
            self.butterflies_fused(rows, row_len, twiddles);
        } else if row_len < BATCHED_ROW_MAX {
            self.butterflies_batched(rows, row_len, twiddles, scratch);
        } else {
            self.butterflies_packed(rows, row_len, twiddles, scratch);
        }
    }

    /// Swap rows into bit-reversed order, in place.
    fn permute(&self, rows: &mut [u8], row_len: usize) {
        for (index, &target) in self.permutation.iter().enumerate() {
            if index < target {
                let (head, tail) = rows.split_at_mut(target * row_len);
                head[index * row_len..(index + 1) * row_len].swap_with_slice(&mut tail[..row_len]);
            }
        }
    }

    /// `log_size` butterfly stages with each element pair held in registers.
    ///
    /// Every butterfly computes `low' = low + t·high` and
    /// `high' = low − t·high` one element at a time: no scratch row, no
    /// whole-row copy, and one backend-independent walk per stage. Operand
    /// bytes are canonical before the first stage, and every store writes a
    /// canonical element, so the invariant holds stage after stage. Geometry
    /// is established by [`NttPlan::validate`].
    fn butterflies_fused(&self, rows: &mut [u8], row_len: usize, twiddles: &[Coeff<F>]) {
        let lanes = row_len / F::BYTES;
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
                    let twiddle = twiddles[base + offset].value();
                    for lane in 0..lanes {
                        let at = lane * F::BYTES;
                        let u = F::decode(&low[at..at + F::BYTES]);
                        let v = F::decode(&high[at..at + F::BYTES]);
                        let w = if twiddle == F::Elem::ONE {
                            v
                        } else {
                            v.mul(twiddle)
                        };
                        F::encode(&mut low[at..at + F::BYTES], u.add(w));
                        F::encode(&mut high[at..at + F::BYTES], u.sub(w));
                    }
                }
            }
        }
    }

    /// The packed op-call butterfly form: one whole-row prepared multiply
    /// into scratch, one row copy, and packed add and subtract per pair.
    /// Kept as the tuning control the fused schedule is measured against;
    /// see the "NTT fused butterflies" record in `BENCHMARKS.md`.
    fn butterflies_packed(
        &self,
        rows: &mut [u8],
        row_len: usize,
        twiddles: &[Coeff<F>],
        scratch: &mut NttScratch,
    ) {
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

    /// The batched region-pass candidate: the same stages, pairs, and
    /// twiddles as [`NttPlan::butterflies_packed`], with the four packed
    /// operations applied to whole runs of pairs instead of one pair at a
    /// time. Runs never cross a block: the low and high regions of one
    /// block are contiguous, those of consecutive blocks are not. Stages
    /// whose block half is narrower than [`batched_region_min_bytes`] keep
    /// the per-pair packed body; the rest tile each run's twiddles across
    /// their lanes in [`NttScratch::batch_twiddles`], form the product
    /// with the high rows in [`NttScratch::batch_temp`] — written fully
    /// before the subtract and add passes read it — and bound both regions
    /// by [`BATCH_BYTES`]. Geometry is established by
    /// [`NttPlan::validate`], and the batch regions by
    /// [`NttPlan::forward_batched`].
    // `as_chunks_mut::<F::BYTES>()` would need a const generic argument
    // depending on `F`, which stable Rust does not accept.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn butterflies_batched(
        &self,
        rows: &mut [u8],
        row_len: usize,
        twiddles: &[Coeff<F>],
        scratch: &mut NttScratch,
    ) {
        for stage in 0..self.log_size {
            let half = 1usize << stage;
            let base = self.stage_offsets[stage];
            let half_bytes = half * row_len;
            if half_bytes < batched_region_min_bytes::<F>() {
                let temp = &mut scratch.row[..row_len];
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
                continue;
            }
            let run_rows = batch_rows(row_len, half);
            let run_bytes = run_rows * row_len;
            for block in rows.chunks_exact_mut(half_bytes * 2) {
                let (lows, highs) = block.split_at_mut(half_bytes);
                let NttScratch {
                    batch_temp,
                    batch_twiddles,
                    ..
                } = scratch;
                let temp = &mut batch_temp[..run_bytes];
                let expanded = &mut batch_twiddles[..run_bytes];
                let runs = lows
                    .chunks_exact_mut(run_bytes)
                    .zip(highs.chunks_exact_mut(run_bytes));
                for (run, (low_run, high_run)) in runs.enumerate() {
                    for (r, tw_row) in expanded.chunks_exact_mut(row_len).enumerate() {
                        let twiddle = twiddles[base + run * run_rows + r].value();
                        for slot in tw_row.chunks_exact_mut(F::BYTES) {
                            F::encode(slot, twiddle);
                        }
                    }
                    ops::mul_elementwise::<F>(temp, expanded, high_run);
                    high_run.copy_from_slice(low_run);
                    ops::sub_assign::<F>(high_run, temp);
                    ops::add_assign::<F>(low_run, temp);
                }
            }
        }
    }
}

impl<F: FieldKernels> NttPlan<F> {
    /// Tuning-only forward transform forced through
    /// [`NttPlan::butterflies_fused`]. Same contract as
    /// [`NttPlan::forward_bytes_scratch`]; reachable through the
    /// `internals` facade only.
    pub(crate) fn forward_fused(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        canonicalize::<F>(rows);
        self.permute(rows, row_len);
        self.butterflies_fused(rows, row_len, &self.forward_twiddles);
        Ok(())
    }

    /// Tuning-only forward transform forced through
    /// [`NttPlan::butterflies_packed`]. Same contract as
    /// [`NttPlan::forward_bytes_scratch`]; reachable through the
    /// `internals` facade only.
    pub(crate) fn forward_packed(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        canonicalize::<F>(rows);
        self.permute(rows, row_len);
        self.butterflies_packed(rows, row_len, &self.forward_twiddles, scratch);
        Ok(())
    }

    /// Tuning-only forward transform forced through
    /// [`NttPlan::butterflies_batched`]. Same contract as
    /// [`NttPlan::forward_bytes_scratch`], except the scratch must carry
    /// batch regions this plan's widest stage fills — any scratch this
    /// plan builds does; reachable through the `internals` facade only.
    pub(crate) fn forward_batched(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        let needed = batch_rows(row_len, self.size >> 1) * row_len;
        if scratch.batch_temp.len() < needed {
            return Err(NttError::ScratchTooSmall {
                required: needed,
                available: scratch.batch_temp.len(),
            });
        }
        canonicalize::<F>(rows);
        self.permute(rows, row_len);
        self.butterflies_batched(rows, row_len, &self.forward_twiddles, scratch);
        Ok(())
    }

    /// Tuning-only forward transform through the cache-blocked four-step
    /// decomposition. Same contract as
    /// [`NttPlan::forward_bytes_scratch`], except the scratch must carry
    /// transpose flags this plan's size fills — any scratch this plan
    /// builds does; reachable through the `internals` facade only. Below
    /// the activation size the entry runs the production schedule.
    pub(crate) fn forward_fourstep(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut NttScratch,
    ) -> Result<(), NttError> {
        self.validate(rows.len(), row_len, scratch)?;
        let Some(schedule) = &self.fourstep else {
            canonicalize::<F>(rows);
            self.butterflies(rows, row_len, &self.forward_twiddles, scratch);
            return Ok(());
        };
        let short = schedule.first.size();
        let long = schedule.second.size();
        let flags = self.size.div_ceil(8);
        let needed = batch_rows(row_len, short)
            .checked_mul(row_len)
            .ok_or(NttError::GeometryOverflow)?;
        if scratch.batch_temp.len() < needed {
            return Err(NttError::ScratchTooSmall {
                required: needed,
                available: scratch.batch_temp.len(),
            });
        }
        if scratch.transpose_bits.len() < flags {
            return Err(NttError::ScratchTooSmall {
                required: flags,
                available: scratch.transpose_bits.len(),
            });
        }
        canonicalize::<F>(rows);
        {
            let NttScratch {
                row,
                transpose_bits,
                ..
            } = scratch;
            transpose_slots(
                rows,
                row_len,
                short,
                long,
                &mut transpose_bits[..flags],
                &mut row[..row_len],
            );
        }
        for group in rows.chunks_exact_mut(short * row_len) {
            schedule
                .first
                .butterflies(group, row_len, &schedule.first.forward_twiddles, scratch);
        }
        self.fourstep_twiddles(rows, row_len, long, short, scratch);
        {
            let NttScratch {
                row,
                transpose_bits,
                ..
            } = scratch;
            transpose_slots(
                rows,
                row_len,
                long,
                short,
                &mut transpose_bits[..flags],
                &mut row[..row_len],
            );
        }
        for group in rows.chunks_exact_mut(long * row_len) {
            schedule
                .second
                .butterflies(group, row_len, &schedule.second.forward_twiddles, scratch);
        }
        {
            let NttScratch {
                row,
                transpose_bits,
                ..
            } = scratch;
            transpose_slots(
                rows,
                row_len,
                short,
                long,
                &mut transpose_bits[..flags],
                &mut row[..row_len],
            );
        }
        Ok(())
    }

    /// The four-step twiddle pass: every entry of the `long`-by-`short`
    /// slot matrix is scaled by `root^(row·column)`, so the two phases
    /// compose into the parent transform. Row zero is skipped because its
    /// factor is one; every other row's factors are powers of
    /// `root^row`, so one running product per row walks the whole matrix
    /// with one multiplication per entry. Wide rows tile their factors
    /// across the batch regions and multiply with the elementwise kernel,
    /// like the batched schedule; narrow rows multiply slot by slot with a
    /// prepared coefficient. Geometry and scratch capacity are established
    /// by [`NttPlan::forward_fourstep`].
    // `as_chunks_mut::<F::BYTES>()` would need a const generic argument
    // depending on `F`, which stable Rust does not accept.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn fourstep_twiddles(
        &self,
        rows: &mut [u8],
        row_len: usize,
        long: usize,
        short: usize,
        scratch: &mut NttScratch,
    ) {
        let run_rows = batch_rows(row_len, short);
        let run_bytes = run_rows * row_len;
        let mut row_factor = self.root;
        for row in 1..long {
            let group = &mut rows[row * short * row_len..(row + 1) * short * row_len];
            if run_bytes < batched_region_min_bytes::<F>() {
                let mut power = F::Elem::ONE;
                for slot in group.chunks_exact_mut(row_len) {
                    ops::mul_assign_with::<F>(slot, &Coeff::<F>::new(power));
                    power = power.mul(row_factor);
                }
            } else {
                let NttScratch {
                    batch_temp,
                    batch_twiddles,
                    ..
                } = scratch;
                let temp = &mut batch_temp[..run_bytes];
                let expanded = &mut batch_twiddles[..run_bytes];
                let mut power = F::Elem::ONE;
                for run in group.chunks_exact_mut(run_bytes) {
                    for tw_row in expanded.chunks_exact_mut(row_len) {
                        for slot in tw_row.chunks_exact_mut(F::BYTES) {
                            F::encode(slot, power);
                        }
                        power = power.mul(row_factor);
                    }
                    ops::mul_elementwise::<F>(temp, expanded, run);
                    run.copy_from_slice(temp);
                }
            }
            row_factor = row_factor.mul(self.root);
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
        F::encode(slot, F::decode(slot).add(F::Elem::ZERO));
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

/// Rows per batch of the batched schedule: the largest power of two that
/// divides `half` and keeps one batch's regions within [`BATCH_BYTES`],
/// or one row when a row alone exceeds the cap.
fn batch_rows(row_len: usize, half: usize) -> usize {
    let cap = (BATCH_BYTES / row_len).max(1);
    half.min(1 << cap.ilog2())
}

/// Conservative minimum block-half bytes at which the batched schedule
/// runs its region passes instead of its per-pair packed body: the vector
/// multiply floor with a two-element add/sub floor beside it, unmeasured
/// until the campaign behind the "NTT fused butterflies" record covers
/// this schedule.
#[inline]
fn batched_region_min_bytes<F: FieldKernels>() -> usize {
    F::vector_elementwise_min_bytes().max(F::BYTES * 2)
}

/// In-place transpose of a `height`-by-`width` matrix of `row_len`-byte
/// slots: slot `width·row + column` moves to `height·column + row`, whole
/// slots at a time. Each cycle of the permutation is followed through a
/// one-slot temporary, with `bits` carrying one visited flag per slot and
/// cleared on entry; fixed points — the diagonal of a square matrix,
/// isolated entries of a rectangular one — are skipped untouched.
/// Geometry is established by [`NttPlan::forward_fourstep`].
fn transpose_slots(
    rows: &mut [u8],
    row_len: usize,
    height: usize,
    width: usize,
    bits: &mut [u8],
    temp: &mut [u8],
) {
    bits.fill(0);
    for start in 0..height * width {
        // Cycles already walked by an earlier start are done: re-walking
        // applies the transpose to already-placed slots and undoes them.
        if bits[start >> 3] & (1 << (start & 7)) != 0 {
            continue;
        }
        // The walk pulls each slot from the next position along the cycle,
        // so it follows the inverse of the transpose mapping: slot
        // `width·row + column` still lands at `height·column + row`.
        if start % height * width + start / height == start {
            continue;
        }
        temp.copy_from_slice(&rows[start * row_len..(start + 1) * row_len]);
        let mut current = start;
        loop {
            bits[current >> 3] |= 1 << (current & 7);
            let next = current % height * width + current / height;
            if next == start {
                rows[current * row_len..(current + 1) * row_len].copy_from_slice(temp);
                break;
            }
            if next < current {
                let (head, tail) = rows.split_at_mut(current * row_len);
                tail[..row_len].copy_from_slice(&head[next * row_len..(next + 1) * row_len]);
            } else {
                let (head, tail) = rows.split_at_mut(next * row_len);
                head[current * row_len..(current + 1) * row_len].copy_from_slice(&tail[..row_len]);
            }
            current = next;
        }
    }
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

#[cfg(test)]
mod tests {
    //! The two butterfly schedules are algebraically identical, so
    //! byte-for-byte agreement at straddling geometries is the contract for
    //! the comparison that can run everywhere. The Goldilocks packed
    //! schedule rides released fgf's packed kernels, which the
    //! "Known correctness finding" in `BENCHMARKS.md` documents as computing
    //! incorrect values nondeterministically once a Goldilocks buffer
    //! reaches 256 elements on GFNI hosts, so wide Goldilocks geometries
    //! check the fused schedule against an independent direct DFT instead
    //! of against the suspect control. `QuadMersenne31` has no packed
    //! kernels and compares at every geometry.
    use super::*;
    use alloc::vec;
    use fgf::{Gf16, Goldilocks, QuadMersenne31};

    /// Deterministic xorshift, matching the integration suites.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn bytes(&mut self, target: &mut [u8]) {
            let (slots, _) = target.as_chunks_mut::<8>();
            for slot in slots {
                slot.copy_from_slice(&self.next().to_le_bytes());
            }
        }
    }

    /// Independent direct transform: `Y[k] = Σ_j x[j] · root^(j·k)`, the
    /// same formulation `tests/ntt.rs` uses as its ground truth.
    fn direct_dft_lane<F: FieldKernels>(input: &[F::Elem], root: F::Elem) -> Vec<F::Elem> {
        let size = input.len();
        (0..size)
            .map(|k| {
                let mut total = F::Elem::ZERO;
                for (j, &value) in input.iter().enumerate() {
                    let exponent = (j * k) % size;
                    total = total.add(value.mul(root.pow(exponent as u64)));
                }
                total
            })
            .collect()
    }

    fn check_schedules_agree<F: FieldKernels>(size: usize, lanes: usize, seed: u64) {
        let row_len = lanes * F::BYTES;
        let plan = NttPlan::<F>::new(size).expect("supported size");
        let mut rng = Rng(seed);
        let mut fused = vec![0u8; size * row_len];
        rng.bytes(&mut fused);
        let mut packed = fused.clone();
        let mut scratch = plan.scratch(row_len).expect("scratch");

        plan.forward_fused(&mut fused, row_len, &mut scratch)
            .expect("geometry");
        plan.forward_packed(&mut packed, row_len, &mut scratch)
            .expect("geometry");
        assert_eq!(
            fused,
            packed,
            "{} size {size} lanes {lanes}: schedules disagree",
            F::NAME
        );
    }

    fn check_fused_matches_direct_dft<F: FieldKernels>(size: usize, lanes: usize, seed: u64) {
        let row_len = lanes * F::BYTES;
        let plan = NttPlan::<F>::new(size).expect("supported size");
        let mut rng = Rng(seed);
        let mut rows = vec![0u8; size * row_len];
        rng.bytes(&mut rows);
        let original = rows.clone();
        let mut scratch = plan.scratch(row_len).expect("scratch");

        plan.forward_fused(&mut rows, row_len, &mut scratch)
            .expect("geometry");
        for lane in 0..lanes {
            let input: Vec<F::Elem> = (0..size)
                .map(|point| {
                    let at = point * row_len + lane * F::BYTES;
                    F::decode(&original[at..at + F::BYTES])
                })
                .collect();
            let expected = direct_dft_lane::<F>(&input, plan.root());
            for (point, want) in expected.iter().enumerate() {
                let at = point * row_len + lane * F::BYTES;
                assert_eq!(
                    F::decode(&rows[at..at + F::BYTES]),
                    *want,
                    "{} size {size} lane {lane}: fused forward disagrees with the direct DFT",
                    F::NAME
                );
            }
        }
    }

    /// Geometry lists shrink under miri: the schedules and the oracle are
    /// the same code at every size, and the quadratic direct DFT has no
    /// business running there.
    fn agreement_sizes() -> &'static [usize] {
        if cfg!(miri) {
            &[1, 2, 4, 8]
        } else {
            &[1, 2, 4, 8, 16, 32, 64, 256, 1024]
        }
    }

    fn dft_sizes() -> &'static [usize] {
        if cfg!(miri) { &[8, 32] } else { &[256, 1024] }
    }

    fn lane_widths() -> &'static [usize] {
        if cfg!(miri) {
            &[1, 2, 4]
        } else {
            &[1, 2, 4, 16]
        }
    }

    #[test]
    fn goldilocks_schedules_agree() {
        for &size in agreement_sizes() {
            for &lanes in lane_widths() {
                check_schedules_agree::<Goldilocks>(
                    size,
                    lanes,
                    0x0f0f_1e2d ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    #[test]
    fn goldilocks_fused_matches_direct_dft_at_wide_geometries() {
        for &size in dft_sizes() {
            for &lanes in lane_widths() {
                check_fused_matches_direct_dft::<Goldilocks>(
                    size,
                    lanes,
                    0x5eed_1234 ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    #[test]
    fn quad_mersenne31_schedules_agree() {
        for &size in agreement_sizes() {
            for &lanes in lane_widths() {
                check_schedules_agree::<QuadMersenne31>(
                    size,
                    lanes,
                    0x0bad_c0de ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    /// The batched schedule applies the packed operations to runs of
    /// pairs, so it must reproduce the fused schedule byte for byte at
    /// every geometry the fixed fgf kernels serve.
    fn check_batched_agrees_with_fused<F: FieldKernels>(size: usize, lanes: usize, seed: u64) {
        let row_len = lanes * F::BYTES;
        let plan = NttPlan::<F>::new(size).expect("supported size");
        let mut rng = Rng(seed);
        let mut fused = vec![0u8; size * row_len];
        rng.bytes(&mut fused);
        let mut batched = fused.clone();
        let mut scratch = plan.scratch(row_len).expect("scratch");

        plan.forward_fused(&mut fused, row_len, &mut scratch)
            .expect("geometry");
        plan.forward_batched(&mut batched, row_len, &mut scratch)
            .expect("geometry");
        assert_eq!(
            fused,
            batched,
            "{} size {size} lanes {lanes}: batched disagrees with fused",
            F::NAME
        );
    }

    /// Geometries whose widest block halves exceed the batch cap, so the
    /// last stages split each block across several batches. The fused
    /// walk over a buffer this size keeps them out of miri.
    fn batch_split_geometries() -> &'static [(usize, usize)] {
        if cfg!(miri) { &[] } else { &[(8192, 16)] }
    }

    #[test]
    fn goldilocks_batched_agrees_with_fused() {
        for &size in agreement_sizes() {
            for &lanes in lane_widths() {
                check_batched_agrees_with_fused::<Goldilocks>(
                    size,
                    lanes,
                    0x1eaf_1f2e ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    #[test]
    fn quad_mersenne31_batched_agrees_with_fused() {
        for &size in agreement_sizes() {
            for &lanes in lane_widths() {
                check_batched_agrees_with_fused::<QuadMersenne31>(
                    size,
                    lanes,
                    0xdec0_ded0 ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    #[test]
    fn batched_agrees_when_blocks_split_into_batches() {
        for &(size, lanes) in batch_split_geometries() {
            check_batched_agrees_with_fused::<Goldilocks>(size, lanes, 0x8bad_f00d);
            check_batched_agrees_with_fused::<QuadMersenne31>(size, lanes, 0xfeed_face);
        }
    }

    /// The binary fields admit only the identity transform, where both
    /// schedules reduce to the unpermuted no-op.
    #[test]
    fn gf16_batched_identity_agrees_with_fused() {
        let row_len = 8 * Gf16::BYTES;
        let plan = NttPlan::<Gf16>::new(1).expect("identity size");
        let mut rng = Rng(0x9f31_7777);
        let mut fused = vec![0u8; row_len];
        rng.bytes(&mut fused);
        let mut batched = fused.clone();
        let mut scratch = plan.scratch(row_len).expect("scratch");

        plan.forward_fused(&mut fused, row_len, &mut scratch)
            .expect("geometry");
        plan.forward_batched(&mut batched, row_len, &mut scratch)
            .expect("geometry");
        assert_eq!(fused, batched);
    }

    /// A scratch whose batch regions were sized by a smaller plan is
    /// rejected before anything is written, mirroring the row-capacity
    /// rejection of the production entries.
    #[test]
    fn batched_rejects_scratch_from_a_smaller_plan() {
        let small = NttPlan::<Goldilocks>::new(64).expect("supported size");
        let mut scratch = small.scratch(8).expect("scratch");
        let plan = NttPlan::<Goldilocks>::new(1024).expect("supported size");
        let mut rows = vec![0xa5u8; 1024 * 8];
        let pristine = rows.clone();
        assert_eq!(
            plan.forward_batched(&mut rows, 8, &mut scratch)
                .unwrap_err(),
            NttError::ScratchTooSmall {
                required: 4096,
                available: 256
            }
        );
        assert_eq!(rows, pristine, "batch rejection mutated the rows");
    }

    /// The four-step decomposition against the fused schedule, over
    /// whichever plan the caller built.
    fn check_fourstep_agrees_with_fused<F: FieldKernels>(
        plan: &NttPlan<F>,
        size: usize,
        lanes: usize,
        seed: u64,
    ) {
        let row_len = lanes * F::BYTES;
        let mut rng = Rng(seed);
        let mut fused = vec![0u8; size * row_len];
        rng.bytes(&mut fused);
        let mut fourstep = fused.clone();
        let mut scratch = plan.scratch(row_len).expect("scratch");

        plan.forward_fused(&mut fused, row_len, &mut scratch)
            .expect("geometry");
        plan.forward_fourstep(&mut fourstep, row_len, &mut scratch)
            .expect("geometry");
        assert_eq!(
            fused,
            fourstep,
            "{} size {size} lanes {lanes}: fourstep disagrees with fused",
            F::NAME
        );
    }

    #[test]
    fn goldilocks_fourstep_agrees_with_fused() {
        let threshold = 1usize << FOURSTEP_MIN_LOG;
        for &size in agreement_sizes().iter().filter(|&&size| size >= threshold) {
            for &lanes in lane_widths() {
                let plan = NttPlan::<Goldilocks>::new(size).expect("supported size");
                check_fourstep_agrees_with_fused::<Goldilocks>(
                    &plan,
                    size,
                    lanes,
                    0x5eed_4b1d ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    #[test]
    fn quad_mersenne31_fourstep_agrees_with_fused() {
        let threshold = 1usize << FOURSTEP_MIN_LOG;
        for &size in agreement_sizes().iter().filter(|&&size| size >= threshold) {
            for &lanes in lane_widths() {
                let plan = NttPlan::<QuadMersenne31>::new(size).expect("supported size");
                check_fourstep_agrees_with_fused::<QuadMersenne31>(
                    &plan,
                    size,
                    lanes,
                    0xdec0_5eed ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    /// Odd log sizes exercise the rectangular transpose: size eight
    /// factors as the long-by-short pair four-by-two, size sixteen as the
    /// square four-by-four, size thirty-two as eight-by-four. These sit
    /// below the activation size, so they enter through the shared
    /// constructor with a lowered threshold, reaching the same machinery
    /// the production plans carry.
    #[test]
    fn fourstep_non_square_decomposition_agrees_with_fused() {
        for &size in &[8usize, 16, 32] {
            for &lanes in lane_widths() {
                let plan = NttPlan::<Goldilocks>::build(size, 3).expect("supported size");
                check_fourstep_agrees_with_fused::<Goldilocks>(
                    &plan,
                    size,
                    lanes,
                    0x7ea5_1f2e ^ (size as u64) << 32 ^ lanes as u64,
                );
                let plan = NttPlan::<QuadMersenne31>::build(size, 3).expect("supported size");
                check_fourstep_agrees_with_fused::<QuadMersenne31>(
                    &plan,
                    size,
                    lanes,
                    0xfeed_7ea5 ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
        if !cfg!(miri) {
            // A production-size rectangular decomposition, eight-by-four
            // in slots, with wide rows.
            let plan = NttPlan::<Goldilocks>::new(8192).expect("supported size");
            check_fourstep_agrees_with_fused::<Goldilocks>(&plan, 8192, 16, 0x8bad_5eed);
            let plan = NttPlan::<QuadMersenne31>::new(8192).expect("supported size");
            check_fourstep_agrees_with_fused::<QuadMersenne31>(&plan, 8192, 16, 0x5eed_face);
        }
    }

    /// Below the activation size the four-step entry runs the production
    /// schedule, byte-identical to the fused tuning entry.
    #[test]
    fn fourstep_below_threshold_passes_through() {
        for &size in &[8usize, 64] {
            for &lanes in lane_widths() {
                let plan = NttPlan::<Goldilocks>::new(size).expect("supported size");
                check_fourstep_agrees_with_fused::<Goldilocks>(
                    &plan,
                    size,
                    lanes,
                    0x0f0f_4b1d ^ (size as u64) << 32 ^ lanes as u64,
                );
                let plan = NttPlan::<QuadMersenne31>::new(size).expect("supported size");
                check_fourstep_agrees_with_fused::<QuadMersenne31>(
                    &plan,
                    size,
                    lanes,
                    0x0bad_4b1d ^ (size as u64) << 32 ^ lanes as u64,
                );
            }
        }
    }

    /// A scratch whose transpose flags were sized by a smaller plan is
    /// rejected before anything is written, mirroring the batch-region
    /// rejection of the batched entry.
    #[test]
    fn fourstep_rejects_scratch_from_a_smaller_plan() {
        let small = NttPlan::<Goldilocks>::new(64).expect("supported size");
        let mut scratch = small.scratch(8).expect("scratch");
        let plan = NttPlan::<Goldilocks>::new(1024).expect("supported size");
        let mut rows = vec![0xa5u8; 1024 * 8];
        let pristine = rows.clone();
        // The shared batch-region check fires first: the batched schedule
        // is the widest region any execution of this plan allocates.
        assert_eq!(
            plan.forward_fourstep(&mut rows, 8, &mut scratch)
                .unwrap_err(),
            NttError::ScratchTooSmall {
                required: 4096,
                available: 256
            }
        );
        assert_eq!(rows, pristine, "fourstep rejection mutated the rows");
    }
}
