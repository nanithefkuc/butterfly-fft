//! Explicit experimental four-step preparation and execution workspace.

use ::alloc::vec::Vec;
use fgf::field::Elem;
use fgf::kernel::FieldKernels;
use fgf::ops::{self, Coeff};

use super::{
    NttError, NttPlan, NttScratch, batch_rows, batched_region_min_bytes, canonicalize, domain,
    reserve, validate_scratch,
};

/// Prepared cache-blocked forward NTT, exposed only through `internals`.
///
/// Owns two radix-two sub-plans with roots `root^long` and `root^short`.
/// Every supported size uses the decomposition, including the identity;
/// there is no production crossover or fallback. Construction allocates the
/// sub-plan tables but no execution workspace. The intervening twiddle pass
/// retains its running-power arithmetic and per-row coefficient preparation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct FourStepPlan<F: FieldKernels> {
    size: usize,
    root: F::Elem,
    first: NttPlan<F>,
    second: NttPlan<F>,
}

/// Reusable four-step workspace, exposed only through `internals`.
///
/// Owns a row temporary, two batch regions shared by the sub-transforms and
/// twiddle pass, and one transpose-visited bit per transform point. Reuse at
/// a shorter whole-element row length and the same element width is allowed;
/// execution checks batch and transpose capacity before changing any rows.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct FourStepScratch {
    work: NttScratch,
    transpose_bits: Vec<u8>,
}

#[allow(dead_code)]
impl<F: FieldKernels> FourStepPlan<F> {
    /// Prepare a four-step forward transform over `size` points.
    ///
    /// Uses the same root, size cap, and domain validation as [`NttPlan::new`].
    ///
    /// # Errors
    /// As [`NttPlan::new`].
    pub fn new(size: usize) -> Result<Self, NttError> {
        let (log_size, root) = domain::<F>(size)?;
        let short = 1usize << (log_size / 2);
        let long = 1usize << log_size.div_ceil(2);
        Ok(Self {
            size,
            root,
            first: NttPlan::new(short)?,
            second: NttPlan::new(long)?,
        })
    }

    /// Number of transform points.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Primitive root defining `Y[k] = Σ_j x[j] · root^(j·k)`.
    #[must_use]
    pub fn root(&self) -> F::Elem {
        self.root
    }

    /// Allocate execution workspace for rows up to `row_len` bytes.
    ///
    /// Batch regions cover both sub-transforms and the intervening twiddle
    /// pass at every shorter valid row length. Transpose storage covers this
    /// plan's size; reuse with larger plans can be rejected.
    ///
    /// # Errors
    /// As [`NttPlan::scratch`].
    pub fn scratch(&self, row_len: usize) -> Result<FourStepScratch, NttError> {
        let work = NttScratch::new::<F>(row_len, self.first.size())?;
        let flags = self.size.div_ceil(8);
        let mut transpose_bits = Vec::new();
        reserve(&mut transpose_bits, flags)?;
        transpose_bits.resize(flags, 0);
        Ok(FourStepScratch {
            work,
            transpose_bits,
        })
    }

    /// Forward transform in place with natural-order input and output.
    ///
    /// The rows and canonicalization contract matches
    /// [`NttPlan::forward_bytes_scratch`]. Allocates nothing; the twiddle pass
    /// computes running powers during execution. All checks precede mutation.
    ///
    /// # Errors
    /// As [`NttPlan::forward_bytes_scratch`], plus
    /// [`NttError::ScratchTransposeTooSmall`] for insufficient transpose flags.
    pub fn forward_bytes_scratch(
        &self,
        rows: &mut [u8],
        row_len: usize,
        scratch: &mut FourStepScratch,
    ) -> Result<(), NttError> {
        let short = self.first.size();
        let long = self.second.size();
        validate_scratch::<F>(self.size, rows.len(), row_len, &scratch.work, short)?;
        let flags = self.size.div_ceil(8);
        if scratch.transpose_bits.len() < flags {
            return Err(NttError::ScratchTransposeTooSmall {
                required: flags,
                available: scratch.transpose_bits.len(),
            });
        }
        let FourStepScratch {
            work,
            transpose_bits,
        } = scratch;
        canonicalize::<F>(rows);
        transpose_slots(
            rows,
            row_len,
            short,
            long,
            &mut transpose_bits[..flags],
            &mut work.row[..row_len],
        );
        for group in rows.chunks_exact_mut(short * row_len) {
            self.first
                .butterflies(group, row_len, &self.first.forward_twiddles, work);
        }
        self.fourstep_twiddles(rows, row_len, long, short, work);
        transpose_slots(
            rows,
            row_len,
            long,
            short,
            &mut transpose_bits[..flags],
            &mut work.row[..row_len],
        );
        for group in rows.chunks_exact_mut(long * row_len) {
            self.second
                .butterflies(group, row_len, &self.second.forward_twiddles, work);
        }
        transpose_slots(
            rows,
            row_len,
            short,
            long,
            &mut transpose_bits[..flags],
            &mut work.row[..row_len],
        );
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
    /// by [`FourStepPlan::forward_bytes_scratch`].
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

/// In-place transpose of a `height`-by-`width` matrix of `row_len`-byte
/// slots: slot `width·row + column` moves to `height·column + row`, whole
/// slots at a time. Each cycle of the permutation is followed through a
/// one-slot temporary, with `bits` carrying one visited flag per slot and
/// cleared on entry; fixed points — the diagonal of a square matrix,
/// isolated entries of a rectangular one — are skipped untouched.
/// Geometry is established by [`FourStepPlan::forward_bytes_scratch`].
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
