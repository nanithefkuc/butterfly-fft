//! Transform plans and in-place execution models.
//!
//! A plan owns a [`crate::core::factors`] table for a power-of-two domain.
//! Execution walks the table recursively, in place, allocating nothing:
//! full forward/inverse, selected and range-restricted outputs, truncated
//! transforms for zero-padded inputs, high-coset evaluation, and the
//! novel-basis formal derivative.
//!
//! ```
//! use cafft::core::transform::TransformPlan;
//! use fgf::{Gf16, gf16};
//!
//! let plan = TransformPlan::<Gf16>::new(4).unwrap();
//! let mut values = [
//!     gf16::Elem(0x1234),
//!     gf16::Elem(0xabcd),
//!     gf16::Elem(0x0108),
//!     gf16::Elem(0xffff),
//! ];
//! let original = values;
//! plan.forward(&mut values).unwrap();
//! plan.inverse(&mut values).unwrap();
//! assert_eq!(values, original);
//! ```

use ::alloc::{sync::Arc, vec::Vec};
use ::core::ops::Range;

use fgf::field::{Elem, Field};
use fgf::ops::{self, Coeff};

use crate::core::factors::{self, FactorTable};
use crate::core::kernel::{
    ButterflyBackend, ButterflyKernels, fused_forward_with, fused_inverse_with,
};

pub use crate::error::{PlanError, TransformLengthError};

/// Largest supported transform dimension: `2^MAX_LOG_SIZE` points.
///
/// Conservative cap keeping a factor table to a few megabytes per field;
/// the field's own extension degree caps smaller fields further.
pub const MAX_LOG_SIZE: usize = 20;

/// Reusable additive-FFT plan for a power-of-two number of field elements.
///
/// Construction precomputes one normalized subspace-polynomial value per
/// recursive node (the twiddle table) and prepares the formal-derivative
/// factors for fgf's selected backend. Forward and inverse execution each take
/// `N·log2(N)` field butterflies and allocate no memory.
///
/// Evaluation points are enumerated in basis order: point `i` is the XOR of
/// the basis elements at the set bits of `i` (see
/// [`TransformPlan::point_element`]). Under the default bit basis, point `i`
/// is the field element whose little-endian bytes hold the value `i`.
#[derive(Clone, Debug)]
pub struct TransformPlan<F: ButterflyKernels> {
    size: usize,
    log_size: usize,
    /// Binary-heap node layout; root is index one.
    table: FactorTable<F>,
    /// Derivative factors resolved once into fgf's selected backend form.
    prepared_derivative_factors: Vec<Coeff<F>>,
    /// The ordered-basis prefix `β_0 … β_{log_size-1}` defining the domain.
    basis: Vec<F::Elem>,
}

/// Validate a requested transform size, returning its base-two logarithm.
pub(crate) fn validate_size<F: Field>(size: usize) -> Result<usize, PlanError> {
    if size == 0 || !size.is_power_of_two() {
        return Err(PlanError::InvalidSize { size });
    }
    let log_size = size.trailing_zeros() as usize;
    let cap = (F::BITS as usize).min(MAX_LOG_SIZE);
    if log_size > cap {
        return Err(PlanError::DomainTooLarge { log_size, cap });
    }
    Ok(log_size)
}

impl<F: ButterflyKernels> TransformPlan<F> {
    /// Construct a plan for a power-of-two size in `1..=2^min(BITS,
    /// MAX_LOG_SIZE)` over the default bit basis (`β_i` = element with bit
    /// pattern `1 << i`).
    ///
    /// # Errors
    /// Returns [`PlanError`] for an invalid or oversized `size`.
    pub fn new(size: usize) -> Result<Self, PlanError> {
        let log_size = validate_size::<F>(size)?;
        Self::construct(size, log_size, factors::bit_basis::<F>(log_size))
    }

    /// Construct a plan over a custom ordered basis.
    ///
    /// `basis` must hold at least `log2(size)` linearly independent
    /// elements; the prefix `basis[..log2(size)]` defines the domain.
    ///
    /// # Errors
    /// Returns [`PlanError`] for an invalid size, a short basis, or a
    /// linearly dependent basis prefix.
    pub fn with_basis(size: usize, basis: &[F::Elem]) -> Result<Self, PlanError> {
        let log_size = validate_size::<F>(size)?;
        if basis.len() < log_size {
            return Err(PlanError::BasisTooShort {
                needed: log_size,
                got: basis.len(),
            });
        }
        if !factors::linearly_independent::<F>(&basis[..log_size]) {
            return Err(PlanError::DependentBasis);
        }
        Self::construct(size, log_size, basis[..log_size].to_vec())
    }

    /// Construct a plan whose domain is the affine coset
    /// `shift + span(basis[..log2(size)])`.
    ///
    /// The twiddle table has the same shape; only the root coset shift
    /// changes, so every execution model runs unmodified. Exposed to
    /// consumers through [`crate::shifted::ShiftedPlan`].
    ///
    /// # Errors
    /// As [`TransformPlan::with_basis`].
    pub(crate) fn with_basis_shift(
        size: usize,
        basis: &[F::Elem],
        shift: F::Elem,
    ) -> Result<Self, PlanError> {
        let log_size = validate_size::<F>(size)?;
        if basis.len() < log_size {
            return Err(PlanError::BasisTooShort {
                needed: log_size,
                got: basis.len(),
            });
        }
        if !factors::linearly_independent::<F>(&basis[..log_size]) {
            return Err(PlanError::DependentBasis);
        }
        let table = FactorTable::build(log_size, basis, shift).ok_or(PlanError::DependentBasis)?;
        Ok(Self::from_parts(
            size,
            log_size,
            table,
            basis[..log_size].to_vec(),
        ))
    }

    fn construct(size: usize, log_size: usize, basis: Vec<F::Elem>) -> Result<Self, PlanError> {
        let table =
            FactorTable::build(log_size, &basis, F::Elem::ZERO).ok_or(PlanError::DependentBasis)?;
        Ok(Self::from_parts(size, log_size, table, basis))
    }

    fn from_parts(
        size: usize,
        log_size: usize,
        table: FactorTable<F>,
        basis: Vec<F::Elem>,
    ) -> Self {
        let prepared_derivative_factors = table
            .derivative_factors
            .iter()
            .copied()
            .map(Coeff::<F>::new)
            .collect();
        Self {
            size,
            log_size,
            table,
            prepared_derivative_factors,
            basis,
        }
    }

    /// The shared plan for this size and field, building it on first use.
    ///
    /// Plans are immutable and thread-safe; repeated encoders/decoders over
    /// the same domain should share one table rather than rebuild it.
    ///
    /// # Errors
    /// Returns [`PlanError`] for an invalid or oversized `size`.
    #[cfg(feature = "std")]
    pub fn shared(size: usize) -> Result<Arc<Self>, PlanError>
    where
        F::Elem: Send + Sync,
    {
        let log_size = validate_size::<F>(size)?;
        let key = (::core::any::TypeId::of::<F>(), log_size);
        {
            let cache = SHARED_PLANS
                .read()
                .unwrap_or_else(::std::sync::PoisonError::into_inner);
            if let Some(plan) = cache
                .get(&key)
                .and_then(|erased| erased.clone().downcast::<Self>().ok())
            {
                return Ok(plan);
            }
        }
        let fresh = Arc::new(Self::new(size)?);
        let erased: Arc<dyn ::core::any::Any + Send + Sync> = fresh.clone();
        let mut cache = SHARED_PLANS
            .write()
            .unwrap_or_else(::std::sync::PoisonError::into_inner);
        let stored = cache.entry(key).or_insert(erased).clone();
        // The key includes F's TypeId, so the downcast cannot fail; fall
        // back to the freshly built plan to keep this path panic-free.
        Ok(stored.downcast::<Self>().unwrap_or(fresh))
    }

    /// Number of transform points.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Base-two logarithm of [`TransformPlan::size`].
    #[must_use]
    pub const fn log_size(&self) -> usize {
        self.log_size
    }

    /// The ordered-basis prefix `β_0 … β_{log_size-1}` defining the domain.
    #[must_use]
    pub fn basis(&self) -> &[F::Elem] {
        &self.basis
    }
    /// Coefficients of `W̄_dimension`, low Frobenius exponent first.
    pub(crate) fn normalized_subspace_polynomial(&self, dimension: usize) -> &[F::Elem] {
        &self.table.polynomials[dimension].normalized_coefficients
    }

    /// The plan's twiddle and derivative tables.
    ///
    /// This unstable inspection API is available only with feature
    /// `internals` and carries no compatibility guarantee.
    #[cfg(feature = "internals")]
    #[must_use]
    pub const fn table(&self) -> &FactorTable<F> {
        &self.table
    }

    /// The field element of transform point `index`: the XOR of the basis
    /// elements at the set bits of `index`.
    #[must_use]
    pub fn point_element(&self, index: usize) -> F::Elem {
        debug_assert!(index < self.size);
        let mut element = F::Elem::ZERO;
        let mut remaining = index;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            element = element.add(self.basis[bit]);
            remaining &= remaining - 1;
        }
        element
    }

    /// Evaluate novel-basis coefficients at the plan's points, in place.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] unless `values.len() == size`.
    pub fn forward(&self, values: &mut [F::Elem]) -> Result<(), TransformLengthError> {
        self.check_len(values.len())?;
        if self.log_size != 0 {
            forward_node(values, &self.table.factors, 1, self.log_size);
        }
        Ok(())
    }

    /// Convert evaluations at the plan's points back to novel-basis
    /// coefficients, in place.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] unless `values.len() == size`.
    pub fn inverse(&self, values: &mut [F::Elem]) -> Result<(), TransformLengthError> {
        self.check_len(values.len())?;
        if self.log_size != 0 {
            inverse_node(values, &self.table.factors, 1, self.log_size);
        }
        Ok(())
    }

    /// Forward transform over interleaved byte rows: row `i` (of `row_len`
    /// bytes) holds the payload for transform point `i`, transformed in
    /// place with SIMD-dispatched fused butterflies.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `rows.len() == size * row_len`.
    ///
    /// Panics if `row_len` is zero, holds a partial trailing element, or the
    /// complete byte length is not representable by [`usize`].
    pub fn forward_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        if self.log_size != 0 {
            crate::core::kernel::dispatch_butterfly!(
                F,
                forward_bytes_node(rows, &self.table.factors, 1, self.log_size)
            );
        }
        Ok(())
    }

    /// Inverse transform over interleaved byte rows.
    ///
    /// # Errors
    /// Same contract as [`TransformPlan::forward_bytes`].
    pub fn inverse_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        if self.log_size != 0 {
            crate::core::kernel::dispatch_butterfly!(
                F,
                inverse_bytes_node(rows, &self.table.factors, 1, self.log_size)
            );
        }
        Ok(())
    }

    /// Formal derivative of the novel-basis coefficient vector, out of
    /// place: `X_i'(x) = Σ_{j ∈ bits(i)} W̄_j'·X_{i ^ 2^j}(x)`.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] unless both slices hold `size`
    /// elements.
    pub fn derivative(
        &self,
        coefficients: &[F::Elem],
        derivative: &mut [F::Elem],
    ) -> Result<(), TransformLengthError> {
        self.check_len(coefficients.len())?;
        self.check_len(derivative.len())?;
        derivative.fill(F::Elem::ZERO);
        // The index is the data: its set bits drive the scatter into
        // `derivative`, so an iterator over coefficients cannot replace it.
        #[expect(clippy::needless_range_loop)]
        for index in 1..self.size {
            let source = coefficients[index];
            let mut remaining = index;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let destination = index ^ (1 << bit);
                derivative[destination] =
                    derivative[destination].add(self.table.derivative_factors[bit].mul(source));
                remaining &= remaining - 1;
            }
        }
        Ok(())
    }

    /// Formal derivative over interleaved byte rows, out of place.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless both
    /// buffers hold `size` rows of `row_len` bytes.
    ///
    /// # Panics
    /// Under the row-geometry conditions documented by
    /// [`TransformPlan::forward_bytes`].
    pub fn derivative_bytes(
        &self,
        coefficients: &[u8],
        row_len: usize,
        derivative: &mut [u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(coefficients.len(), row_len)?;
        self.check_len_bytes(derivative.len(), row_len)?;
        derivative.fill(0);
        for (bit, factor) in self.prepared_derivative_factors.iter().enumerate() {
            let half_len = (1 << bit) * row_len;
            let block_len = half_len * 2;
            // For a fixed bit, every source with that bit set occupies the
            // upper half of a block and contributes to the matching lower half.
            for (source_block, destination_block) in coefficients
                .chunks_exact(block_len)
                .zip(derivative.chunks_exact_mut(block_len))
            {
                ops::mul_add_with::<F>(
                    &mut destination_block[..half_len],
                    factor,
                    &source_block[half_len..],
                );
            }
        }
        Ok(())
    }

    /// Forward transform that evaluates only the rows in `selected`
    /// (sorted, unique, in range).
    ///
    /// Only selected rows are final outputs. Every other row may hold an
    /// undefined intermediate value after the call. Used by decoders to
    /// evaluate repairs or missing points without paying for the full domain.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `rows.len() == size * row_len`.
    ///
    /// # Panics
    /// Panics if `selected` is not strictly increasing or names a row
    /// `>= size`, or under the row-geometry conditions documented by
    /// [`TransformPlan::forward_bytes`].
    pub fn forward_bytes_selected(
        &self,
        rows: &mut [u8],
        row_len: usize,
        selected: &[usize],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        assert!(
            selected.iter().all(|&index| index < self.size),
            "selected row out of range"
        );
        assert!(
            selected.windows(2).all(|pair| pair[0] < pair[1]),
            "selected rows must be sorted and unique"
        );
        if self.log_size != 0 && !selected.is_empty() {
            crate::core::kernel::dispatch_butterfly!(
                F,
                forward_bytes_selected_node(
                    rows,
                    row_len,
                    &self.table.factors,
                    1,
                    self.log_size,
                    0,
                    selected,
                )
            );
        }
        Ok(())
    }

    /// Forward transform restricted to the contiguous output `range` of
    /// rows; other rows are left holding intermediate values.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `rows.len() == size * row_len`.
    ///
    /// # Panics
    /// Panics unless `range.start <= range.end <= size`, or under the
    /// row-geometry conditions documented by [`TransformPlan::forward_bytes`].
    pub fn forward_bytes_range(
        &self,
        rows: &mut [u8],
        row_len: usize,
        range: Range<usize>,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        assert!(
            range.start <= range.end && range.end <= self.size,
            "range out of bounds"
        );
        if self.log_size != 0 && !range.is_empty() {
            crate::core::kernel::dispatch_butterfly!(
                F,
                forward_bytes_range_node(
                    rows,
                    row_len,
                    &self.table.factors,
                    1,
                    self.log_size,
                    range.start,
                    range.end,
                )
            );
        }
        Ok(())
    }

    /// Truncated forward transform: exploits a leading-`active` nonzero
    /// coefficient prefix (rows `active..size` must be zero on entry) *and*
    /// restricts output to `range`.
    ///
    /// Butterflies whose entire high coefficient half lies in the zero
    /// region reduce to a copy with no field multiply, so the padding
    /// between the message dimension and the power-of-two domain costs no
    /// arithmetic. On the requested rows the result is identical to a full
    /// forward transform.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `rows.len() == size * row_len`.
    ///
    /// # Panics
    /// Panics unless `active <= size` and `range.start <= range.end <=
    /// size`, or under the row-geometry conditions documented by
    /// [`TransformPlan::forward_bytes`].
    pub fn forward_bytes_trunc_range(
        &self,
        rows: &mut [u8],
        row_len: usize,
        active: usize,
        range: Range<usize>,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        assert!(active <= self.size, "active prefix out of range");
        assert!(
            range.start <= range.end && range.end <= self.size,
            "range out of bounds"
        );
        if self.log_size != 0 && !range.is_empty() && active != 0 {
            crate::core::kernel::dispatch_butterfly!(
                F,
                forward_bytes_trunc_range_node(
                    rows,
                    row_len,
                    &self.table.factors,
                    1,
                    self.log_size,
                    active,
                    range,
                )
            );
        }
        Ok(())
    }

    /// Evaluate the `size / 2` novel-basis coefficients in `rows` at the
    /// high coset (transform points `size/2 .. size`), writing the first
    /// `range` evaluations in place.
    ///
    /// This is the forward subtree rooted at node 3 (the high child of the
    /// root). The fused systematic encoder calls it directly on the inverse
    /// output so repair evaluations need neither a copy into the padded
    /// high half nor a full-domain workspace.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] unless `size >= 4` (reported as
    /// `expected: 4`) and `rows.len() == (size / 2) * row_len`.
    ///
    /// # Panics
    /// Panics unless `range.start <= range.end <= size / 2`, or under the
    /// row-geometry conditions documented by [`TransformPlan::forward_bytes`].
    pub fn forward_bytes_high_coset_range(
        &self,
        rows: &mut [u8],
        row_len: usize,
        range: Range<usize>,
    ) -> Result<(), TransformLengthError> {
        if self.log_size < 2 {
            return Err(TransformLengthError {
                expected: 4,
                got: self.size,
            });
        }
        let half = self.size / 2;
        Self::validate_row_len(row_len);
        let expected = half
            .checked_mul(row_len)
            .expect("transform byte length overflow");
        if rows.len() != expected {
            return Err(TransformLengthError {
                expected,
                got: rows.len(),
            });
        }
        assert!(
            range.start <= range.end && range.end <= half,
            "range out of bounds"
        );
        if !range.is_empty() {
            crate::core::kernel::dispatch_butterfly!(
                F,
                forward_bytes_range_node(
                    rows,
                    row_len,
                    &self.table.factors,
                    3,
                    self.log_size - 1,
                    range.start,
                    range.end,
                )
            );
        }
        Ok(())
    }

    /// Temporary rows required by [`TransformPlan::inverse_truncated_bytes`]
    /// for the given active coefficient prefix.
    ///
    /// # Panics
    /// Panics unless `1 <= active <= size`.
    #[must_use]
    pub fn inverse_truncated_scratch_rows(&self, active: usize) -> usize {
        assert!(
            active > 0 && active <= self.size,
            "active prefix out of range"
        );
        let mut row_count = self.size;
        while active < row_count {
            let half_rows = row_count / 2;
            if active > half_rows {
                return half_rows;
            }
            row_count = half_rows;
        }
        0
    }

    /// Truncated inverse transform: recovers the lowest `active` novel-
    /// basis coefficients from the evaluations in `rows`, using `scratch`
    /// as workspace.
    ///
    /// Rows `0..active` hold the coefficients on return; rows beyond
    /// `active` hold undefined intermediate values. Identical to a full
    /// inverse when `active == size` (no scratch needed then).
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `rows.len() == size * row_len` and `scratch.len() >=
    /// inverse_truncated_scratch_rows(active) * row_len`.
    ///
    /// # Panics
    /// Panics unless `1 <= active <= size`, or under the row-geometry
    /// conditions documented by [`TransformPlan::forward_bytes`].
    pub fn inverse_truncated_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
        active: usize,
        scratch: &mut [u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        let scratch_rows = self.inverse_truncated_scratch_rows(active);
        let expected = scratch_rows * row_len;
        if scratch.len() < expected {
            return Err(TransformLengthError {
                expected,
                got: scratch.len(),
            });
        }
        if self.log_size != 0 {
            crate::core::kernel::dispatch_butterfly!(
                F,
                inverse_truncated_bytes_node(
                    rows,
                    row_len,
                    &self.table.factors,
                    1,
                    self.log_size,
                    active,
                    scratch,
                )
            );
        }
        Ok(())
    }

    fn check_len(&self, got: usize) -> Result<(), TransformLengthError> {
        if got == self.size {
            Ok(())
        } else {
            Err(TransformLengthError {
                expected: self.size,
                got,
            })
        }
    }

    fn check_len_bytes(&self, got: usize, row_len: usize) -> Result<(), TransformLengthError> {
        Self::validate_row_len(row_len);
        let expected = self
            .size
            .checked_mul(row_len)
            .expect("transform byte length overflow");
        if got == expected {
            Ok(())
        } else {
            Err(TransformLengthError { expected, got })
        }
    }

    fn validate_row_len(row_len: usize) {
        assert_ne!(row_len, 0, "row length must be nonzero");
        assert_eq!(row_len % F::BYTES, 0, "partial trailing element");
    }
}

/// Process-wide shared plan cache, keyed by field and log size.
#[cfg(feature = "std")]
type SharedPlans = ::std::sync::LazyLock<
    ::std::sync::RwLock<
        ::std::collections::HashMap<
            (::core::any::TypeId, usize),
            Arc<dyn ::core::any::Any + Send + Sync>,
        >,
    >,
>;

#[cfg(feature = "std")]
static SHARED_PLANS: SharedPlans =
    ::std::sync::LazyLock::new(|| ::std::sync::RwLock::new(::std::collections::HashMap::new()));

/// An explicit, instance-level plan cache.
///
/// Unlike the process-wide [`TransformPlan::shared`], a `PlanCache` is owned
/// and droppable, which suits long-lived services that rebuild their codec
/// configuration or want deterministic teardown.
#[derive(Clone, Debug)]
pub struct PlanCache<F: ButterflyKernels> {
    /// Sparse by log size: `plans[log_size]` is the plan for `1 << log_size`.
    plans: Vec<Option<Arc<TransformPlan<F>>>>,
}

impl<F: ButterflyKernels> Default for PlanCache<F> {
    fn default() -> Self {
        Self { plans: Vec::new() }
    }
}

impl<F: ButterflyKernels> PlanCache<F> {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The plan for `size`, building and caching it on first use.
    ///
    /// # Errors
    /// Returns [`PlanError`] for an invalid or oversized `size`.
    pub fn shared(&mut self, size: usize) -> Result<Arc<TransformPlan<F>>, PlanError> {
        let log_size = validate_size::<F>(size)?;
        if self.plans.len() <= log_size {
            self.plans.resize_with(log_size + 1, || None);
        }
        let slot = &mut self.plans[log_size];
        if let Some(plan) = slot {
            return Ok(plan.clone());
        }
        let plan = Arc::new(TransformPlan::new(size)?);
        *slot = Some(plan.clone());
        Ok(plan)
    }
}

fn forward_node<E: Elem>(values: &mut [E], factors: &[E], node: usize, dimension: usize) {
    let half = values.len() / 2;
    let factor = factors[node];
    for position in 0..half {
        let low = values[position];
        let high = values[half + position];
        let left = low.add(factor.mul(high));
        values[position] = left;
        values[half + position] = left.add(high);
    }
    if dimension > 1 {
        let (left, right) = values.split_at_mut(half);
        forward_node(left, factors, node * 2, dimension - 1);
        forward_node(right, factors, node * 2 + 1, dimension - 1);
    }
}

fn inverse_node<E: Elem>(values: &mut [E], factors: &[E], node: usize, dimension: usize) {
    let half = values.len() / 2;
    if dimension > 1 {
        let (left, right) = values.split_at_mut(half);
        inverse_node(left, factors, node * 2, dimension - 1);
        inverse_node(right, factors, node * 2 + 1, dimension - 1);
    }
    let factor = factors[node];
    for position in 0..half {
        let left = values[position];
        let right = values[half + position];
        let high = left.add(right);
        values[position] = left.add(factor.mul(high));
        values[half + position] = high;
    }
}

fn forward_bytes_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
) {
    let half_bytes = rows.len() / 2;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    fused_forward_with::<F, B>(low_half, high_half, factor);
    if dimension > 1 {
        forward_bytes_node::<F, B>(low_half, factors, node * 2, dimension - 1);
        forward_bytes_node::<F, B>(high_half, factors, node * 2 + 1, dimension - 1);
    }
}

fn inverse_bytes_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
) {
    let half_bytes = rows.len() / 2;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    if dimension > 1 {
        inverse_bytes_node::<F, B>(low_half, factors, node * 2, dimension - 1);
        inverse_bytes_node::<F, B>(high_half, factors, node * 2 + 1, dimension - 1);
    }
    let factor = factors[node];
    fused_inverse_with::<F, B>(low_half, high_half, factor);
}

fn forward_bytes_selected_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    row_offset: usize,
    selected: &[usize],
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / row_len;
    let middle = row_offset + half_rows;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    fused_forward_with::<F, B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    let split = selected.partition_point(|&index| index < middle);
    if split != 0 {
        forward_bytes_selected_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            row_offset,
            &selected[..split],
        );
    }
    if split != selected.len() {
        forward_bytes_selected_node::<F, B>(
            high_half,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            middle,
            &selected[split..],
        );
    }
}

fn forward_bytes_range_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    range_start: usize,
    range_end: usize,
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / row_len;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    fused_forward_with::<F, B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    if range_start < half_rows {
        forward_bytes_range_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            range_start,
            range_end.min(half_rows),
        );
    }
    if range_end > half_rows {
        forward_bytes_range_node::<F, B>(
            high_half,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            range_start.saturating_sub(half_rows),
            range_end - half_rows,
        );
    }
}

fn forward_bytes_trunc_range_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    active: usize,
    range: Range<usize>,
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / row_len;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);

    // The active prefix fits in the low half: the high half's coefficients
    // are all zero, so the butterfly degenerates to a copy and only the
    // low half needs recursion.
    if active <= half_rows {
        if dimension <= 1 {
            if range.end > half_rows {
                high_half[..active * row_len].copy_from_slice(&low_half[..active * row_len]);
            }
            return;
        }
        let need_low = range.start < half_rows;
        let need_high = range.end > half_rows;
        if need_high {
            high_half[..active * row_len].copy_from_slice(&low_half[..active * row_len]);
        }
        if need_low {
            forward_bytes_trunc_range_node::<F, B>(
                low_half,
                row_len,
                factors,
                node * 2,
                dimension - 1,
                active,
                range.start..range.end.min(half_rows),
            );
        }
        if need_high {
            forward_bytes_trunc_range_node::<F, B>(
                high_half,
                row_len,
                factors,
                node * 2 + 1,
                dimension - 1,
                active,
                range.start.saturating_sub(half_rows)..range.end - half_rows,
            );
        }
        return;
    }

    let factor = factors[node];
    fused_forward_with::<F, B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    if range.start < half_rows {
        forward_bytes_trunc_range_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            half_rows,
            range.start..range.end.min(half_rows),
        );
    }
    if range.end > half_rows {
        forward_bytes_trunc_range_node::<F, B>(
            high_half,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            half_rows,
            range.start.saturating_sub(half_rows)..range.end - half_rows,
        );
    }
}

fn inverse_truncated_bytes_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    active: usize,
    scratch: &mut [u8],
) {
    let row_count = rows.len() / row_len;
    if row_count == 1 {
        return;
    }
    if active == row_count {
        inverse_bytes_node::<F, B>(rows, factors, node, dimension);
        return;
    }
    let half_rows = row_count / 2;
    let half_bytes = half_rows * row_len;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    if active <= half_rows {
        inverse_truncated_bytes_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            active,
            scratch,
        );
        return;
    }

    let right_active = active - half_rows;
    inverse_bytes_node::<F, B>(low_half, factors, node * 2, dimension - 1);

    // The recovered low half's known tail (coefficients right_active..half)
    // is zero by the truncation contract; subtract its contribution to the
    // high half's evaluations before recursing.
    let tail_start = right_active * row_len;
    {
        let known_tail = &mut scratch[..half_bytes];
        known_tail.fill(0);
        known_tail[tail_start..].copy_from_slice(&low_half[tail_start..]);
        forward_bytes_range_node::<F, B>(
            known_tail,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            0,
            right_active,
        );
        for (evaluation, contribution) in high_half[..tail_start]
            .iter_mut()
            .zip(&known_tail[..tail_start])
        {
            *evaluation ^= contribution;
        }
    }
    inverse_truncated_bytes_node::<F, B>(
        high_half,
        row_len,
        factors,
        node * 2 + 1,
        dimension - 1,
        right_active,
        scratch,
    );

    let factor = factors[node];
    let active_bytes = right_active * row_len;
    fused_inverse_with::<F, B>(
        &mut low_half[..active_bytes],
        &mut high_half[..active_bytes],
        factor,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factors::{
        NormalizedSubspacePolynomial, bit_basis, element_from_index, subspace_polynomials,
    };
    use fgf::{Gf8, Gf16};

    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn random_elements<F: Field>(state: &mut u32, count: usize) -> Vec<F::Elem> {
        (0..count)
            .map(|_| {
                let mut bytes = [0u8; 8];
                for byte in &mut bytes[..F::BYTES] {
                    *byte = lcg(state).to_le_bytes()[1];
                }
                F::read(&bytes[..F::BYTES])
            })
            .collect()
    }

    /// Naive `O(n²)` novel-basis evaluation: `f(x) = Σ_j a_j·X_j(x)` with
    /// `X_j(x) = Π_{b ∈ bits(j)} W̄_b(x)`.
    fn direct_evaluate<F: ButterflyKernels>(
        polynomials: &[NormalizedSubspacePolynomial<F::Elem>],
        coefficients: &[F::Elem],
        point: F::Elem,
    ) -> F::Elem {
        let normalized: Vec<F::Elem> = polynomials
            .iter()
            .map(|polynomial| polynomial.evaluate(point))
            .collect();
        let mut result = F::Elem::ZERO;
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let mut basis_value = F::Elem::ONE;
            for (bit, &value) in normalized.iter().enumerate() {
                if index & (1 << bit) != 0 {
                    basis_value = basis_value.mul(value);
                }
            }
            result = result.add(coefficient.mul(basis_value));
        }
        result
    }

    #[test]
    fn validates_sizes_and_lengths() {
        fn check<F: ButterflyKernels>(max_log: u32) {
            assert_eq!(
                TransformPlan::<F>::new(0).unwrap_err(),
                PlanError::InvalidSize { size: 0 }
            );
            assert_eq!(
                TransformPlan::<F>::new(3).unwrap_err(),
                PlanError::InvalidSize { size: 3 }
            );
            assert!(TransformPlan::<F>::new(1 << max_log).is_ok());
            let cap = (F::BITS as usize).min(MAX_LOG_SIZE);
            assert_eq!(
                TransformPlan::<F>::new(1 << (cap + 1)).unwrap_err(),
                PlanError::DomainTooLarge {
                    log_size: cap + 1,
                    cap,
                }
            );
            let plan = TransformPlan::<F>::new(8).unwrap();
            let mut short = random_elements::<F>(&mut 7, 4);
            assert_eq!(
                plan.forward(&mut short),
                Err(TransformLengthError {
                    expected: 8,
                    got: 4
                })
            );
        }
        check::<Gf8>(8);
        check::<Gf16>(16);
    }

    #[test]
    fn transform_matches_direct_novel_basis_evaluation() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 0..=5usize {
                let size = 1 << log_size;
                let coefficients = random_elements::<F>(state, size);
                let basis = bit_basis::<F>(log_size);
                let polynomials = subspace_polynomials(&basis).unwrap();
                let expected: Vec<F::Elem> = (0..size)
                    .map(|point| {
                        direct_evaluate::<F>(
                            &polynomials,
                            &coefficients,
                            element_from_index::<F>(point),
                        )
                    })
                    .collect();
                let mut actual = coefficients;
                TransformPlan::<F>::new(size)
                    .unwrap()
                    .forward(&mut actual)
                    .unwrap();
                assert_eq!(actual, expected, "forward diverged at size {size}");
            }
        }
        check::<Gf8>(&mut 0x1234_5678);
        check::<Gf16>(&mut 0x9abc_def0);
    }

    #[test]
    fn forward_inverse_roundtrips() {
        fn check<F: ButterflyKernels>(state: &mut u32, max_log: usize) {
            for log_size in 0..=max_log {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let mut values = random_elements::<F>(state, size);
                let expected = values.clone();
                plan.forward(&mut values).unwrap();
                plan.inverse(&mut values).unwrap();
                assert_eq!(values, expected, "roundtrip failed at size {size}");
            }
        }
        check::<Gf8>(&mut 11, 8);
        check::<Gf16>(&mut 13, 10);
    }

    /// Byte-domain transform must match the element-domain transform lane
    /// by lane: byte lane `l` of every row transforms together.
    #[test]
    fn bytes_match_element_domain_per_lane() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            const LANES: usize = 3;
            let row_len = LANES * F::BYTES;
            for log_size in 1..=6usize {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let lanes: Vec<Vec<F::Elem>> = (0..LANES)
                    .map(|_| random_elements::<F>(state, size))
                    .collect();
                let mut rows = ::alloc::vec![0u8; size * row_len];
                for (lane, elements) in lanes.iter().enumerate() {
                    for (row, &element) in elements.iter().enumerate() {
                        let start = row * row_len + lane * F::BYTES;
                        F::write(&mut rows[start..start + F::BYTES], element);
                    }
                }
                let original = rows.clone();
                plan.forward_bytes(&mut rows, row_len).unwrap();
                for (lane, elements) in lanes.iter().enumerate() {
                    let mut expected = elements.clone();
                    plan.forward(&mut expected).unwrap();
                    for (row, &element) in expected.iter().enumerate() {
                        let start = row * row_len + lane * F::BYTES;
                        let mut packed = [0u8; 8];
                        F::write(&mut packed[..F::BYTES], element);
                        assert_eq!(
                            &rows[start..start + F::BYTES],
                            &packed[..F::BYTES],
                            "lane {lane} row {row} diverged at size {size}"
                        );
                    }
                }
                plan.inverse_bytes(&mut rows, row_len).unwrap();
                assert_eq!(rows, original, "byte roundtrip failed at size {size}");
            }
        }
        check::<Gf8>(&mut 17);
        check::<Gf16>(&mut 19);
    }

    #[test]
    fn derivative_matches_unit_basis_formula() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=5usize {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let basis_index = (lcg(state) as usize % (size - 1)) + 1;
                let mut coefficients = ::alloc::vec![F::Elem::ZERO; size];
                coefficients[basis_index] = random_elements::<F>(state, 1)[0];
                let mut expected = ::alloc::vec![F::Elem::ZERO; size];
                let mut remaining = basis_index;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    expected[basis_index ^ (1 << bit)] = expected[basis_index ^ (1 << bit)]
                        .add(plan.table.derivative_factors[bit].mul(coefficients[basis_index]));
                    remaining &= remaining - 1;
                }
                let mut derivative = ::alloc::vec![F::Elem::ZERO; size];
                plan.derivative(&coefficients, &mut derivative).unwrap();
                assert_eq!(derivative, expected, "derivative diverged at size {size}");
            }
        }
        check::<Gf8>(&mut 23);
        check::<Gf16>(&mut 29);
    }

    /// A constant polynomial (only `X_0` nonzero) has zero derivative.
    #[test]
    fn derivative_annihilates_constants() {
        fn check<F: ButterflyKernels>() {
            let plan = TransformPlan::<F>::new(16).unwrap();
            let mut coefficients = ::alloc::vec![F::Elem::ZERO; 16];
            coefficients[0] = F::Elem::ONE;
            let mut derivative = ::alloc::vec![F::Elem::ONE; 16];
            plan.derivative(&coefficients, &mut derivative).unwrap();
            assert!(derivative.iter().all(|element| element.is_zero()));
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    #[test]
    fn derivative_bytes_match_element_domain() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            let size = 16;
            let plan = TransformPlan::<F>::new(size).unwrap();
            let coefficients = random_elements::<F>(state, size);
            let row_len = 2 * F::BYTES;
            let mut coefficient_rows = ::alloc::vec![0u8; size * row_len];
            for (row, &element) in coefficients.iter().enumerate() {
                F::write(
                    &mut coefficient_rows[row * row_len..row * row_len + F::BYTES],
                    element,
                );
            }
            let mut derivative_rows = ::alloc::vec![0u8; size * row_len];
            plan.derivative_bytes(&coefficient_rows, row_len, &mut derivative_rows)
                .unwrap();
            let mut expected = ::alloc::vec![F::Elem::ZERO; size];
            plan.derivative(&coefficients, &mut expected).unwrap();
            for (row, &element) in expected.iter().enumerate() {
                let mut packed = [0u8; 8];
                F::write(&mut packed[..F::BYTES], element);
                assert_eq!(
                    &derivative_rows[row * row_len..row * row_len + F::BYTES],
                    &packed[..F::BYTES],
                    "byte derivative row {row} diverged"
                );
            }
        }
        check::<Gf8>(&mut 31);
        check::<Gf16>(&mut 37);
    }

    #[test]
    fn with_basis_constructs_and_roundtrips() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            // Reversed independent prefix: different domain ordering, still
            // a valid transform.
            let mut basis = bit_basis::<F>(4);
            basis.reverse();
            let plan = TransformPlan::<F>::with_basis(16, &basis).unwrap();
            let mut values = random_elements::<F>(state, 16);
            let expected = values.clone();
            plan.forward(&mut values).unwrap();
            plan.inverse(&mut values).unwrap();
            assert_eq!(values, expected);
            // Point 2^j is the basis element itself.
            for (j, &element) in basis.iter().enumerate() {
                assert_eq!(plan.point_element(1 << j), element);
            }
            // Dependent and short bases are rejected.
            let dependent = [basis[0], basis[0]];
            assert_eq!(
                TransformPlan::<F>::with_basis(4, &dependent).unwrap_err(),
                PlanError::DependentBasis
            );
            assert_eq!(
                TransformPlan::<F>::with_basis(16, &basis[..2]).unwrap_err(),
                PlanError::BasisTooShort { needed: 4, got: 2 }
            );
        }
        check::<Gf8>(&mut 41);
        check::<Gf16>(&mut 43);
    }

    #[test]
    fn point_element_matches_bit_pattern() {
        fn check<F: ButterflyKernels>() {
            let plan = TransformPlan::<F>::new(32).unwrap();
            for index in 0..32 {
                assert_eq!(plan.point_element(index), element_from_index::<F>(index));
            }
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    #[cfg(feature = "std")]
    #[test]
    fn shared_plans_are_actually_shared() {
        fn check<F: ButterflyKernels>()
        where
            F::Elem: Send + Sync,
        {
            let first = TransformPlan::<F>::shared(64).unwrap();
            let second = TransformPlan::<F>::shared(64).unwrap();
            assert!(Arc::ptr_eq(&first, &second));
            let mut cache = PlanCache::<F>::new();
            let a = cache.shared(64).unwrap();
            let b = cache.shared(64).unwrap();
            assert!(Arc::ptr_eq(&a, &b));
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    // ------------------------------------------------------------------
    // P3 execution variants
    // ------------------------------------------------------------------

    fn pack_elements<F: Field>(elements: &[F::Elem]) -> Vec<u8> {
        let mut rows = ::alloc::vec![0u8; elements.len() * F::BYTES];
        for (row, &element) in elements.iter().enumerate() {
            F::write(&mut rows[row * F::BYTES..(row + 1) * F::BYTES], element);
        }
        rows
    }

    fn unpack_row<F: Field>(rows: &[u8], index: usize) -> F::Elem {
        F::read(&rows[index * F::BYTES..(index + 1) * F::BYTES])
    }

    #[test]
    fn forward_selected_matches_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=6usize {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let coefficients = random_elements::<F>(state, size);
                let mut reference = coefficients.clone();
                plan.forward(&mut reference).unwrap();
                let mut subsets: Vec<Vec<usize>> = ::alloc::vec![
                    ::alloc::vec![],
                    ::alloc::vec![0],
                    ::alloc::vec![size - 1],
                    (0..size).step_by(3).collect(),
                    (0..size).collect(),
                ];
                if size >= 4 {
                    subsets.push(::alloc::vec![1, size / 2, size - 1]);
                }
                for selected in &subsets {
                    let mut rows = pack_elements::<F>(&coefficients);
                    plan.forward_bytes_selected(&mut rows, F::BYTES, selected)
                        .unwrap();
                    for &index in selected {
                        assert_eq!(
                            unpack_row::<F>(&rows, index),
                            reference[index],
                            "selected row {index} diverged at size {size}"
                        );
                    }
                }
            }
        }
        check::<Gf8>(&mut 47);
        check::<Gf16>(&mut 53);
    }

    #[test]
    fn forward_range_matches_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=6usize {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let coefficients = random_elements::<F>(state, size);
                let mut reference = coefficients.clone();
                plan.forward(&mut reference).unwrap();
                let ranges = [
                    0..size,
                    0..1,
                    size - 1..size,
                    size / 4..size / 2,
                    size / 2..size,
                ];
                for range in ranges {
                    let mut rows = pack_elements::<F>(&coefficients);
                    plan.forward_bytes_range(&mut rows, F::BYTES, range.clone())
                        .unwrap();
                    for index in range {
                        assert_eq!(
                            unpack_row::<F>(&rows, index),
                            reference[index],
                            "range row {index} diverged at size {size}"
                        );
                    }
                }
            }
        }
        check::<Gf8>(&mut 59);
        check::<Gf16>(&mut 61);
    }

    #[test]
    fn trunc_range_matches_padded_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 2..=6usize {
                let size = 1 << log_size;
                let half = size / 2;
                let plan = TransformPlan::<F>::new(size).unwrap();
                for active in [1, 3, half - 1, half, half + 1, size - 1, size] {
                    let mut coefficients = random_elements::<F>(state, active);
                    coefficients.resize(size, F::Elem::ZERO);
                    let mut reference = coefficients.clone();
                    plan.forward(&mut reference).unwrap();
                    let ranges = [0..size, 0..active.min(size), half..size];
                    for range in ranges {
                        let mut rows = pack_elements::<F>(&coefficients);
                        plan.forward_bytes_trunc_range(&mut rows, F::BYTES, active, range.clone())
                            .unwrap();
                        for index in range {
                            assert_eq!(
                                unpack_row::<F>(&rows, index),
                                reference[index],
                                "trunc row {index} diverged at size {size} active {active}"
                            );
                        }
                    }
                }
            }
        }
        check::<Gf8>(&mut 67);
        check::<Gf16>(&mut 71);
    }

    #[test]
    fn high_coset_matches_padded_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 2..=7usize {
                let size = 1 << log_size;
                let half = size / 2;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let mut coefficients = random_elements::<F>(state, half);
                coefficients.resize(size, F::Elem::ZERO);
                // High-coset input: the original half-length coefficients.
                let mut rows = pack_elements::<F>(&coefficients[..half]);
                let mut reference = coefficients;
                plan.forward(&mut reference).unwrap();
                plan.forward_bytes_high_coset_range(&mut rows, F::BYTES, 0..half)
                    .unwrap();
                for index in 0..half {
                    assert_eq!(
                        unpack_row::<F>(&rows, index),
                        reference[half + index],
                        "high-coset row {index} diverged at size {size}"
                    );
                }
            }
            // Sizes below 4 are rejected.
            let mut one_row = ::alloc::vec![0u8; F::BYTES];
            assert!(
                TransformPlan::<F>::new(2)
                    .unwrap()
                    .forward_bytes_high_coset_range(&mut one_row, F::BYTES, 0..1)
                    .is_err()
            );
        }
        check::<Gf8>(&mut 73);
        check::<Gf16>(&mut 79);
    }

    #[test]
    fn inverse_truncated_recovers_active_prefix() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=7usize {
                let size = 1 << log_size;
                let half = size / 2;
                let plan = TransformPlan::<F>::new(size).unwrap();
                for active in [1, half.max(1), (half + 1).min(size), size] {
                    let mut coefficients = random_elements::<F>(state, active);
                    coefficients.resize(size, F::Elem::ZERO);
                    let mut evaluations = coefficients.clone();
                    plan.forward(&mut evaluations).unwrap();
                    let mut rows = pack_elements::<F>(&evaluations);
                    let scratch_rows = plan.inverse_truncated_scratch_rows(active);
                    let mut scratch = ::alloc::vec![0u8; scratch_rows * F::BYTES];
                    plan.inverse_truncated_bytes(&mut rows, F::BYTES, active, &mut scratch)
                        .unwrap();
                    for (index, &coefficient) in coefficients.iter().enumerate().take(active) {
                        assert_eq!(
                            unpack_row::<F>(&rows, index),
                            coefficient,
                            "truncated inverse row {index} diverged at size {size} active {active}"
                        );
                    }
                    // Undersized scratch is reported.
                    if scratch_rows > 0 {
                        let mut short_scratch = ::alloc::vec![0u8; (scratch_rows - 1) * F::BYTES];
                        let mut rows = pack_elements::<F>(&evaluations);
                        assert!(
                            plan.inverse_truncated_bytes(
                                &mut rows,
                                F::BYTES,
                                active,
                                &mut short_scratch,
                            )
                            .is_err()
                        );
                    }
                }
            }
        }
        check::<Gf8>(&mut 83);
        check::<Gf16>(&mut 89);
    }

    #[test]
    #[should_panic(expected = "sorted and unique")]
    fn forward_selected_rejects_unsorted() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let mut rows = [0u8; 16];
        let _ = plan.forward_bytes_selected(&mut rows, 2, &[5, 2]);
    }

    #[test]
    #[should_panic(expected = "range out of bounds")]
    fn forward_range_rejects_out_of_bounds() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let mut rows = [0u8; 16];
        let _ = plan.forward_bytes_range(&mut rows, 2, 0..9);
    }
    #[test]
    #[should_panic(expected = "row length must be nonzero")]
    fn byte_rows_reject_zero_width_before_walking() {
        let plan = TransformPlan::<Gf8>::new(4).unwrap();
        let _ = plan.forward_bytes_selected(&mut [], 0, &[0]);
    }

    #[test]
    #[should_panic(expected = "transform byte length overflow")]
    fn byte_rows_reject_unrepresentable_geometry() {
        let plan = TransformPlan::<Gf8>::new(4).unwrap();
        let row_len = 1usize << (usize::BITS - 2);
        let _ = plan.forward_bytes(&mut [], row_len);
    }
}
