//! Additive-FFT plans and in-place execution models.
//!
//! A [`TransformPlan`] evaluates polynomials over a power-of-two additive
//! domain of a binary field and interpolates them back. Both directions
//! consume and produce coefficients in the novel basis
//! `X_i(x) = ∏_j W̄_j(x)^{bit_j(i)}`; [`crate::basis`] converts between
//! that and the monomial basis. Evaluation points are enumerated in basis
//! order: point `i` is the coset shift plus the XOR of the basis elements
//! at the set bits of `i` ([`TransformPlan::point_element`]). Under the
//! default bit basis, point `i` is the field element whose little-endian
//! bytes hold the value `i`.
//!
//! # Layout
//!
//! Element methods operate on `size` field elements. Byte-row methods
//! operate on `size` rows of `row_len` bytes: one row per transform point,
//! each row a vector of packed little-endian field elements whose lanes
//! undergo the same transform independently. Plans own their twiddle and
//! derivative tables; every execution model runs in place and allocates
//! nothing. Byte-row geometry — nonzero row length, whole elements,
//! representable total length — is validated before the first row is
//! touched. Restricted selected, range, and truncated models define only
//! their documented output rows; the remaining rows hold undefined
//! intermediate values.
//!
//! ```
//! use butterfly_fft::TransformPlan;
//! use fgf::{Gf16, gf16};
//!
//! let plan = TransformPlan::<Gf16>::new(4).unwrap();
//! let mut values = [
//!     gf16::Elem::from_raw(0x1234),
//!     gf16::Elem::from_raw(0xabcd),
//!     gf16::Elem::from_raw(0x0108),
//!     gf16::Elem::from_raw(0xffff),
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

use crate::kernel::{Backend, ButterflyKernels, backend_for};

pub use crate::error::{PlanError, TransformLengthError};

pub(crate) mod factors;
mod walk;

use factors::FactorTable;
use walk::{
    derivative_into_bytes_overwrite_node, forward_bytes_node, forward_bytes_range_node,
    forward_bytes_selected_node, forward_bytes_truncated_range_node, forward_node,
    inverse_bytes_node, inverse_bytes_truncated_node, inverse_node,
};

/// Largest supported transform dimension: `2^MAX_LOG_SIZE` points.
///
/// Conservative cap keeping a factor table to a few megabytes per field;
/// the field's own extension degree caps smaller fields further.
pub const MAX_LOG_SIZE: usize = 20;

/// Row-length floor past which the overwrite-first derivative schedule wins
/// on the scalar and GFNI backends; the measurement lives in
/// `BENCHMARKS.md` under "Derivative schedules".
const DERIVATIVE_OVERWRITE_ROW_MIN: usize = 1 << 10;

/// Row-length floor past which GFNI's register-blocked gather beats
/// overwrite-first; the measurement lives in `BENCHMARKS.md` under
/// "Derivative schedules".
const DERIVATIVE_GATHER_ROW_MIN: usize = 1 << 16;

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
    /// Whether every derivative factor is zero or one, letting the byte
    /// derivative sweep run on plain XORs (true for the Cantor basis).
    derivative_factors_trivial: bool,
    /// The ordered-basis prefix `β_0 … β_{log_size-1}` defining the domain.
    basis: Vec<F::Elem>,
    /// Coset shift `α`: zero for a plain subspace, nonzero for an affine
    /// coset `α + span(basis)`. Only the twiddle table and the vanishing
    /// polynomial's constant term depend on it.
    shift: F::Elem,
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
    /// The twiddle table has the same shape as a subspace plan's; only the
    /// root coset shift changes, so every execution model — truncated and
    /// selected variants included — runs unmodified over the coset.
    /// Coset plans are not cached by `TransformPlan::shared`, which covers
    /// the unshifted bit-basis plans every consumer shares; a consumer
    /// working one coset across many payloads should hold its plan.
    ///
    /// # Errors
    /// As [`TransformPlan::with_basis`].
    pub fn with_shift(size: usize, basis: &[F::Elem], shift: F::Elem) -> Result<Self, PlanError> {
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
            shift,
        ))
    }

    fn construct(size: usize, log_size: usize, basis: Vec<F::Elem>) -> Result<Self, PlanError> {
        let table =
            FactorTable::build(log_size, &basis, F::Elem::ZERO).ok_or(PlanError::DependentBasis)?;
        Ok(Self::from_parts(
            size,
            log_size,
            table,
            basis,
            F::Elem::ZERO,
        ))
    }

    fn from_parts(
        size: usize,
        log_size: usize,
        table: FactorTable<F>,
        basis: Vec<F::Elem>,
        shift: F::Elem,
    ) -> Self {
        let prepared_derivative_factors: ::alloc::vec::Vec<Coeff<F>> = table
            .derivative_factors
            .iter()
            .copied()
            .map(Coeff::<F>::new)
            .collect();
        let derivative_factors_trivial = prepared_derivative_factors
            .iter()
            .all(|factor| factor.value().is_zero() || factor.value().is_one());
        Self {
            size,
            log_size,
            table,
            prepared_derivative_factors,
            derivative_factors_trivial,
            basis,
            shift,
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
    /// Crate-private: exposed to the `tuning` module, reachable externally
    /// only through the `internals` facade.
    // Reachable only through the `internals` facade.
    #[allow(dead_code)]
    #[must_use]
    pub(crate) const fn table(&self) -> &FactorTable<F> {
        &self.table
    }

    /// The field element of transform point `index`: the coset shift plus
    /// the XOR of the basis elements at the set bits of `index`.
    ///
    /// # Panics
    /// Panics if `index >= size`: the basis lookup runs past the domain's
    /// basis prefix.
    #[must_use]
    pub fn point_element(&self, index: usize) -> F::Elem {
        debug_assert!(index < self.size);
        let mut element = self.shift;
        let mut remaining = index;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            element = element.add(self.basis[bit]);
            remaining &= remaining - 1;
        }
        element
    }

    /// Every evaluation point of the domain, in transform order.
    ///
    /// Allocates `size` elements.
    #[must_use]
    pub fn points(&self) -> Vec<F::Elem> {
        (0..self.size)
            .map(|index| self.point_element(index))
            .collect()
    }

    /// The coset shift `α` of this plan's domain.
    ///
    /// Zero for a plain additive subspace; nonzero for an affine coset
    /// `α + span(basis)`.
    #[must_use]
    pub const fn shift(&self) -> F::Elem {
        self.shift
    }

    /// Dense monomial coefficients of the domain vanishing polynomial
    /// `G(X) = ∏_{α ∈ domain}(X − α)`, low-to-high (length `size + 1`).
    ///
    /// For a subspace plan (`shift == 0`) this is the monic subspace
    /// polynomial `W_L(X)`, whose only nonzero terms are at Frobenius
    /// degrees `X^(2^j)`. For an affine-coset plan it is the translate
    /// `W_L(X) + W_L(shift)`, identical except for a nonzero constant term.
    /// The result is always monic of degree `size`.
    ///
    /// Allocates `size + 1` elements.
    #[must_use]
    pub fn vanishing_polynomial(&self) -> Vec<F::Elem> {
        let sparse = factors::monic_subspace_polynomial(&self.basis);
        let mut dense = ::alloc::vec![F::Elem::ZERO; self.size + 1];
        for (exponent, &coefficient) in sparse.iter().enumerate() {
            dense[1 << exponent] = coefficient;
        }
        // Affine coset: G(X) = W_L(X + shift) = W_L(X) + W_L(shift), using
        // the linearity of the linearized W_L. The subspace case (shift 0)
        // leaves the constant term zero, since W_L vanishes at the origin.
        if !self.shift.is_zero() {
            dense[0] = factors::evaluate_linearized(&sparse, self.shift);
        }
        dense
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
    /// # Panics
    /// Panics if `row_len` is zero, holds a partial trailing element, or the
    /// complete byte length is not representable by [`usize`].
    pub fn forward_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        if self.log_size != 0 {
            crate::kernel::dispatch_butterfly!(
                F,
                forward_bytes_node(rows, &self.table.factors, 1, self.log_size)
            );
        }
        Ok(())
    }

    /// Inverse transform over interleaved byte rows.
    ///
    /// # Errors
    /// As [`TransformPlan::forward_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::forward_bytes`].
    pub fn inverse_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        if self.log_size != 0 {
            crate::kernel::dispatch_butterfly!(
                F,
                inverse_bytes_node(rows, &self.table.factors, 1, self.log_size)
            );
        }
        Ok(())
    }

    /// Formal derivative of the novel-basis coefficient vector, out of
    /// place: `X_i'(x) = Σ_{j ∈ bits(i)} W̄_j'·X_{i ^ 2^j}(x)`.
    ///
    /// `derivative` is write-only: its prior contents are ignored.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] unless both slices hold `size`
    /// elements.
    pub fn derivative_into(
        &self,
        derivative: &mut [F::Elem],
        coefficients: &[F::Elem],
    ) -> Result<(), TransformLengthError> {
        self.check_len(derivative.len())?;
        self.check_len(coefficients.len())?;
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
    /// `derivative` is write-only and must not overlap `coefficients`.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless both
    /// buffers hold `size` rows of `row_len` bytes.
    ///
    /// # Panics
    /// Under the row-geometry conditions documented by
    /// [`TransformPlan::forward_bytes`].
    pub fn derivative_into_bytes(
        &self,
        derivative: &mut [u8],
        row_len: usize,
        coefficients: &[u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(coefficients.len(), row_len)?;
        self.check_len_bytes(derivative.len(), row_len)?;
        let selected = backend_for::<F>();
        let measured = matches!(selected, Backend::Scalar | Backend::V3GfniCrypto);
        if selected == Backend::V3GfniCrypto
            && (row_len >= DERIVATIVE_GATHER_ROW_MIN
                || (F::BYTES == 1 && row_len >= DERIVATIVE_OVERWRITE_ROW_MIN))
        {
            self.derivative_into_bytes_gather(derivative, row_len, coefficients)?;
        } else if measured && F::BYTES <= 2 && row_len >= DERIVATIVE_OVERWRITE_ROW_MIN {
            self.derivative_into_bytes_overwrite(derivative, row_len, coefficients)?;
        } else {
            self.derivative_into_bytes_sweep(derivative, row_len, coefficients)?;
        }
        Ok(())
    }

    fn derivative_into_bytes_sweep_unchecked(
        &self,
        derivative: &mut [u8],
        row_len: usize,
        coefficients: &[u8],
    ) {
        let factors = &self.prepared_derivative_factors;
        derivative.fill(0);
        for index in 1..self.size {
            let width = index & index.wrapping_neg();
            let level = width.trailing_zeros() as usize;
            let factor = &factors[level];
            if factor.value().is_zero() {
                continue;
            }
            let span = width * row_len;
            let source_start = index * row_len;
            // The destination block sits just below the source block; both
            // stay within the validated buffer geometry.
            let destination = &mut derivative[source_start - span..source_start];
            let source = &coefficients[source_start..source_start + span];
            debug_assert!(!factor.value().is_zero());
            if self.derivative_factors_trivial {
                ops::add_assign::<F>(destination, source);
            } else {
                ops::mul_add_with::<F>(destination, factor, source);
            }
        }
    }

    /// Tuning-only exact derivative using the source-oriented lowbit sweep.
    ///
    /// This bypasses production crossover selection for benchmark controls.
    /// It has the same output and geometry contract as
    /// [`TransformPlan::derivative_into_bytes`] and allocates nothing.
    ///
    /// # Errors
    /// As [`TransformPlan::derivative_into_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::derivative_into_bytes`].
    pub(crate) fn derivative_into_bytes_sweep(
        &self,
        derivative: &mut [u8],
        row_len: usize,
        coefficients: &[u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(coefficients.len(), row_len)?;
        self.check_len_bytes(derivative.len(), row_len)?;
        self.derivative_into_bytes_sweep_unchecked(derivative, row_len, coefficients);
        Ok(())
    }

    /// Add the formal derivative to its coefficients in place: `c <- c + D(c)`.
    ///
    /// This augmented derivative is useful when the rows are subsequently
    /// evaluated only at roots of the represented polynomial: at any root
    /// `x`, `c(x) + D(c)(x) = D(c)(x)`. It is not the coefficient vector of
    /// the exact formal derivative; use [`TransformPlan::derivative_into_bytes`]
    /// when every derivative coefficient is required.
    ///
    /// The ascending sweep is safe in place because each source block lies
    /// above its destination and has not yet been modified.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless `rows` holds
    /// `size` rows of `row_len` bytes.
    ///
    /// # Panics
    /// Under the row-geometry conditions documented by
    /// [`TransformPlan::forward_bytes`].
    pub fn derivative_plus_identity_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        let factors = &self.prepared_derivative_factors;
        for index in 1..self.size {
            let width = index & index.wrapping_neg();
            let level = width.trailing_zeros() as usize;
            let factor = &factors[level];
            if factor.value().is_zero() {
                continue;
            }
            let span = width * row_len;
            let source_start = index * row_len;
            let (below, source_and_above) = rows.split_at_mut(source_start);
            let destination = &mut below[source_start - span..];
            let source = &source_and_above[..span];
            ops::mul_add_with::<F>(destination, factor, source);
        }
        Ok(())
    }

    /// Tuning-only exact derivative using one multi-source gather per output row.
    ///
    /// This exposes an alternative execution schedule for benchmark crossover
    /// measurements. It has the same output and geometry contract as
    /// [`TransformPlan::derivative_into_bytes`] and allocates nothing.
    ///
    /// # Errors
    /// As [`TransformPlan::derivative_into_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::derivative_into_bytes`].
    pub(crate) fn derivative_into_bytes_gather(
        &self,
        derivative: &mut [u8],
        row_len: usize,
        coefficients: &[u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(coefficients.len(), row_len)?;
        self.check_len_bytes(derivative.len(), row_len)?;

        self.derivative_into_bytes_gather_unchecked(derivative, row_len, coefficients);
        Ok(())
    }

    fn derivative_into_bytes_gather_unchecked(
        &self,
        derivative: &mut [u8],
        row_len: usize,
        coefficients: &[u8],
    ) {
        let mut values = [F::Elem::ZERO; MAX_LOG_SIZE];
        let mut sources: [&[u8]; MAX_LOG_SIZE] = [&[]; MAX_LOG_SIZE];
        for destination in 0..self.size {
            let start = destination * row_len;
            let output = &mut derivative[start..start + row_len];
            output.fill(0);

            let mut count = 0;
            for level in 0..self.log_size {
                let bit = 1 << level;
                let factor = self.table.derivative_factors[level];
                if destination & bit != 0 || factor.is_zero() {
                    continue;
                }
                let source_start = (destination | bit) * row_len;
                values[count] = factor;
                sources[count] = &coefficients[source_start..source_start + row_len];
                count += 1;
            }
            ops::mul_add_gather::<F>(output, &values[..count], &sources[..count]);
        }
    }

    /// Tuning-only exact derivative initialized from first contributions.
    ///
    /// This exposes an alternative execution schedule for benchmark crossover
    /// measurements. It has the same output and geometry contract as
    /// [`TransformPlan::derivative_into_bytes`] and allocates nothing.
    ///
    /// # Errors
    /// As [`TransformPlan::derivative_into_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::derivative_into_bytes`].
    pub(crate) fn derivative_into_bytes_overwrite(
        &self,
        derivative: &mut [u8],
        row_len: usize,
        coefficients: &[u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(coefficients.len(), row_len)?;
        self.check_len_bytes(derivative.len(), row_len)?;
        derivative_into_bytes_overwrite_node::<F>(
            derivative,
            coefficients,
            &self.prepared_derivative_factors,
            self.log_size,
        );
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
            crate::kernel::dispatch_butterfly!(
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
            crate::kernel::dispatch_butterfly!(
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
    pub fn forward_bytes_truncated_range(
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
            crate::kernel::dispatch_butterfly!(
                F,
                forward_bytes_truncated_range_node(
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
            crate::kernel::dispatch_butterfly!(
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

    /// Temporary rows required by
    /// [`TransformPlan::inverse_bytes_truncated_scratch`]
    /// for the given active coefficient prefix.
    ///
    /// # Panics
    /// Panics unless `1 <= active <= size`.
    #[must_use]
    pub fn inverse_bytes_truncated_scratch_rows(&self, active: usize) -> usize {
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
    /// The evaluations in `rows` must be the forward transform of a
    /// polynomial whose novel-basis coefficients `active..size` are zero —
    /// the counterpart of the zero padding
    /// [`TransformPlan::forward_bytes_truncated_range`] requires. The premise is
    /// a property of the represented polynomial and cannot be validated
    /// from the buffer; arbitrary evaluations produce wrong coefficients
    /// without any diagnostic.
    ///
    /// # Errors
    /// Returns [`TransformLengthError`] (lengths in bytes) unless
    /// `rows.len() == size * row_len` and `scratch.len() >=
    /// inverse_bytes_truncated_scratch_rows(active) * row_len`.
    ///
    /// # Panics
    /// Panics unless `1 <= active <= size`, or under the row-geometry
    /// conditions documented by [`TransformPlan::forward_bytes`].
    pub fn inverse_bytes_truncated_scratch(
        &self,
        rows: &mut [u8],
        row_len: usize,
        active: usize,
        scratch: &mut [u8],
    ) -> Result<(), TransformLengthError> {
        self.check_len_bytes(rows.len(), row_len)?;
        let scratch_rows = self.inverse_bytes_truncated_scratch_rows(active);
        let expected = scratch_rows * row_len;
        if scratch.len() < expected {
            return Err(TransformLengthError {
                expected,
                got: scratch.len(),
            });
        }
        if self.log_size != 0 {
            crate::kernel::dispatch_butterfly!(
                F,
                inverse_bytes_truncated_node(
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
/// Unlike the process-wide `TransformPlan::shared`, a `PlanCache` is owned
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::factors::{
        NormalizedSubspacePolynomial, bit_basis, element_from_index, subspace_polynomials,
    };
    use ::alloc::vec;
    use fgf::{Gf8B, Gf8D, Gf16};

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
                F::decode(&bytes[..F::BYTES])
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

    /// Sweep ceiling for size-based tests: the exhaustive checks are
    /// superlinear in the size, so the miri run exercises the same
    /// recursion and boundary classes at smaller powers of two.
    fn log_cap(default: usize) -> usize {
        if cfg!(miri) { default.min(4) } else { default }
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
        check::<Gf8B>(8);
        check::<Gf16>(16);
    }

    #[test]
    fn transform_matches_direct_novel_basis_evaluation() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 0..=log_cap(5) {
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
        check::<Gf8B>(&mut 0x1234_5678);
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
        check::<Gf8B>(&mut 11, log_cap(8));
        check::<Gf16>(&mut 13, log_cap(10));
    }

    /// Byte-domain transform must match the element-domain transform lane
    /// by lane: byte lane `l` of every row transforms together.
    #[test]
    fn bytes_match_element_domain_per_lane() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            const LANES: usize = 3;
            let row_len = LANES * F::BYTES;
            for log_size in 1..=log_cap(6) {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let lanes: Vec<Vec<F::Elem>> = (0..LANES)
                    .map(|_| random_elements::<F>(state, size))
                    .collect();
                let mut rows = ::alloc::vec![0u8; size * row_len];
                for (lane, elements) in lanes.iter().enumerate() {
                    for (row, &element) in elements.iter().enumerate() {
                        let start = row * row_len + lane * F::BYTES;
                        F::encode(&mut rows[start..start + F::BYTES], element);
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
                        F::encode(&mut packed[..F::BYTES], element);
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
        check::<Gf8B>(&mut 17);
        check::<Gf16>(&mut 19);
    }

    #[test]
    fn derivative_matches_unit_basis_formula() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=log_cap(5) {
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
                plan.derivative_into(&mut derivative, &coefficients)
                    .unwrap();
                assert_eq!(derivative, expected, "derivative diverged at size {size}");
            }
        }
        check::<Gf8B>(&mut 23);
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
            plan.derivative_into(&mut derivative, &coefficients)
                .unwrap();
            assert!(derivative.iter().all(|element| element.is_zero()));
        }
        check::<Gf8B>();
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
                F::encode(
                    &mut coefficient_rows[row * row_len..row * row_len + F::BYTES],
                    element,
                );
            }
            let mut derivative_rows = ::alloc::vec![0u8; size * row_len];
            plan.derivative_into_bytes(&mut derivative_rows, row_len, &coefficient_rows)
                .unwrap();
            let mut expected = ::alloc::vec![F::Elem::ZERO; size];
            plan.derivative_into(&mut expected, &coefficients).unwrap();
            for (row, &element) in expected.iter().enumerate() {
                let mut packed = [0u8; 8];
                F::encode(&mut packed[..F::BYTES], element);
                assert_eq!(
                    &derivative_rows[row * row_len..row * row_len + F::BYTES],
                    &packed[..F::BYTES],
                    "byte derivative row {row} diverged"
                );
            }
        }
        check::<Gf8B>(&mut 31);
        check::<Gf16>(&mut 37);
    }

    #[test]
    fn byte_derivative_paths_match_monomial_differentiation() {
        fn check<F: ButterflyKernels>(state: &mut u32, row_len: usize) {
            let size = 8;
            let plan = TransformPlan::<F>::new(size).unwrap();
            let coefficients = random_elements::<F>(state, size);

            let mut monomial = coefficients.clone();
            crate::basis::novel_to_monomial(&mut monomial, &plan).unwrap();
            let mut expected = vec![F::Elem::ZERO; size];
            for degree in (1..size).step_by(2) {
                expected[degree - 1] = monomial[degree];
            }
            crate::basis::monomial_to_novel(&mut expected, &plan).unwrap();

            let mut coefficient_rows = vec![0u8; size * row_len];
            for (row, &coefficient) in coefficient_rows
                .chunks_exact_mut(row_len)
                .zip(&coefficients)
            {
                for element in row.chunks_mut(F::BYTES) {
                    F::encode(element, coefficient);
                }
            }

            let mut derivative_rows = vec![0xA5; coefficient_rows.len()];
            plan.derivative_into_bytes(&mut derivative_rows, row_len, &coefficient_rows)
                .unwrap();
            let mut augmented_rows = coefficient_rows.clone();
            plan.derivative_plus_identity_bytes(&mut augmented_rows, row_len)
                .unwrap();

            for row in 0..size {
                let expected_derivative = expected[row];
                let expected_augmented = coefficients[row].add(expected_derivative);
                for element in derivative_rows[row * row_len..][..row_len].chunks(F::BYTES) {
                    assert_eq!(F::decode(element), expected_derivative);
                }
                for element in augmented_rows[row * row_len..][..row_len].chunks(F::BYTES) {
                    assert_eq!(F::decode(element), expected_augmented);
                }
            }
        }

        // Exercise overwrite-first with nontrivial factors on every measured
        // backend, and GFNI gather at its wide-row crossover.
        check::<Gf8B>(&mut 101, DERIVATIVE_OVERWRITE_ROW_MIN);
        check::<Gf16>(&mut 103, DERIVATIVE_OVERWRITE_ROW_MIN);
        check::<Gf16>(&mut 107, DERIVATIVE_GATHER_ROW_MIN);
    }

    /// The byte-derivative sweep must match the straightforward
    /// one-direction-at-a-time schedule (kept here as an independent oracle),
    /// over both the default basis (generic scaled steps) and the Cantor
    /// basis (trivial factors, plain-XOR steps), for even and odd dimension
    /// counts, several row lengths, and arbitrary prior destination contents.
    #[test]
    fn derivative_bytes_sweep_matches_per_direction_schedule() {
        fn reference<F: ButterflyKernels>(
            plan: &TransformPlan<F>,
            coefficients: &[u8],
            row_len: usize,
            derivative: &mut [u8],
        ) {
            derivative.fill(0);
            for (bit, factor) in plan.prepared_derivative_factors.iter().enumerate() {
                let half_len = (1 << bit) * row_len;
                let block_len = half_len * 2;
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
        }
        fn check_plan<F: ButterflyKernels>(
            state: &mut u32,
            label: &str,
            plan_for: impl Fn(usize) -> TransformPlan<F>,
        ) {
            for log_size in 0..=log_cap(usize::min(9, F::BITS as usize)) {
                let size = 1 << log_size;
                let plan = plan_for(size);
                for lane_count in [1usize, 3, 7] {
                    let row_len = lane_count * F::BYTES;
                    let coefficients: Vec<u8> = (0..size * row_len)
                        .map(|_| lcg(state).to_le_bytes()[1])
                        .collect();
                    // Garbage-prefilled destinations catch missing
                    // initialization; a zeroed one catches double adds.
                    for preset in [0xA5u8, 0x00] {
                        let mut actual = vec![preset; size * row_len];
                        plan.derivative_into_bytes(&mut actual, row_len, &coefficients)
                            .unwrap();
                        let mut expected = vec![preset; size * row_len];
                        reference(&plan, &coefficients, row_len, &mut expected);
                        assert_eq!(
                            actual, expected,
                            "{label} derivative diverged at log {log_size} lanes {lane_count} preset {preset:#x}"
                        );

                        let mut augmented = coefficients.clone();
                        plan.derivative_plus_identity_bytes(&mut augmented, row_len)
                            .unwrap();
                        let expected_augmented: Vec<u8> = coefficients
                            .iter()
                            .zip(&expected)
                            .map(|(&coefficient, &derivative)| coefficient ^ derivative)
                            .collect();
                        assert_eq!(
                            augmented, expected_augmented,
                            "{label} augmented derivative diverged at log {log_size} lanes {lane_count} preset {preset:#x}"
                        );
                    }
                }
            }
        }
        fn check<F: ButterflyKernels>(state: &mut u32)
        where
            <F as fgf::Field>::Elem: Send + Sync,
        {
            check_plan(state, "bit", |size| TransformPlan::<F>::new(size).unwrap());
            #[cfg(feature = "std")]
            check_plan(state, "cantor", |size| {
                let basis = crate::basis::cantor_basis::<F>().unwrap();
                let log = size.ilog2() as usize;
                TransformPlan::<F>::with_basis(size, &basis.elements()[..log]).unwrap()
            });
        }
        check::<Gf8B>(&mut 41);
        check::<Gf16>(&mut 43);
    }

    #[cfg(feature = "std")]
    #[test]
    fn derivative_production_crossovers_match_cantor_oracle() {
        fn check<F: ButterflyKernels>()
        where
            <F as Field>::Elem: Send + Sync,
        {
            let size = 8;
            let basis = crate::basis::cantor_basis::<F>().unwrap();
            let plan =
                TransformPlan::<F>::with_basis(size, &basis.elements()[..size.ilog2() as usize])
                    .unwrap();
            for row_len in [DERIVATIVE_OVERWRITE_ROW_MIN, DERIVATIVE_GATHER_ROW_MIN] {
                let coefficients: Vec<u8> = (0..size * row_len)
                    .map(|index| index.to_le_bytes()[0].wrapping_mul(73).wrapping_add(19))
                    .collect();
                let mut expected = vec![0u8; coefficients.len()];
                for source in 1..size {
                    let mut bits = source;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        let destination = source ^ (1 << bit);
                        let source_start = source * row_len;
                        let destination_start = destination * row_len;
                        for offset in 0..row_len {
                            expected[destination_start + offset] ^=
                                coefficients[source_start + offset];
                        }
                        bits &= bits - 1;
                    }
                }

                let mut actual = vec![0xA5; coefficients.len()];
                plan.derivative_into_bytes(&mut actual, row_len, &coefficients)
                    .unwrap();
                assert_eq!(actual, expected, "{} row length {row_len}", F::NAME);
            }
        }

        check::<Gf8B>();
        check::<Gf16>();
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
        check::<Gf8B>(&mut 41);
        check::<Gf8D>(&mut 43);
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
        check::<Gf8B>();
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
        check::<Gf8B>();
        check::<Gf8D>();
        check::<Gf16>();
    }

    // ------------------------------------------------------------------
    // P3 execution variants
    // ------------------------------------------------------------------

    fn pack_elements<F: Field>(elements: &[F::Elem]) -> Vec<u8> {
        let mut rows = ::alloc::vec![0u8; elements.len() * F::BYTES];
        for (row, &element) in elements.iter().enumerate() {
            F::encode(&mut rows[row * F::BYTES..(row + 1) * F::BYTES], element);
        }
        rows
    }

    fn unpack_row<F: Field>(rows: &[u8], index: usize) -> F::Elem {
        F::decode(&rows[index * F::BYTES..(index + 1) * F::BYTES])
    }

    #[test]
    fn forward_selected_matches_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=log_cap(6) {
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
        check::<Gf8B>(&mut 47);
        check::<Gf16>(&mut 53);
    }

    #[test]
    fn forward_range_matches_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=log_cap(6) {
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
        check::<Gf8B>(&mut 59);
        check::<Gf16>(&mut 61);
    }

    #[test]
    fn trunc_range_matches_padded_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 2..=log_cap(6) {
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
                        plan.forward_bytes_truncated_range(
                            &mut rows,
                            F::BYTES,
                            active,
                            range.clone(),
                        )
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
        check::<Gf8B>(&mut 67);
        check::<Gf16>(&mut 71);
    }

    #[test]
    fn high_coset_matches_padded_full() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 2..=log_cap(7) {
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
        check::<Gf8B>(&mut 73);
        check::<Gf16>(&mut 79);
    }

    #[test]
    fn inverse_truncated_recovers_active_prefix() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=log_cap(7) {
                let size = 1 << log_size;
                let half = size / 2;
                let plan = TransformPlan::<F>::new(size).unwrap();
                for active in [1, half.max(1), (half + 1).min(size), size] {
                    let mut coefficients = random_elements::<F>(state, active);
                    coefficients.resize(size, F::Elem::ZERO);
                    let mut evaluations = coefficients.clone();
                    plan.forward(&mut evaluations).unwrap();
                    let mut rows = pack_elements::<F>(&evaluations);
                    let scratch_rows = plan.inverse_bytes_truncated_scratch_rows(active);
                    let mut scratch = ::alloc::vec![0u8; scratch_rows * F::BYTES];
                    plan.inverse_bytes_truncated_scratch(&mut rows, F::BYTES, active, &mut scratch)
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
                            plan.inverse_bytes_truncated_scratch(
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
        check::<Gf8B>(&mut 83);
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
        let plan = TransformPlan::<Gf8B>::new(4).unwrap();
        let _ = plan.forward_bytes_selected(&mut [], 0, &[0]);
    }

    #[test]
    #[should_panic(expected = "transform byte length overflow")]
    fn byte_rows_reject_unrepresentable_geometry() {
        let plan = TransformPlan::<Gf8B>::new(4).unwrap();
        let row_len = 1usize << (usize::BITS - 2);
        let _ = plan.forward_bytes(&mut [], row_len);
    }

    /// Schoolbook polynomial product over dense monomial coefficients.
    fn poly_mul<F: ButterflyKernels>(a: &[F::Elem], b: &[F::Elem]) -> Vec<F::Elem> {
        let mut result = vec![F::Elem::ZERO; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                result[i + j] = result[i + j].add(ai.mul(bj));
            }
        }
        result
    }

    /// Horner evaluation, independent of the transform.
    fn horner_eval<F: ButterflyKernels>(coefficients: &[F::Elem], point: F::Elem) -> F::Elem {
        coefficients
            .iter()
            .rev()
            .fold(F::Elem::ZERO, |acc, &c| acc.mul(point).add(c))
    }

    /// Incremental `∏(X + point)` — independent oracle for the vanishing
    /// polynomial.
    fn product_vanishing<F: ButterflyKernels>(points: &[F::Elem]) -> Vec<F::Elem> {
        let mut product = vec![F::Elem::ONE];
        for &point in points {
            let factor = [point, F::Elem::ONE];
            product = poly_mul::<F>(&product, &factor);
        }
        product
    }

    #[test]
    fn vanishing_polynomial_vanishes_on_subspace_and_matches_product() {
        fn check<F: ButterflyKernels>(_state: &mut u32) {
            for log_size in 1..=log_cap(6) {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let g = plan.vanishing_polynomial();
                assert_eq!(g.len(), size + 1, "degree mismatch at size {size}");
                assert_eq!(g[size], F::Elem::ONE, "not monic at size {size}");
                for index in 0..size {
                    assert_eq!(
                        horner_eval::<F>(&g, plan.point_element(index)),
                        F::Elem::ZERO,
                        "G nonzero at point {index} size {size}"
                    );
                }
                let points: Vec<F::Elem> = (0..size).map(|i| plan.point_element(i)).collect();
                assert_eq!(
                    g,
                    product_vanishing::<F>(&points),
                    "subspace G mismatch size {size}"
                );
            }
        }
        check::<Gf8B>(&mut 11);
        check::<Gf16>(&mut 13);
    }

    #[test]
    fn vanishing_polynomial_vanishes_on_affine_coset_and_matches_product() {
        fn check<F: ButterflyKernels>(_state: &mut u32) {
            for log_size in 1..=log_cap(6) {
                let size = 1 << log_size;
                // A shift outside span(β_0..β_{log_size-1}): the next bit-basis
                // element, so the coset is distinct from the subspace.
                let mut shift_bytes = [0u8; 8];
                shift_bytes[0] = 1 << log_size;
                let shift = F::decode(&shift_bytes[..F::BYTES]);
                let basis = factors::bit_basis::<F>(log_size);
                let plan = TransformPlan::<F>::with_shift(size, &basis, shift).unwrap();
                let g = plan.vanishing_polynomial();
                assert_eq!(g.len(), size + 1);
                assert_eq!(g[size], F::Elem::ONE, "not monic at size {size}");
                // Affine coset has a nonzero constant term.
                assert!(!g[0].is_zero(), "affine G has zero constant at size {size}");
                for index in 0..size {
                    assert_eq!(
                        horner_eval::<F>(&g, plan.point_element(index)),
                        F::Elem::ZERO,
                        "affine G nonzero at point {index} size {size}"
                    );
                }
                let points: Vec<F::Elem> = (0..size).map(|i| plan.point_element(i)).collect();
                assert_eq!(
                    g,
                    product_vanishing::<F>(&points),
                    "affine G mismatch size {size}"
                );
            }
        }
        check::<Gf8B>(&mut 17);
        check::<Gf16>(&mut 19);
    }

    /// A plan over the bit basis with a coset
    /// shift at the root built.
    fn bit_shifted_plan<F: ButterflyKernels>(
        size: usize,
        shift: F::Elem,
        state: &mut u32,
    ) -> TransformPlan<F> {
        let log_size = size.trailing_zeros() as usize;
        let basis = factors::bit_basis::<F>(log_size);
        let _ = lcg(state);
        TransformPlan::<F>::with_shift(size, &basis, shift).unwrap()
    }

    fn shifted_random<F: Field>(
        state: &mut u32,
        shift_seed: u32,
        size: usize,
    ) -> (F::Elem, Vec<F::Elem>) {
        *state = state.wrapping_mul(shift_seed).wrapping_add(1);
        (
            random_elements::<F>(state, 1)[0],
            random_elements::<F>(state, size),
        )
    }

    /// Contract: shifted forward row `i` is the polynomial's value at
    /// `α ⊕ element(i)`. Anchored on the monomial form so the check is
    /// independent of the novel-basis machinery.
    #[test]
    fn shifted_forward_evaluates_at_coset_points() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 0..=log_cap(7) {
                let size = 1 << log_size;
                let (shift, novel) = shifted_random::<F>(state, 3, size);
                let plan = bit_shifted_plan::<F>(size, shift, state);

                let mut monomial = novel.clone();
                crate::basis::novel_to_monomial(&mut monomial, &plan).unwrap();

                let mut values = novel;
                plan.forward(&mut values).unwrap();
                for (index, &value) in values.iter().enumerate() {
                    assert_eq!(
                        value,
                        horner_eval::<F>(&monomial, plan.point_element(index)),
                        "{} size {size} point {index}",
                        F::NAME
                    );
                }
            }
        }
        check::<Gf8B>(&mut 0x1122_3344);
        check::<Gf16>(&mut 0x5566_7788);
    }

    #[test]
    fn shifted_round_trips() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 0..=log_cap(8) {
                let size = 1 << log_size;
                let (shift, original) = shifted_random::<F>(state, 5, size);
                let plan = bit_shifted_plan::<F>(size, shift, state);
                let mut values = original.clone();
                plan.forward(&mut values).unwrap();
                plan.inverse(&mut values).unwrap();
                assert_eq!(values, original, "{} size {size}", F::NAME);

                let row_len = 3 * F::BYTES;
                let bytes: Vec<u8> = (0..size * row_len)
                    .map(|_| lcg(state).to_le_bytes()[1])
                    .collect();
                let mut rows = bytes.clone();
                plan.forward_bytes(&mut rows, row_len).unwrap();
                plan.inverse_bytes(&mut rows, row_len).unwrap();
                assert_eq!(rows, bytes, "{} byte rows size {size}", F::NAME);
            }
        }
        check::<Gf8B>(&mut 0xaaaa_5555);
        check::<Gf16>(&mut 0x5555_aaaa);
    }

    /// Coset decomposition: an unshifted transform of dimension `k` is the
    /// concatenation of two dimension-`(k-1)` transforms over the cosets
    /// `0 + V_{k-1}` and `β_{k-1} + V_{k-1}`, applied to the novel-basis
    /// halves. This is the identity the encoder's high-coset shortcut rests
    /// on, generalized off node 3.
    #[test]
    fn coset_halves_reconstruct_the_full_transform() {
        fn check<F: ButterflyKernels>(state: &mut u32) {
            for log_size in 1..=log_cap(7) {
                let size = 1 << log_size;
                let half = size / 2;
                let full = TransformPlan::<F>::new(size).unwrap();
                let coefficients = random_elements::<F>(state, size);
                let mut expected = coefficients.clone();
                full.forward(&mut expected).unwrap();

                let direction = &full.basis()[..log_size - 1];
                let split = full.basis()[log_size - 1];
                let low = TransformPlan::<F>::with_shift(half, direction, F::Elem::ZERO).unwrap();
                let high = TransformPlan::<F>::with_shift(half, direction, split).unwrap();

                // f = f_lo + W̄_{k-1}·f_hi; over the low coset W̄ evaluates
                // to the node factor, so each half is the same novel
                // coefficient vector seen through its own coset table.
                let mut low_values = coefficients[..half].to_vec();
                let mut high_values = coefficients[..half].to_vec();
                let tail = &coefficients[half..];
                low.forward(&mut low_values).unwrap();
                high.forward(&mut high_values).unwrap();

                let mut tail_low = tail.to_vec();
                let mut tail_high = tail.to_vec();
                low.forward(&mut tail_low).unwrap();
                high.forward(&mut tail_high).unwrap();

                for index in 0..half {
                    // W̄_{k-1} at the low coset point, then the high one.
                    let w_low =
                        normalized_top::<F>(full.basis(), log_size, low.point_element(index));
                    let w_high =
                        normalized_top::<F>(full.basis(), log_size, high.point_element(index));
                    assert_eq!(
                        low_values[index].add(w_low.mul(tail_low[index])),
                        expected[index],
                        "{} low coset {index} size {size}",
                        F::NAME
                    );
                    assert_eq!(
                        high_values[index].add(w_high.mul(tail_high[index])),
                        expected[half + index],
                        "{} high coset {index} size {size}",
                        F::NAME
                    );
                }
            }
        }
        check::<Gf8B>(&mut 0x0f0f_f0f0);
        check::<Gf16>(&mut 0xf0f0_0f0f);
    }

    /// `W̄_{log_size-1}` evaluated at `point`, straight from the polynomial
    /// chain rather than through any table.
    fn normalized_top<F: ButterflyKernels>(
        basis: &[F::Elem],
        log_size: usize,
        point: F::Elem,
    ) -> F::Elem {
        factors::subspace_polynomials(&basis[..log_size]).expect("plan basis is independent")
            [log_size - 1]
            .evaluate(point)
    }

    #[cfg(feature = "std")]
    #[test]
    fn shifted_plans_over_a_cantor_basis() {
        let cantor = crate::basis::cantor_basis::<Gf16>().unwrap();
        let mut state = 0xbeef_cafeu32;
        let _ = lcg(&mut state);
        let (shift, original) = shifted_random::<Gf16>(&mut state, 7, 64);
        let plan = TransformPlan::<Gf16>::with_shift(64, &cantor.elements()[..6], shift).unwrap();
        assert_eq!(plan.point_element(0), shift);
        assert_eq!(plan.basis(), &cantor.elements()[..6]);

        let mut values = original.clone();
        plan.forward(&mut values).unwrap();
        plan.inverse(&mut values).unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn zero_shift_matches_the_unshifted_plan() {
        let mut state = 0x1357_9bdfu32;
        let (.., original) = shifted_random::<Gf16>(&mut state, 11, 128);
        let plan = bit_shifted_plan::<Gf16>(128, <Gf16 as Field>::Elem::ZERO, &mut state);
        let unshifted = TransformPlan::<Gf16>::new(128).unwrap();
        let mut shifted_values = original.clone();
        let mut plain_values = original;
        plan.forward(&mut shifted_values).unwrap();
        unshifted.forward(&mut plain_values).unwrap();
        assert_eq!(shifted_values, plain_values);
        for index in 0..128 {
            assert_eq!(plan.point_element(index), unshifted.point_element(index));
        }
    }

    /// The derivative is a coefficient-basis operation: the same plan with
    /// and without a coset shift produces the same derivative bytes.
    #[test]
    fn derivative_is_shift_independent() {
        let mut state = 0x2468_ace0u32;
        let (shift, coefficients) = shifted_random::<Gf16>(&mut state, 13, 16);
        let row_len = <Gf16 as Field>::BYTES;
        let mut coefficient_rows = vec![0u8; 16 * row_len];
        for (row, &coefficient) in coefficients.iter().enumerate() {
            Gf16::encode(
                &mut coefficient_rows[row * row_len..][..row_len],
                coefficient,
            );
        }
        let shifted = bit_shifted_plan::<Gf16>(16, shift, &mut state);
        let plain = TransformPlan::<Gf16>::new(16).unwrap();
        let mut shifted_rows = vec![0u8; coefficient_rows.len()];
        let mut plain_rows = vec![0u8; coefficient_rows.len()];
        shifted
            .derivative_into_bytes(&mut shifted_rows, row_len, &coefficient_rows)
            .unwrap();
        plain
            .derivative_into_bytes(&mut plain_rows, row_len, &coefficient_rows)
            .unwrap();
        assert_eq!(shifted_rows, plain_rows);
    }

    #[test]
    fn with_shift_shares_size_validation() {
        let shift = <Gf8B as Field>::Elem::ZERO;
        let expected = PlanError::DomainTooLarge {
            log_size: 9,
            cap: 8,
        };
        assert_eq!(
            TransformPlan::<Gf8B>::with_shift(512, &[], shift).unwrap_err(),
            expected
        );
    }
}
