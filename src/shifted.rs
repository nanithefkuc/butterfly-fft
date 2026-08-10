//! Shifted subspace transforms over affine cosets `α + V`.
//!
//! A shifted plan is the same twiddle-table shape built with a nonzero coset
//! shift at the root, so every execution model in
//! [`crate::core::transform`] — truncated and selected variants included —
//! composes with arbitrary shifts.
//!
//! # Why this is free
//!
//! The twiddle at recursion node `ν` is `W̄_d(s)` for that node's coset
//! shift `s`, and the table builder already threads `s` down from the root.
//! Setting the root shift to `α` therefore evaluates over `α + V` with
//! *identical* walkers and identical cost — no extra pass, no larger table.
//! Reed–Solomon repair evaluation is the special case `α = β_{n-1}`, which
//! [`crate::core::transform::TransformPlan::forward_bytes_high_coset_range`]
//! reaches directly through the unshifted table's node 3.
//!
//! # Caching
//!
//! Shifted tables are per `(size, basis, shift)` and are *not* cached: the
//! process-wide cache covers the unshifted bit-basis plans that every
//! consumer shares. A consumer working one coset across many payloads should
//! hold its `ShiftedPlan`.
//!
//! ```
//! use butterfly_fft::basis::{BitBasis, OrderedBasis};
//! use butterfly_fft::shifted::ShiftedPlan;
//! use fgf::{Gf16, field::Elem, gf16};
//!
//! // Evaluate over the coset α + span{β_0, β_1}.
//! let shift = gf16::Elem(0x2ba7);
//! let plan = ShiftedPlan::<Gf16>::new(4, shift).unwrap();
//! assert_eq!(plan.point_element(0), shift);
//! assert_eq!(
//!     plan.point_element(3),
//!     shift
//!         .add(OrderedBasis::<Gf16>::element(&BitBasis, 0))
//!         .add(OrderedBasis::<Gf16>::element(&BitBasis, 1)),
//! );
//! ```

use ::alloc::vec::Vec;
use ::core::ops::Range;

use fgf::field::Elem;

use crate::basis::{BitBasis, OrderedBasis};
use crate::core::kernel::ButterflyKernels;
use crate::core::transform::{TransformPlan, validate_size};
use crate::error::{PlanError, TransformLengthError};

/// An additive-FFT plan over the affine coset `shift + span(basis)`.
///
/// Transform point `i` is `shift ⊕ point_i`, where `point_i` is the XOR of
/// the basis elements at the set bits of `i`. With a zero shift this is
/// exactly a [`TransformPlan`].
#[derive(Clone, Debug)]
pub struct ShiftedPlan<F: ButterflyKernels> {
    base: TransformPlan<F>,
    shift: F::Elem,
}

impl<F: ButterflyKernels> ShiftedPlan<F> {
    /// A shifted plan over the bit basis.
    ///
    /// # Errors
    /// As [`TransformPlan::new`].
    pub fn new(size: usize, shift: F::Elem) -> Result<Self, PlanError> {
        let log_size = validate_size::<F>(size)?;
        let basis = <BitBasis as OrderedBasis<F>>::prefix(&BitBasis, log_size);
        Self::from_elements(size, &basis, shift)
    }

    /// A shifted plan over an arbitrary ordered basis.
    ///
    /// # Errors
    /// As [`TransformPlan::with_basis`].
    pub fn with_basis(
        size: usize,
        basis: &impl OrderedBasis<F>,
        shift: F::Elem,
    ) -> Result<Self, PlanError> {
        let log_size = validate_size::<F>(size)?;
        if basis.bits() < log_size {
            return Err(PlanError::BasisTooShort {
                needed: log_size,
                got: basis.bits(),
            });
        }
        Self::from_elements(size, &basis.prefix(log_size), shift)
    }

    /// A shifted plan over an explicit ordered-basis prefix.
    ///
    /// # Errors
    /// As [`TransformPlan::with_basis`].
    pub fn from_elements(
        size: usize,
        basis: &[F::Elem],
        shift: F::Elem,
    ) -> Result<Self, PlanError> {
        Ok(Self {
            base: TransformPlan::<F>::with_basis_shift(size, basis, shift)?,
            shift,
        })
    }

    /// The coset shift `α`.
    #[must_use]
    pub const fn shift(&self) -> F::Elem {
        self.shift
    }

    /// The underlying plan, for execution models not re-exposed here.
    ///
    /// Its own `point_element` reports the *unshifted* subspace point; use
    /// [`ShiftedPlan::point_element`] for this plan's evaluation points.
    #[must_use]
    pub const fn plan(&self) -> &TransformPlan<F> {
        &self.base
    }

    /// Number of transform points.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.base.size()
    }

    /// Base-two logarithm of [`ShiftedPlan::size`].
    #[must_use]
    pub const fn log_size(&self) -> usize {
        self.base.log_size()
    }

    /// The ordered-basis prefix spanning the coset's direction space.
    #[must_use]
    pub fn basis(&self) -> &[F::Elem] {
        self.base.basis()
    }

    /// The field element of transform point `index`: `shift ⊕ point_index`.
    #[must_use]
    pub fn point_element(&self, index: usize) -> F::Elem {
        self.shift.add(self.base.point_element(index))
    }

    /// Every evaluation point of the coset, in transform order.
    #[must_use]
    pub fn points(&self) -> Vec<F::Elem> {
        (0..self.size())
            .map(|index| self.point_element(index))
            .collect()
    }

    /// Evaluate novel-basis coefficients over the coset, in place.
    ///
    /// # Errors
    /// As [`TransformPlan::forward`].
    pub fn forward(&self, values: &mut [F::Elem]) -> Result<(), TransformLengthError> {
        self.base.forward(values)
    }

    /// Interpolate coset evaluations back to novel-basis coefficients.
    ///
    /// # Errors
    /// As [`TransformPlan::inverse`].
    pub fn inverse(&self, values: &mut [F::Elem]) -> Result<(), TransformLengthError> {
        self.base.inverse(values)
    }

    /// Formal derivative of novel-basis coefficients, out of place.
    ///
    /// The derivative is a coefficient-basis operation and uses the same
    /// basis-relative derivative factors regardless of the coset shift.
    ///
    /// # Errors
    /// As [`TransformPlan::derivative`].
    pub fn derivative(
        &self,
        coefficients: &[F::Elem],
        derivative: &mut [F::Elem],
    ) -> Result<(), TransformLengthError> {
        self.base.derivative(coefficients, derivative)
    }

    /// Byte-row coset forward transform.
    ///
    /// # Errors
    /// As [`TransformPlan::forward_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::forward_bytes`].
    pub fn forward_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.base.forward_bytes(rows, row_len)
    }

    /// Byte-row coset inverse transform.
    ///
    /// # Errors
    /// As [`TransformPlan::inverse_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::inverse_bytes`].
    pub fn inverse_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
    ) -> Result<(), TransformLengthError> {
        self.base.inverse_bytes(rows, row_len)
    }

    /// Formal derivative over interleaved byte rows, out of place.
    ///
    /// # Errors
    /// As [`TransformPlan::derivative_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::derivative_bytes`].
    pub fn derivative_bytes(
        &self,
        coefficients: &[u8],
        row_len: usize,
        derivative: &mut [u8],
    ) -> Result<(), TransformLengthError> {
        self.base
            .derivative_bytes(coefficients, row_len, derivative)
    }

    /// Byte-row coset forward transform restricted to `selected` rows.
    ///
    /// # Errors
    /// As [`TransformPlan::forward_bytes_selected`].
    ///
    /// # Panics
    /// As [`TransformPlan::forward_bytes_selected`].
    pub fn forward_bytes_selected(
        &self,
        rows: &mut [u8],
        row_len: usize,
        selected: &[usize],
    ) -> Result<(), TransformLengthError> {
        self.base.forward_bytes_selected(rows, row_len, selected)
    }

    /// Byte-row coset forward transform restricted to a contiguous range.
    ///
    /// # Errors
    /// As [`TransformPlan::forward_bytes_range`].
    ///
    /// # Panics
    /// As [`TransformPlan::forward_bytes_range`].
    pub fn forward_bytes_range(
        &self,
        rows: &mut [u8],
        row_len: usize,
        range: Range<usize>,
    ) -> Result<(), TransformLengthError> {
        self.base.forward_bytes_range(rows, row_len, range)
    }

    /// Truncated byte-row coset forward transform: zero-padded coefficients
    /// beyond `active`, output restricted to `range`.
    ///
    /// # Errors
    /// As [`TransformPlan::forward_bytes_trunc_range`].
    ///
    /// # Panics
    /// As [`TransformPlan::forward_bytes_trunc_range`].
    pub fn forward_bytes_trunc_range(
        &self,
        rows: &mut [u8],
        row_len: usize,
        active: usize,
        range: Range<usize>,
    ) -> Result<(), TransformLengthError> {
        self.base
            .forward_bytes_trunc_range(rows, row_len, active, range)
    }

    /// Evaluate the low coefficient half over the high half-coset of this
    /// shifted domain, restricted to `range`.
    ///
    /// # Errors
    /// As [`TransformPlan::forward_bytes_high_coset_range`].
    ///
    /// # Panics
    /// As [`TransformPlan::forward_bytes_high_coset_range`].
    pub fn forward_bytes_high_coset_range(
        &self,
        rows: &mut [u8],
        row_len: usize,
        range: Range<usize>,
    ) -> Result<(), TransformLengthError> {
        self.base
            .forward_bytes_high_coset_range(rows, row_len, range)
    }

    /// Scratch rows required by [`ShiftedPlan::inverse_truncated_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::inverse_truncated_scratch_rows`].
    #[must_use]
    pub fn inverse_truncated_scratch_rows(&self, active: usize) -> usize {
        self.base.inverse_truncated_scratch_rows(active)
    }

    /// Truncated byte-row coset inverse: recover the lowest `active`
    /// novel-basis coefficients from coset evaluations.
    ///
    /// # Errors
    /// As [`TransformPlan::inverse_truncated_bytes`].
    ///
    /// # Panics
    /// As [`TransformPlan::inverse_truncated_bytes`].
    pub fn inverse_truncated_bytes(
        &self,
        rows: &mut [u8],
        row_len: usize,
        active: usize,
        scratch: &mut [u8],
    ) -> Result<(), TransformLengthError> {
        self.base
            .inverse_truncated_bytes(rows, row_len, active, scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{CantorBasis, novel_to_monomial};
    use ::alloc::vec;
    use fgf::field::Field;
    use fgf::{Gf8, Gf16};

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn elem<F: Field>(&mut self) -> F::Elem {
            F::read(&self.next_u64().to_le_bytes()[..F::BYTES])
        }

        fn elems<F: Field>(&mut self, count: usize) -> Vec<F::Elem> {
            (0..count).map(|_| self.elem::<F>()).collect()
        }
    }

    fn horner<E: Elem>(coefficients: &[E], point: E) -> E {
        coefficients
            .iter()
            .rev()
            .fold(E::ZERO, |accumulator, &coefficient| {
                accumulator.mul(point).add(coefficient)
            })
    }

    /// Contract 3: shifted forward row `i` is the polynomial's value at
    /// `α ⊕ element(i)`. Anchored on the monomial form so the check is
    /// independent of the novel-basis machinery.
    #[test]
    fn shifted_forward_evaluates_at_coset_points() {
        fn check<F: ButterflyKernels>(rng: &mut Rng) {
            for log_size in 0..=7usize {
                let size = 1 << log_size;
                let shift = rng.elem::<F>();
                let plan = ShiftedPlan::<F>::new(size, shift).unwrap();

                let novel = rng.elems::<F>(size);
                let mut monomial = novel.clone();
                novel_to_monomial(&mut monomial, plan.plan()).unwrap();

                let mut values = novel;
                plan.forward(&mut values).unwrap();
                for (index, &value) in values.iter().enumerate() {
                    assert_eq!(
                        value,
                        horner(&monomial, plan.point_element(index)),
                        "{} size {size} point {index}",
                        F::NAME
                    );
                }
            }
        }
        check::<Gf8>(&mut Rng(0x1122_3344_5566_7788));
        check::<Gf16>(&mut Rng(0x8877_6655_4433_2211));
    }

    #[test]
    fn shifted_round_trips() {
        fn check<F: ButterflyKernels>(rng: &mut Rng) {
            for log_size in 0..=8usize {
                let size = 1 << log_size;
                let plan = ShiftedPlan::<F>::new(size, rng.elem::<F>()).unwrap();
                let original = rng.elems::<F>(size);
                let mut values = original.clone();
                plan.forward(&mut values).unwrap();
                plan.inverse(&mut values).unwrap();
                assert_eq!(values, original, "{} size {size}", F::NAME);

                let row_len = 3 * F::BYTES;
                let bytes: Vec<u8> = (0..size * row_len)
                    .map(|_| rng.next_u64().to_le_bytes()[0])
                    .collect();
                let mut rows = bytes.clone();
                plan.forward_bytes(&mut rows, row_len).unwrap();
                plan.inverse_bytes(&mut rows, row_len).unwrap();
                assert_eq!(rows, bytes, "{} byte rows size {size}", F::NAME);
            }
        }
        check::<Gf8>(&mut Rng(0xaaaa_5555_aaaa_5555));
        check::<Gf16>(&mut Rng(0x5555_aaaa_5555_aaaa));
    }

    /// Coset decomposition: an unshifted transform of dimension `k` is the
    /// concatenation of two dimension-`(k-1)` transforms over the cosets
    /// `0 + V_{k-1}` and `β_{k-1} + V_{k-1}`, applied to the novel-basis
    /// halves. This is the identity the encoder's high-coset shortcut rests
    /// on, generalized off node 3.
    #[test]
    fn coset_halves_reconstruct_the_full_transform() {
        fn check<F: ButterflyKernels>(rng: &mut Rng) {
            for log_size in 1..=7usize {
                let size = 1 << log_size;
                let half = size / 2;
                let full = TransformPlan::<F>::new(size).unwrap();
                let coefficients = rng.elems::<F>(size);
                let mut expected = coefficients.clone();
                full.forward(&mut expected).unwrap();

                let direction = &full.basis()[..log_size - 1];
                let split = full.basis()[log_size - 1];
                let low = ShiftedPlan::<F>::from_elements(half, direction, F::Elem::ZERO).unwrap();
                let high = ShiftedPlan::<F>::from_elements(half, direction, split).unwrap();

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
        check::<Gf8>(&mut Rng(0x0f0f_0f0f_f0f0_f0f0));
        check::<Gf16>(&mut Rng(0xf0f0_f0f0_0f0f_0f0f));
    }

    /// `W̄_{log_size-1}` evaluated at `point`, straight from the polynomial
    /// chain rather than through any table.
    fn normalized_top<F: ButterflyKernels>(
        basis: &[F::Elem],
        log_size: usize,
        point: F::Elem,
    ) -> F::Elem {
        crate::core::factors::subspace_polynomials(&basis[..log_size])
            .expect("plan basis is independent")[log_size - 1]
            .evaluate(point)
    }

    #[test]
    fn shifted_plans_over_a_cantor_basis() {
        let cantor = CantorBasis::<Gf16>::build().unwrap();
        let mut rng = Rng(0xbeef_cafe_dead_f00d);
        let shift = rng.elem::<Gf16>();
        let plan = ShiftedPlan::<Gf16>::with_basis(64, &cantor, shift).unwrap();
        assert_eq!(plan.point_element(0), shift);
        assert_eq!(plan.basis(), &cantor.elements()[..6]);

        let original = rng.elems::<Gf16>(64);
        let mut values = original.clone();
        plan.forward(&mut values).unwrap();
        plan.inverse(&mut values).unwrap();
        assert_eq!(values, original);
    }

    #[test]
    fn zero_shift_matches_the_unshifted_plan() {
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        let plan = ShiftedPlan::<Gf16>::new(128, <Gf16 as Field>::Elem::ZERO).unwrap();
        let unshifted = TransformPlan::<Gf16>::new(128).unwrap();
        let original = rng.elems::<Gf16>(128);
        let mut shifted_values = original.clone();
        let mut plain_values = original;
        plan.forward(&mut shifted_values).unwrap();
        unshifted.forward(&mut plain_values).unwrap();
        assert_eq!(shifted_values, plain_values);
        for index in 0..128 {
            assert_eq!(plan.point_element(index), unshifted.point_element(index));
        }
    }
    #[test]
    fn shifted_facade_delegates_derivative_and_high_coset() {
        let mut rng = Rng(0x2468_ace0_1357_9bdf);
        let plan = ShiftedPlan::<Gf16>::new(16, rng.elem::<Gf16>()).unwrap();
        let coefficients = rng.elems::<Gf16>(16);

        let mut derivative = vec![<Gf16 as Field>::Elem::ZERO; 16];
        let mut expected_derivative = derivative.clone();
        plan.derivative(&coefficients, &mut derivative).unwrap();
        plan.plan()
            .derivative(&coefficients, &mut expected_derivative)
            .unwrap();
        assert_eq!(derivative, expected_derivative);

        let row_len = <Gf16 as Field>::BYTES;
        let mut coefficient_rows = vec![0u8; 16 * row_len];
        for (row, &coefficient) in coefficients.iter().enumerate() {
            Gf16::write(
                &mut coefficient_rows[row * row_len..][..row_len],
                coefficient,
            );
        }
        let mut derivative_rows = vec![0u8; coefficient_rows.len()];
        let mut expected_rows = derivative_rows.clone();
        plan.derivative_bytes(&coefficient_rows, row_len, &mut derivative_rows)
            .unwrap();
        plan.plan()
            .derivative_bytes(&coefficient_rows, row_len, &mut expected_rows)
            .unwrap();
        assert_eq!(derivative_rows, expected_rows);

        let half = plan.size() / 2;
        let mut padded = coefficients[..half].to_vec();
        padded.resize(plan.size(), <Gf16 as Field>::Elem::ZERO);
        plan.forward(&mut padded).unwrap();
        let mut high_rows = vec![0u8; half * row_len];
        for (row, &coefficient) in coefficients[..half].iter().enumerate() {
            Gf16::write(&mut high_rows[row * row_len..][..row_len], coefficient);
        }
        plan.forward_bytes_high_coset_range(&mut high_rows, row_len, 0..half)
            .unwrap();
        for (row, &expected) in padded[half..].iter().enumerate() {
            assert_eq!(Gf16::read(&high_rows[row * row_len..][..row_len]), expected);
        }
    }

    #[test]
    fn shifted_constructors_share_size_validation() {
        let shift = <Gf8 as Field>::Elem::ZERO;
        let expected = PlanError::DomainTooLarge {
            log_size: 9,
            cap: 8,
        };
        assert_eq!(ShiftedPlan::<Gf8>::new(512, shift).unwrap_err(), expected);
        assert_eq!(
            ShiftedPlan::<Gf8>::with_basis(512, &BitBasis, shift).unwrap_err(),
            expected
        );
    }
}
