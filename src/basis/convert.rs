//! Monomial ↔ novel coefficient-basis conversion.
//!
//! The novel basis is `X_i(x) = ∏_j W̄_j(x)^{bit_j(i)}` with
//! `deg X_i = i`, so both directions are triangular. The naive
//! term-rewriting is `O(n²)`; the recursion below is `O(n log² n)` and no
//! harder to follow, because it is just the transform's own split read as
//! polynomial algebra:
//!
//! ```text
//! f = Σ_{i < n} a_i X_i = f_lo(x) + W̄_{k-1}(x) · f_hi(x)
//! ```
//!
//! where `f_lo` and `f_hi` are the conversions of the two coefficient
//! halves over dimension `k-1`. Novel → monomial multiplies by the sparse
//! linearized `W̄_{k-1}` (`k` nonzero terms) on the way up; monomial → novel
//! divides by it on the way down. Both are `O(n·k)` per level.

use ::alloc::vec;

use fgf::field::Elem;

use crate::core::kernel::{ButterflyKernels, xor_scaled_bytes};
use crate::core::transform::TransformPlan;
use crate::error::TransformLengthError;

/// Number of field elements required by scratch-taking coefficient
/// conversions for a domain of `size` coefficients.
#[inline]
#[must_use]
pub const fn conversion_scratch_elements(size: usize) -> usize {
    size / 2
}

/// Rewrite novel-basis coefficients as monomial coefficients in place.
///
/// On return `coefficients[d]` is the coefficient of `x^d` in the same
/// polynomial. Allocates one scratch buffer of `size / 2` elements.
///
/// # Errors
/// Returns [`TransformLengthError`] unless `coefficients.len() ==
/// plan.size()`.
pub fn novel_to_monomial<F: ButterflyKernels>(
    coefficients: &mut [F::Elem],
    plan: &TransformPlan<F>,
) -> Result<(), TransformLengthError> {
    check_len(coefficients.len(), plan.size())?;
    if plan.log_size() == 0 {
        return Ok(());
    }
    let mut scratch = vec![F::Elem::ZERO; conversion_scratch_elements(plan.size())];
    novel_to_monomial_node(coefficients, plan, plan.log_size(), &mut scratch);
    Ok(())
}
/// Rewrite novel-basis coefficients as monomial coefficients in place using
/// caller-provided scratch.
///
/// This allocation-free variant requires at least
/// [`conversion_scratch_elements`]`(plan.size())` scratch elements. All
/// coefficient entries are final outputs on return.
///
/// # Errors
/// Returns [`TransformLengthError`] unless `coefficients.len() ==
/// plan.size()` and `scratch` is large enough.
pub fn novel_to_monomial_with_scratch<F: ButterflyKernels>(
    coefficients: &mut [F::Elem],
    plan: &TransformPlan<F>,
    scratch: &mut [F::Elem],
) -> Result<(), TransformLengthError> {
    check_len(coefficients.len(), plan.size())?;
    let required = conversion_scratch_elements(plan.size());
    check_min_len(scratch.len(), required)?;
    if plan.log_size() != 0 {
        novel_to_monomial_node(
            coefficients,
            plan,
            plan.log_size(),
            &mut scratch[..required],
        );
    }
    Ok(())
}

/// Rewrite monomial coefficients as novel-basis coefficients in place, the
/// inverse of [`novel_to_monomial`].
///
/// # Errors
/// Returns [`TransformLengthError`] unless `coefficients.len() ==
/// plan.size()`.
pub fn monomial_to_novel<F: ButterflyKernels>(
    coefficients: &mut [F::Elem],
    plan: &TransformPlan<F>,
) -> Result<(), TransformLengthError> {
    check_len(coefficients.len(), plan.size())?;
    if plan.log_size() == 0 {
        return Ok(());
    }
    let mut scratch = vec![F::Elem::ZERO; conversion_scratch_elements(plan.size())];
    monomial_to_novel_node(coefficients, plan, plan.log_size(), &mut scratch);
    Ok(())
}
/// Rewrite monomial coefficients as novel-basis coefficients in place using
/// caller-provided scratch.
///
/// This allocation-free variant requires at least
/// [`conversion_scratch_elements`]`(plan.size())` scratch elements. All
/// coefficient entries are final outputs on return.
///
/// # Errors
/// Returns [`TransformLengthError`] unless `coefficients.len() ==
/// plan.size()` and `scratch` is large enough.
pub fn monomial_to_novel_with_scratch<F: ButterflyKernels>(
    coefficients: &mut [F::Elem],
    plan: &TransformPlan<F>,
    scratch: &mut [F::Elem],
) -> Result<(), TransformLengthError> {
    check_len(coefficients.len(), plan.size())?;
    let required = conversion_scratch_elements(plan.size());
    check_min_len(scratch.len(), required)?;
    if plan.log_size() != 0 {
        monomial_to_novel_node(
            coefficients,
            plan,
            plan.log_size(),
            &mut scratch[..required],
        );
    }
    Ok(())
}

/// Rewrite SIMD-batched novel-basis coefficient rows as monomial coefficient
/// rows in place.
///
/// Row `d` contains one or more packed field elements, each an independent
/// polynomial lane. On return every row is a final output: row `d` contains
/// the monomial coefficient of `x^d` for each lane. `scratch` must hold at
/// least `plan.size() / 2` rows of `row_len` bytes.
///
/// # Errors
/// Returns [`TransformLengthError`] (lengths in bytes) unless `coefficients`
/// holds exactly `plan.size()` rows and `scratch` is large enough.
///
/// # Panics
/// Panics if `row_len` is zero, holds a partial trailing element, or a
/// complete byte length is not representable by [`usize`].
pub fn novel_to_monomial_bytes<F: ButterflyKernels>(
    coefficients: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    scratch: &mut [u8],
) -> Result<(), TransformLengthError> {
    let required = check_byte_geometry::<F>(coefficients.len(), row_len, plan.size())?;
    check_min_len(scratch.len(), required)?;
    if plan.log_size() != 0 {
        novel_to_monomial_bytes_node(
            coefficients,
            row_len,
            plan,
            plan.log_size(),
            &mut scratch[..required],
        );
    }
    Ok(())
}

/// Rewrite SIMD-batched monomial coefficient rows as novel-basis coefficient
/// rows in place.
///
/// Row `d` contains one or more packed field elements, each an independent
/// polynomial lane. On return every row is a final output: row `i` contains
/// the novel-basis coefficient of `X_i` for each lane. `scratch` must hold at
/// least `plan.size() / 2` rows of `row_len` bytes.
///
/// # Errors
/// Returns [`TransformLengthError`] (lengths in bytes) unless `coefficients`
/// holds exactly `plan.size()` rows and `scratch` is large enough.
///
/// # Panics
/// Panics if `row_len` is zero, holds a partial trailing element, or a
/// complete byte length is not representable by [`usize`].
pub fn monomial_to_novel_bytes<F: ButterflyKernels>(
    coefficients: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    scratch: &mut [u8],
) -> Result<(), TransformLengthError> {
    let required = check_byte_geometry::<F>(coefficients.len(), row_len, plan.size())?;
    check_min_len(scratch.len(), required)?;
    if plan.log_size() != 0 {
        monomial_to_novel_bytes_node(
            coefficients,
            row_len,
            plan,
            plan.log_size(),
            &mut scratch[..required],
        );
    }
    Ok(())
}

/// Interpolate domain evaluations into monomial coefficients in place.
///
/// Composes [`TransformPlan::inverse_bytes`] with
/// [`novel_to_monomial_bytes`]: each `row_len`-byte row `i` starts as the
/// evaluation at the plan's point `i` and ends as the monomial coefficient
/// of `x^i` of the unique degree-`< plan.size()` interpolating polynomial.
/// Multiple independent polynomials may be packed across each byte row.
///
/// `scratch` must hold at least `plan.size() / 2` rows of `row_len` bytes
/// (see [`conversion_scratch_elements`]).
///
/// # Errors
/// As [`TransformPlan::inverse_bytes`] and [`novel_to_monomial_bytes`].
///
/// # Panics
/// As those functions.
pub fn inverse_interpolate_bytes<F: ButterflyKernels>(
    rows: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    scratch: &mut [u8],
) -> Result<(), TransformLengthError> {
    plan.inverse_bytes(rows, row_len)?;
    novel_to_monomial_bytes(rows, row_len, plan, scratch)
}

fn check_len(got: usize, expected: usize) -> Result<(), TransformLengthError> {
    if got == expected {
        Ok(())
    } else {
        Err(TransformLengthError { expected, got })
    }
}

fn check_min_len(got: usize, expected: usize) -> Result<(), TransformLengthError> {
    if got >= expected {
        Ok(())
    } else {
        Err(TransformLengthError { expected, got })
    }
}

fn check_byte_geometry<F: ButterflyKernels>(
    got: usize,
    row_len: usize,
    size: usize,
) -> Result<usize, TransformLengthError> {
    assert_ne!(row_len, 0, "row length must be nonzero");
    assert_eq!(row_len % F::BYTES, 0, "partial trailing element");
    let expected = size
        .checked_mul(row_len)
        .expect("coefficient byte length overflow");
    let required = conversion_scratch_elements(size)
        .checked_mul(row_len)
        .expect("scratch byte length overflow");
    check_len(got, expected)?;
    Ok(required)
}

/// Bottom-up: convert both halves, then fold them together as
/// `f = f_lo + W̄_{dimension-1}·f_hi`.
fn novel_to_monomial_node<F: ButterflyKernels>(
    values: &mut [F::Elem],
    plan: &TransformPlan<F>,
    dimension: usize,
    scratch: &mut [F::Elem],
) {
    if dimension == 0 {
        return;
    }
    let half = values.len() / 2;
    {
        let (low, high) = values.split_at_mut(half);
        novel_to_monomial_node(low, plan, dimension - 1, scratch);
        novel_to_monomial_node(high, plan, dimension - 1, scratch);
    }

    // `f_hi` currently sits in the upper half as coefficients of degree
    // `0..half`; the product scatters it across the whole range, including
    // over itself, so it is lifted out first.
    let factor = plan.normalized_subspace_polynomial(dimension - 1);
    let product_source = &mut scratch[..half];
    product_source.copy_from_slice(&values[half..]);
    values[half..].fill(F::Elem::ZERO);
    for (degree, &coefficient) in product_source.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        for (exponent, &term) in factor.iter().enumerate() {
            if term.is_zero() {
                continue;
            }
            let target = degree + (1 << exponent);
            values[target] = values[target].add(coefficient.mul(term));
        }
    }
}

/// Top-down: divide by `W̄_{dimension-1}`; the remainder is the low
/// coefficient half and the quotient is the high half, then recurse.
fn monomial_to_novel_node<F: ButterflyKernels>(
    values: &mut [F::Elem],
    plan: &TransformPlan<F>,
    dimension: usize,
    scratch: &mut [F::Elem],
) {
    if dimension == 0 {
        return;
    }
    let length = values.len();
    let half = length / 2;
    let factor = plan.normalized_subspace_polynomial(dimension - 1);
    // `W̄` has degree `2^(dimension-1) == half`; its leading coefficient is
    // the only one that can normalize the division.
    let leading_inverse = factor[dimension - 1].inv();

    let quotient = &mut scratch[..half];
    for degree in (half..length).rev() {
        let coefficient = values[degree].mul(leading_inverse);
        quotient[degree - half] = coefficient;
        if coefficient.is_zero() {
            continue;
        }
        for (exponent, &term) in factor.iter().enumerate() {
            if term.is_zero() {
                continue;
            }
            let target = degree - half + (1 << exponent);
            values[target] = values[target].add(coefficient.mul(term));
        }
        debug_assert!(values[degree].is_zero(), "division left a leading term");
    }
    values[half..].copy_from_slice(quotient);

    // The quotient has been moved into `values`, so the whole scratch is
    // free again; children need only `half / 2` of it.
    let (low, high) = values.split_at_mut(half);
    monomial_to_novel_node(low, plan, dimension - 1, scratch);
    monomial_to_novel_node(high, plan, dimension - 1, scratch);
}

fn row_range(index: usize, row_len: usize) -> ::core::ops::Range<usize> {
    let start = index.checked_mul(row_len).expect("row offset overflow");
    let end = start.checked_add(row_len).expect("row end overflow");
    start..end
}

/// Byte-row form of the bottom-up coefficient conversion. Each row operation
/// runs across every packed lane through fgf's dispatched vector primitives.
fn novel_to_monomial_bytes_node<F: ButterflyKernels>(
    values: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    dimension: usize,
    scratch: &mut [u8],
) {
    if dimension == 0 {
        return;
    }
    let shift = u32::try_from(dimension - 1).expect("coefficient dimension overflow");
    let half_rows = 1usize
        .checked_shl(shift)
        .expect("coefficient row count overflow");
    let half_bytes = half_rows
        .checked_mul(row_len)
        .expect("coefficient half length overflow");
    {
        let (low, high) = values.split_at_mut(half_bytes);
        novel_to_monomial_bytes_node(low, row_len, plan, dimension - 1, scratch);
        novel_to_monomial_bytes_node(high, row_len, plan, dimension - 1, scratch);
    }

    let factor = plan.normalized_subspace_polynomial(dimension - 1);
    let product_source = &mut scratch[..half_bytes];
    product_source.copy_from_slice(&values[half_bytes..]);
    values[half_bytes..].fill(0);
    for degree in 0..half_rows {
        let source = &product_source[row_range(degree, row_len)];
        for (exponent, &term) in factor.iter().enumerate() {
            if term.is_zero() {
                continue;
            }
            let target = degree
                .checked_add(1usize << exponent)
                .expect("coefficient row index overflow");
            xor_scaled_bytes::<F>(&mut values[row_range(target, row_len)], term, source);
        }
    }
}

/// Byte-row form of top-down polynomial division. Quotient rows live in
/// scratch, letting each scaled row update batch every independent lane.
fn monomial_to_novel_bytes_node<F: ButterflyKernels>(
    values: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    dimension: usize,
    scratch: &mut [u8],
) {
    if dimension == 0 {
        return;
    }
    let shift = u32::try_from(dimension).expect("coefficient dimension overflow");
    let length_rows = 1usize
        .checked_shl(shift)
        .expect("coefficient row count overflow");
    let half_rows = length_rows / 2;
    let half_bytes = half_rows
        .checked_mul(row_len)
        .expect("coefficient half length overflow");
    let factor = plan.normalized_subspace_polynomial(dimension - 1);
    let leading_inverse = factor[dimension - 1].inv();

    for degree in (half_rows..length_rows).rev() {
        let quotient_index = degree - half_rows;
        let quotient_range = row_range(quotient_index, row_len);
        {
            let quotient = &mut scratch[quotient_range.clone()];
            quotient.fill(0);
            xor_scaled_bytes::<F>(
                quotient,
                leading_inverse,
                &values[row_range(degree, row_len)],
            );
        }
        for (exponent, &term) in factor.iter().enumerate() {
            if term.is_zero() {
                continue;
            }
            let target = quotient_index
                .checked_add(1usize << exponent)
                .expect("coefficient row index overflow");
            xor_scaled_bytes::<F>(
                &mut values[row_range(target, row_len)],
                term,
                &scratch[quotient_range.clone()],
            );
        }
    }
    values[half_bytes..].copy_from_slice(&scratch[..half_bytes]);

    let (low, high) = values.split_at_mut(half_bytes);
    monomial_to_novel_bytes_node(low, row_len, plan, dimension - 1, scratch);
    monomial_to_novel_bytes_node(high, row_len, plan, dimension - 1, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BitBasis, CantorBasis, OrderedBasis};
    use ::alloc::vec::Vec;
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

    fn pack<F: Field>(lanes: &[Vec<F::Elem>]) -> Vec<u8> {
        let size = lanes.first().map_or(0, Vec::len);
        let row_len = lanes.len() * F::BYTES;
        let mut rows = vec![0u8; size * row_len];
        for (lane, coefficients) in lanes.iter().enumerate() {
            assert_eq!(coefficients.len(), size);
            for (degree, &coefficient) in coefficients.iter().enumerate() {
                let start = degree * row_len + lane * F::BYTES;
                F::write(&mut rows[start..start + F::BYTES], coefficient);
            }
        }
        rows
    }

    fn assert_rows<F: Field>(rows: &[u8], expected: &[Vec<F::Elem>]) {
        let row_len = expected.len() * F::BYTES;
        for (lane, coefficients) in expected.iter().enumerate() {
            for (degree, &coefficient) in coefficients.iter().enumerate() {
                let start = degree * row_len + lane * F::BYTES;
                assert_eq!(
                    F::read(&rows[start..start + F::BYTES]),
                    coefficient,
                    "{} degree {degree} lane {lane}",
                    F::NAME
                );
            }
        }
    }

    /// Ground truth: evaluate a monomial-basis polynomial by Horner.
    fn horner<E: Elem>(coefficients: &[E], point: E) -> E {
        coefficients
            .iter()
            .rev()
            .fold(E::ZERO, |accumulator, &coefficient| {
                accumulator.mul(point).add(coefficient)
            })
    }

    fn round_trip<F: ButterflyKernels>(plan: &TransformPlan<F>, rng: &mut Rng) {
        let original = rng.elems::<F>(plan.size());
        let mut values = original.clone();
        novel_to_monomial(&mut values, plan).unwrap();
        monomial_to_novel(&mut values, plan).unwrap();
        assert_eq!(values, original, "{} size {}", F::NAME, plan.size());

        let mut values = original.clone();
        monomial_to_novel(&mut values, plan).unwrap();
        novel_to_monomial(&mut values, plan).unwrap();
        assert_eq!(values, original, "{} size {}", F::NAME, plan.size());
    }

    fn scratch_and_byte_variants<F: ButterflyKernels>(
        plan: &TransformPlan<F>,
        lanes: usize,
        rng: &mut Rng,
    ) {
        let required = conversion_scratch_elements(plan.size());

        let monomial = rng.elems::<F>(plan.size());
        let mut expected_novel = monomial.clone();
        monomial_to_novel(&mut expected_novel, plan).unwrap();
        let mut scratch_novel = monomial.clone();
        let mut element_scratch = vec![F::Elem::ZERO; required + 1];
        monomial_to_novel_with_scratch(&mut scratch_novel, plan, &mut element_scratch).unwrap();
        assert_eq!(scratch_novel, expected_novel);

        let novel = rng.elems::<F>(plan.size());
        let mut expected_monomial = novel.clone();
        novel_to_monomial(&mut expected_monomial, plan).unwrap();
        let mut scratch_monomial = novel.clone();
        novel_to_monomial_with_scratch(&mut scratch_monomial, plan, &mut element_scratch).unwrap();
        assert_eq!(scratch_monomial, expected_monomial);

        let monomial_lanes: Vec<Vec<F::Elem>> =
            (0..lanes).map(|_| rng.elems::<F>(plan.size())).collect();
        let expected_novel_lanes: Vec<Vec<F::Elem>> = monomial_lanes
            .iter()
            .map(|lane| {
                let mut converted = lane.clone();
                monomial_to_novel(&mut converted, plan).unwrap();
                converted
            })
            .collect();
        let row_len = lanes * F::BYTES;
        let mut rows = pack::<F>(&monomial_lanes);
        let mut byte_scratch = vec![0xa5; required * row_len + F::BYTES];
        monomial_to_novel_bytes(&mut rows, row_len, plan, &mut byte_scratch).unwrap();
        assert_rows::<F>(&rows, &expected_novel_lanes);

        let novel_lanes: Vec<Vec<F::Elem>> =
            (0..lanes).map(|_| rng.elems::<F>(plan.size())).collect();
        let expected_monomial_lanes: Vec<Vec<F::Elem>> = novel_lanes
            .iter()
            .map(|lane| {
                let mut converted = lane.clone();
                novel_to_monomial(&mut converted, plan).unwrap();
                converted
            })
            .collect();
        let mut rows = pack::<F>(&novel_lanes);
        novel_to_monomial_bytes(&mut rows, row_len, plan, &mut byte_scratch).unwrap();
        assert_rows::<F>(&rows, &expected_monomial_lanes);
    }

    #[test]
    fn scratch_and_simd_batched_variants_match_element_conversions() {
        let mut rng = Rng(0x6a09_e667_f3bc_c909);
        for &(log_size, lanes) in &[(0usize, 1usize), (1, 3), (3, 17), (5, 33)] {
            scratch_and_byte_variants(
                &TransformPlan::<Gf8>::new(1 << log_size).unwrap(),
                lanes,
                &mut rng,
            );
            scratch_and_byte_variants(
                &TransformPlan::<Gf16>::new(1 << log_size).unwrap(),
                lanes,
                &mut rng,
            );
        }
        let cantor = CantorBasis::<Gf16>::build().unwrap();
        scratch_and_byte_variants(
            &TransformPlan::<Gf16>::with_basis(32, &cantor.prefix(5)).unwrap(),
            19,
            &mut rng,
        );
    }

    #[test]
    fn conversions_are_mutual_inverses() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for log_size in 0..=8usize {
            round_trip(
                &TransformPlan::<Gf16>::new(1 << log_size).unwrap(),
                &mut rng,
            );
        }
        for log_size in 0..=8usize {
            round_trip(&TransformPlan::<Gf8>::new(1 << log_size).unwrap(), &mut rng);
        }
    }

    /// The contract that makes the conversion useful: converted coefficients
    /// evaluate, through the core transform, to exactly what Horner gives on
    /// the monomial form at every domain point.
    fn agrees_with_horner<F: ButterflyKernels>(plan: &TransformPlan<F>, rng: &mut Rng) {
        let monomial = rng.elems::<F>(plan.size());
        let mut novel = monomial.clone();
        monomial_to_novel(&mut novel, plan).unwrap();
        plan.forward(&mut novel).unwrap();
        for (index, &value) in novel.iter().enumerate() {
            assert_eq!(
                value,
                horner(&monomial, plan.point_element(index)),
                "{} point {index} of {}",
                F::NAME,
                plan.size()
            );
        }
    }

    #[test]
    fn monomial_coefficients_evaluate_through_the_transform() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for log_size in 0..=8usize {
            agrees_with_horner(
                &TransformPlan::<Gf16>::new(1 << log_size).unwrap(),
                &mut rng,
            );
        }
        for log_size in 0..=8usize {
            agrees_with_horner(&TransformPlan::<Gf8>::new(1 << log_size).unwrap(), &mut rng);
        }
    }

    /// Conversion is basis-relative: it must work over any ordered basis the
    /// plan was built with, not just the bit basis.
    #[test]
    fn conversion_follows_the_plan_basis() {
        let mut rng = Rng(0xdead_beef_0bad_f00d);
        let cantor = CantorBasis::<Gf16>::build().unwrap();
        for log_size in 1..=8usize {
            let plan =
                TransformPlan::<Gf16>::with_basis(1 << log_size, &cantor.prefix(log_size)).unwrap();
            round_trip(&plan, &mut rng);
            agrees_with_horner(&plan, &mut rng);
        }
        // A basis that is neither bit nor Cantor.
        let mixed: Vec<_> = (0..8)
            .map(|index| {
                OrderedBasis::<Gf16>::element(&BitBasis, index)
                    .add(OrderedBasis::<Gf16>::element(&BitBasis, index + 1))
            })
            .collect();
        let plan = TransformPlan::<Gf16>::with_basis(256, &mixed).unwrap();
        round_trip(&plan, &mut rng);
        agrees_with_horner(&plan, &mut rng);
    }

    /// `X_i` has degree exactly `i`: converting the `i`-th novel unit vector
    /// must give a monomial polynomial of degree `i`.
    #[test]
    fn novel_basis_is_degree_triangular() {
        let plan = TransformPlan::<Gf16>::new(64).unwrap();
        for index in 0..plan.size() {
            let mut values = vec![<Gf16 as Field>::Elem::ZERO; plan.size()];
            values[index] = <Gf16 as Field>::Elem::ONE;
            novel_to_monomial(&mut values, &plan).unwrap();
            assert!(!values[index].is_zero(), "X_{index} lost its leading term");
            assert!(
                values[index + 1..].iter().all(|value| value.is_zero()),
                "X_{index} has degree above {index}"
            );
        }
    }

    #[test]
    fn wrong_lengths_are_rejected() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let mut short_values = vec![<Gf16 as Field>::Elem::ZERO; 7];
        let coefficient_error = TransformLengthError {
            expected: 8,
            got: 7,
        };
        assert_eq!(
            novel_to_monomial(&mut short_values, &plan).unwrap_err(),
            coefficient_error
        );
        assert_eq!(
            monomial_to_novel(&mut short_values, &plan).unwrap_err(),
            coefficient_error
        );

        let mut values = vec![<Gf16 as Field>::Elem::ZERO; 8];
        let mut short_scratch = vec![<Gf16 as Field>::Elem::ZERO; 3];
        let scratch_error = TransformLengthError {
            expected: 4,
            got: 3,
        };
        assert_eq!(
            novel_to_monomial_with_scratch(&mut values, &plan, &mut short_scratch).unwrap_err(),
            scratch_error
        );
        assert_eq!(
            monomial_to_novel_with_scratch(&mut values, &plan, &mut short_scratch).unwrap_err(),
            scratch_error
        );

        let mut short_rows = vec![0u8; 15];
        let mut byte_scratch = vec![0u8; 8];
        let byte_coefficient_error = TransformLengthError {
            expected: 16,
            got: 15,
        };
        assert_eq!(
            novel_to_monomial_bytes(&mut short_rows, 2, &plan, &mut byte_scratch).unwrap_err(),
            byte_coefficient_error
        );
        assert_eq!(
            monomial_to_novel_bytes(&mut short_rows, 2, &plan, &mut byte_scratch).unwrap_err(),
            byte_coefficient_error
        );

        let mut rows = vec![0u8; 16];
        let mut short_byte_scratch = vec![0u8; 7];
        let byte_scratch_error = TransformLengthError {
            expected: 8,
            got: 7,
        };
        assert_eq!(
            novel_to_monomial_bytes(&mut rows, 2, &plan, &mut short_byte_scratch).unwrap_err(),
            byte_scratch_error
        );
        assert_eq!(
            monomial_to_novel_bytes(&mut rows, 2, &plan, &mut short_byte_scratch).unwrap_err(),
            byte_scratch_error
        );
    }

    #[test]
    #[should_panic(expected = "row length must be nonzero")]
    fn zero_row_length_panics_before_execution() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        novel_to_monomial_bytes(&mut [], 0, &plan, &mut []).unwrap();
    }

    #[test]
    #[should_panic(expected = "partial trailing element")]
    fn partial_element_row_panics_before_execution() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        monomial_to_novel_bytes(&mut [], 3, &plan, &mut []).unwrap();
    }

    #[test]
    #[should_panic(expected = "coefficient byte length overflow")]
    fn overflowing_byte_geometry_panics_before_execution() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        novel_to_monomial_bytes(&mut [], usize::MAX - 1, &plan, &mut []).unwrap();
    }

    #[test]
    fn inverse_interpolate_recovers_monomial_coefficients() {
        fn check<F: ButterflyKernels>(rng: &mut Rng) {
            for log_size in 1..=7usize {
                let size = 1 << log_size;
                let plan = TransformPlan::<F>::new(size).unwrap();
                let evaluations = rng.elems::<F>(size);
                let row_len = F::BYTES;
                let mut rows = vec![0u8; size * row_len];
                for (point, &evaluation) in evaluations.iter().enumerate() {
                    F::write(
                        &mut rows[point * row_len..(point + 1) * row_len],
                        evaluation,
                    );
                }
                let mut scratch = vec![0u8; conversion_scratch_elements(size) * row_len];
                inverse_interpolate_bytes(&mut rows, row_len, &plan, &mut scratch).unwrap();
                let monomial: Vec<F::Elem> = (0..size)
                    .map(|degree| F::read(&rows[degree * row_len..(degree + 1) * row_len]))
                    .collect();
                for (point, &evaluation) in evaluations.iter().enumerate() {
                    assert_eq!(
                        horner(&monomial, plan.point_element(point)),
                        evaluation,
                        "R(point) != received at size {size} point {point}"
                    );
                }
            }
        }
        let mut rng = Rng(0x9e37_79b9);
        check::<Gf8>(&mut rng);
        check::<Gf16>(&mut rng);
    }

    #[test]
    fn inverse_interpolate_handles_affine_coset() {
        fn check<F: ButterflyKernels>(rng: &mut Rng) {
            for log_size in 1..=6usize {
                let size = 1 << log_size;
                let mut shift_bytes = [0u8; 8];
                shift_bytes[0] = 1 << log_size;
                let shift = F::read(&shift_bytes[..F::BYTES]);
                let plan = crate::shifted::ShiftedPlan::<F>::new(size, shift).unwrap();
                let evaluations = rng.elems::<F>(size);
                let row_len = F::BYTES;
                let mut rows = vec![0u8; size * row_len];
                for (point, &evaluation) in evaluations.iter().enumerate() {
                    F::write(
                        &mut rows[point * row_len..(point + 1) * row_len],
                        evaluation,
                    );
                }
                let mut scratch = vec![0u8; conversion_scratch_elements(size) * row_len];
                inverse_interpolate_bytes(&mut rows, row_len, plan.plan(), &mut scratch).unwrap();
                let monomial: Vec<F::Elem> = (0..size)
                    .map(|degree| F::read(&rows[degree * row_len..(degree + 1) * row_len]))
                    .collect();
                for (point, &evaluation) in evaluations.iter().enumerate() {
                    assert_eq!(
                        horner(&monomial, plan.point_element(point)),
                        evaluation,
                        "affine R(point) != received at size {size} point {point}"
                    );
                }
            }
        }
        let mut rng = Rng(0x6c62_2c73);
        check::<Gf8>(&mut rng);
        check::<Gf16>(&mut rng);
    }
}
