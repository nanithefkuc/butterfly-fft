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

use ::alloc::{vec, vec::Vec};

use fff::field::Elem;

use crate::core::factors;
use crate::core::kernel::ButterflyKernels;
use crate::core::transform::TransformPlan;
use crate::error::TransformLengthError;

/// The normalized subspace polynomials of a plan's basis, as coefficient
/// lists of `x^(2^j)` (low `j` first) with the normalizer folded in.
///
/// `chain[d]` has `d + 1` entries and leading term `x^(2^d)`, so it is the
/// factor splitting a dimension-`(d+1)` node.
fn normalized_chain<F: ButterflyKernels>(plan: &TransformPlan<F>) -> Vec<Vec<F::Elem>> {
    factors::subspace_polynomials(plan.basis())
        .expect("a plan's basis is independent by construction")
        .iter()
        .map(|polynomial| {
            polynomial
                .coefficients
                .iter()
                .map(|&coefficient| coefficient.mul(polynomial.normalizer_inverse))
                .collect()
        })
        .collect()
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
    let chain = normalized_chain(plan);
    let mut scratch = vec![F::Elem::ZERO; plan.size() / 2];
    novel_to_monomial_node(coefficients, &chain, plan.log_size(), &mut scratch);
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
    let chain = normalized_chain(plan);
    let mut scratch = vec![F::Elem::ZERO; plan.size() / 2];
    monomial_to_novel_node(coefficients, &chain, plan.log_size(), &mut scratch);
    Ok(())
}

fn check_len(got: usize, expected: usize) -> Result<(), TransformLengthError> {
    if got == expected {
        Ok(())
    } else {
        Err(TransformLengthError { expected, got })
    }
}

/// Bottom-up: convert both halves, then fold them together as
/// `f = f_lo + W̄_{dimension-1}·f_hi`.
fn novel_to_monomial_node<E: Elem>(
    values: &mut [E],
    chain: &[Vec<E>],
    dimension: usize,
    scratch: &mut [E],
) {
    if dimension == 0 {
        return;
    }
    let half = values.len() / 2;
    {
        let (low, high) = values.split_at_mut(half);
        novel_to_monomial_node(low, chain, dimension - 1, scratch);
        novel_to_monomial_node(high, chain, dimension - 1, scratch);
    }

    // `f_hi` currently sits in the upper half as coefficients of degree
    // `0..half`; the product scatters it across the whole range, including
    // over itself, so it is lifted out first.
    let factor = &chain[dimension - 1];
    let (product_source, _) = scratch.split_at_mut(half);
    product_source.copy_from_slice(&values[half..]);
    values[half..].fill(E::ZERO);
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
fn monomial_to_novel_node<E: Elem>(
    values: &mut [E],
    chain: &[Vec<E>],
    dimension: usize,
    scratch: &mut [E],
) {
    if dimension == 0 {
        return;
    }
    let length = values.len();
    let half = length / 2;
    let factor = &chain[dimension - 1];
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
    monomial_to_novel_node(low, chain, dimension - 1, scratch);
    monomial_to_novel_node(high, chain, dimension - 1, scratch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{BitBasis, CantorBasis, OrderedBasis};
    use fff::field::Field;
    use fff::{Gf8, Gf16};

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
    fn wrong_length_is_rejected() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let mut values = vec![<Gf16 as Field>::Elem::ZERO; 7];
        assert_eq!(
            novel_to_monomial(&mut values, &plan).unwrap_err(),
            TransformLengthError {
                expected: 8,
                got: 7
            }
        );
        assert!(monomial_to_novel(&mut values, &plan).is_err());
    }
}
