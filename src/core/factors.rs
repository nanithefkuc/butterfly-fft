//! Subspace polynomials and twiddle factor tables.
//!
//! Normalized subspace polynomials `W̄_k` over nested additive subspaces of
//! the field (Lin–Chung–Han novel-basis construction), their per-recursion-
//! node evaluations in binary-heap layout (the twiddle table), and the
//! derivative table powering formal derivatives in the novel basis.
//!
//! Given an ordered GF(2)-basis `β_0, …` with nested subspaces
//! `V_k = span{β_0..β_{k-1}}`:
//!
//! - `W_0(x) = x`, `W_{k+1}(x) = W_k(x)² + W_k(β_k)·W_k(x)` — the monic
//!   linearized polynomial vanishing on `V_k`, with `k+1` nonzero
//!   coefficients at positions `x^(2^j)`.
//! - `W̄_k = W_k / W_k(β_k)` is constant on each affine half-subspace and
//!   differs by exactly one between sibling halves, giving the butterfly
//!   `f = f₀ + W̄·f₁`.
//! - The twiddle at recursion node `ν` (depth `d`, coset shift `s`) is
//!   `W̄_d(s)`; the table precomputes the whole recursion tree once.

use ::alloc::{vec, vec::Vec};

use fff::field::{Elem, Field};

/// A normalized subspace polynomial, stored as the coefficients of `x^(2^j)`
/// (low `j` first) plus the inverse of the normalizer `W_k(β_k)`.
#[derive(Clone, Debug)]
pub(crate) struct NormalizedSubspacePolynomial<E> {
    /// Coefficients of `x^(2^j)`, low exponent first.
    pub(crate) coefficients: Vec<E>,
    /// `1 / W_k(β_k)`.
    pub(crate) normalizer_inverse: E,
    /// Coefficients with `normalizer_inverse` folded in.
    pub(crate) normalized_coefficients: Vec<E>,
}

impl<E: Elem> NormalizedSubspacePolynomial<E> {
    /// Evaluate the normalized polynomial at `value` in `O(log degree)`.
    pub(crate) fn evaluate(&self, value: E) -> E {
        let mut power = value;
        let mut result = E::ZERO;
        for &coefficient in &self.coefficients {
            result = result.add(scale(coefficient, power));
            power = power.square();
        }
        scale(self.normalizer_inverse, result)
    }
}

/// Multiply while preserving the zero/one cases as pure selection. Besides
/// avoiding needless work generally, this makes Cantor factor-table
/// construction multiplication-free: all its coefficients and normalizers
/// are zero or one.
fn scale<E: Elem>(coefficient: E, value: E) -> E {
    if coefficient.is_zero() {
        E::ZERO
    } else if coefficient.is_one() {
        value
    } else {
        coefficient.mul(value)
    }
}

/// Evaluate an unnormalized linearized polynomial at `value`.
pub(crate) fn evaluate_linearized<E: Elem>(coefficients: &[E], value: E) -> E {
    let mut power = value;
    let mut result = E::ZERO;
    for &coefficient in coefficients {
        result = result.add(scale(coefficient, power));
        power = power.square();
    }
    result
}

/// The normalized subspace polynomials `W̄_0 … W̄_{n-1}` for the ordered
/// basis `β_0 … β_{n-1}`, or `None` if the basis elements are linearly
/// dependent (a zero normalizer).
pub(crate) fn subspace_polynomials<E: Elem>(
    basis: &[E],
) -> Option<Vec<NormalizedSubspacePolynomial<E>>> {
    let mut result = Vec::with_capacity(basis.len());
    let mut coefficients = vec![E::ONE]; // W_0(x) = x
    for (dimension, &basis_element) in basis.iter().enumerate() {
        let normalizer = evaluate_linearized(&coefficients, basis_element);
        if normalizer.is_zero() {
            return None;
        }
        let normalizer_inverse = normalizer.inv();
        result.push(NormalizedSubspacePolynomial {
            coefficients: coefficients.clone(),
            normalizer_inverse,
            normalized_coefficients: coefficients
                .iter()
                .map(|&coefficient| scale(normalizer_inverse, coefficient))
                .collect(),
        });

        if dimension + 1 != basis.len() {
            // W_{i+1}(x) = W_i(x)^2 + W_i(β_i)·W_i(x).
            let mut next = vec![E::ZERO; coefficients.len() + 1];
            for (index, &coefficient) in coefficients.iter().enumerate() {
                next[index] = next[index].add(scale(normalizer, coefficient));
                next[index + 1] = next[index + 1].add(coefficient.square());
            }
            coefficients = next;
        }
    }
    Some(result)
}

/// Fill the twiddle table recursively: node `ν` at depth `dimension` with
/// coset shift `shift` gets `W̄_dimension(shift)`. The high child shifts the
/// coset by `β_{dimension-1}`, the basis element splitting this level.
pub(crate) fn fill_factors<E: Elem>(
    factors: &mut [E],
    polynomials: &[NormalizedSubspacePolynomial<E>],
    basis: &[E],
    node: usize,
    dimension: usize,
    shift: E,
) {
    factors[node] = polynomials[dimension - 1].evaluate(shift);
    if dimension > 1 {
        fill_factors(factors, polynomials, basis, node * 2, dimension - 1, shift);
        fill_factors(
            factors,
            polynomials,
            basis,
            node * 2 + 1,
            dimension - 1,
            shift.add(basis[dimension - 1]),
        );
    }
}

/// The twiddle and derivative tables for one transform plan.
///
/// `factors` uses binary-heap node layout (root at index 1, children of `ν`
/// at `2ν` and `2ν+1`); `derivative_factors[j]` is the formal derivative of
/// `W̄_j` (a constant, since `W̄_j` is linearized).
#[derive(Clone, Debug)]
pub struct FactorTable<F: Field> {
    pub(crate) factors: Vec<F::Elem>,
    pub(crate) derivative_factors: Vec<F::Elem>,
    /// Normalized subspace polynomials retained for coefficient conversion.
    pub(crate) polynomials: Vec<NormalizedSubspacePolynomial<F::Elem>>,
}

impl<F: Field> FactorTable<F> {
    /// Build the table for a `2^log_size` domain over `basis` (at least
    /// `log_size` elements, linearly independent), evaluating over the coset
    /// `shift + V_log_size` (`shift` zero for the plain subspace transform).
    ///
    /// Returns `None` if the basis prefix is linearly dependent.
    pub(crate) fn build(log_size: usize, basis: &[F::Elem], shift: F::Elem) -> Option<Self> {
        debug_assert!(basis.len() >= log_size);
        let polynomials = subspace_polynomials(&basis[..log_size])?;
        let derivative_factors = polynomials
            .iter()
            .map(|polynomial| polynomial.normalized_coefficients[0])
            .collect();
        let mut factors = vec![F::Elem::ZERO; 1 << log_size];
        if log_size != 0 {
            fill_factors(&mut factors, &polynomials, basis, 1, log_size, shift);
        }
        Some(Self {
            factors,
            derivative_factors,
            polynomials,
        })
    }
}

#[cfg(feature = "internals")]
impl<F: Field> FactorTable<F> {
    /// Binary-heap twiddle values; index zero is unused.
    #[must_use]
    pub fn factors(&self) -> &[F::Elem] {
        &self.factors
    }

    /// Formal derivatives of the normalized subspace polynomials.
    #[must_use]
    pub fn derivative_factors(&self) -> &[F::Elem] {
        &self.derivative_factors
    }
}

/// The element whose little-endian byte representation is `1 << index`: the
/// `index`-th bit-basis element `β_index`.
pub(crate) fn bit_basis_element<F: Field>(index: usize) -> F::Elem {
    debug_assert!(index < F::BITS as usize);
    let mut bytes = [0u8; 8];
    bytes[index / 8] = 1 << (index % 8);
    F::read(&bytes[..F::BYTES])
}

/// The first `count` bit-basis elements `β_0 … β_{count-1}`.
pub(crate) fn bit_basis<F: Field>(count: usize) -> Vec<F::Elem> {
    (0..count).map(bit_basis_element::<F>).collect()
}

/// The field element whose little-endian bytes hold the value `index`:
/// transform point `index` under the bit basis.
#[cfg(test)]
pub(crate) fn element_from_index<F: Field>(index: usize) -> F::Elem {
    let mut bytes = [0u8; 8];
    bytes[..F::BYTES].copy_from_slice(&index.to_le_bytes()[..F::BYTES]);
    F::read(&bytes[..F::BYTES])
}

/// The subspace points of `V_k` under the bit basis: elements with bit
/// pattern `< 2^k`.
#[cfg(test)]
pub(crate) fn subspace_points<F: Field>(dimension: usize) -> Vec<F::Elem> {
    (0..(1usize << dimension))
        .map(element_from_index::<F>)
        .collect()
}

/// Whether `elements` are GF(2)-linearly independent as bit vectors.
///
/// XOR-basis elimination: each pivot carries a distinct highest set bit;
/// `min(v, v ^ pivot)` clears that bit from `v` exactly when it was set.
pub(crate) fn linearly_independent<F: Field>(elements: &[F::Elem]) -> bool {
    let mut pivots: Vec<u128> = Vec::with_capacity(elements.len());
    for &element in elements {
        let mut bytes = [0u8; 16];
        F::write(&mut bytes[..F::BYTES], element);
        let mut vector = u128::from_le_bytes(bytes);
        for &pivot in &pivots {
            vector = vector.min(vector ^ pivot);
        }
        if vector == 0 {
            return false;
        }
        for pivot in &mut pivots {
            *pivot = (*pivot).min(*pivot ^ vector);
        }
        pivots.push(vector);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use fff::{Gf8, Gf16};

    /// Ground truth: `W̄_k` vanishes on every point of `V_k`.
    fn vanishes_on_subspace<F: Field>() {
        for dimension in 1..=5usize {
            let basis = bit_basis::<F>(dimension + 1);
            let polynomials = subspace_polynomials(&basis).unwrap();
            let top = &polynomials[dimension];
            for point in subspace_points::<F>(dimension) {
                assert!(
                    top.evaluate(point).is_zero(),
                    "W̄_{dimension} nonzero at subspace point"
                );
            }
        }
    }

    #[test]
    fn subspace_polynomials_vanish() {
        vanishes_on_subspace::<Gf8>();
        vanishes_on_subspace::<Gf16>();
    }

    /// Ground truth: the normalizer `W_k(β_k)` is invertible and correctly
    /// recorded, i.e. `W_k(β_k) · normalizer_inverse == 1`.
    fn normalizer_inverts<F: Field>() {
        for dimension in 1..=6usize {
            let basis = bit_basis::<F>(dimension);
            let polynomials = subspace_polynomials(&basis).unwrap();
            for (k, polynomial) in polynomials.iter().enumerate() {
                let normalizer = evaluate_linearized(&polynomial.coefficients, basis[k]);
                assert!(
                    normalizer.mul(polynomial.normalizer_inverse).is_one(),
                    "bad normalizer at level {k}"
                );
            }
        }
    }

    #[test]
    fn normalizers_match_definition() {
        normalizer_inverts::<Gf8>();
        normalizer_inverts::<Gf16>();
    }

    #[test]
    fn bit_basis_elements_are_bit_patterns() {
        fn check<F: Field>() {
            for index in 0..F::BITS as usize {
                let element = bit_basis_element::<F>(index);
                assert_eq!(element, element_from_index::<F>(1 << index));
            }
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    #[test]
    fn independence_detection() {
        fn check<F: Field>() {
            assert!(linearly_independent::<F>(&bit_basis::<F>(F::BITS as usize)));
            assert!(!linearly_independent::<F>(&[F::Elem::ZERO]));
            // β_0, β_1, β_0^β_1 is dependent.
            let mut dependent = bit_basis::<F>(2);
            dependent.push(dependent[0].add(dependent[1]));
            assert!(!linearly_independent::<F>(&dependent));
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    #[test]
    fn dependent_basis_yields_no_polynomials() {
        fn check<F: Field>() {
            let basis = bit_basis::<F>(2);
            let dependent = [basis[0], basis[0]];
            assert!(subspace_polynomials(&dependent).is_none());
        }
        check::<Gf8>();
        check::<Gf16>();
    }

    fn random_element<F: Field>(state: &mut u32) -> F::Elem {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut bytes = [0u8; 8];
        bytes[..F::BYTES].copy_from_slice(&state.to_le_bytes()[..F::BYTES]);
        F::read(&bytes[..F::BYTES])
    }

    /// The normalized-polynomial property the butterfly rests on: `W̄_d`
    /// differs by exactly one between sibling half-subspaces,
    /// `W̄_d(s) + W̄_d(s + β_d) == 1` for every shift `s`.
    #[test]
    fn normalized_siblings_differ_by_one() {
        fn check<F: Field>(state: &mut u32) {
            for dimension in 1..=5usize {
                let basis = bit_basis::<F>(dimension + 1);
                let polynomials = subspace_polynomials(&basis).unwrap();
                let poly = &polynomials[dimension];
                for _ in 0..8 {
                    let shift = random_element::<F>(state);
                    let sum = poly
                        .evaluate(shift)
                        .add(poly.evaluate(shift.add(basis[dimension])));
                    assert!(sum.is_one(), "sibling property failed at level {dimension}");
                }
            }
        }
        check::<Gf8>(&mut 0xdead_beef);
        check::<Gf16>(&mut 0x0bad_f00d);
    }

    /// Shift propagation through `fill_factors`: the root twiddle is
    /// `W̄_log(shift)`, and children thread the coset shift as expected.
    #[test]
    fn shifted_table_matches_polynomial_evaluations() {
        fn check<F: Field>() {
            let log_size = 4;
            let basis = bit_basis::<F>(log_size);
            let polynomials = subspace_polynomials(&basis).unwrap();
            // The unshifted root twiddle vanishes (zero lies in the subspace).
            let plain = FactorTable::<F>::build(log_size, &basis, F::Elem::ZERO).unwrap();
            assert!(plain.factors[1].is_zero());
            // An arbitrary coset shift lands at the root and propagates.
            let shift = basis[0].add(basis[2]);
            let table = FactorTable::<F>::build(log_size, &basis, shift).unwrap();
            assert_eq!(
                table.factors[1],
                polynomials[log_size - 1].evaluate(shift),
                "root twiddle"
            );
            assert_eq!(
                table.factors[2],
                polynomials[log_size - 2].evaluate(shift),
                "low child"
            );
            assert_eq!(
                table.factors[3],
                polynomials[log_size - 2].evaluate(shift.add(basis[log_size - 1])),
                "high child"
            );
        }
        check::<Gf8>();
        check::<Gf16>();
    }
}
