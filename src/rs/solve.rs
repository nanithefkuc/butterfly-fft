//! Targeted small-erasure solver helpers.
//!
//! When only a few systematic points are missing, running two full-domain
//! transforms per block is wasteful: a dense `missing × missing` solve over
//! the received repair equations is cheaper. Both pieces of that path live
//! here — the generator coefficients of one repair point, and a small dense
//! inverse.

use fff::field::Elem;

use crate::core::transform::TransformPlan;
use crate::rs::locator::ErasureLocator;
use crate::rs::tables::RsField;

/// Coefficients expressing the evaluation at `point` as a combination of the
/// evaluations at the systematic points `0..row.len()`.
///
/// This is additive-domain Lagrange interpolation with the systematic
/// locator `Λ(x) = ∏_{d < k} (x ⊕ d_point)`:
///
/// ```text
/// row[d] = Λ(point) / ((point ⊕ d_point) · Λ'(d_point))
/// ```
///
/// Computed in exponents, so one row costs `k` table lookups and no field
/// multiplications or inversions. `locator` must be the systematic locator
/// for `row.len()` systematic points (see
/// [`super::SystematicLocators::get`]).
///
/// # Panics
/// Panics unless `row.len() <= point < plan.size()` and `locator` covers the
/// plan's domain.
pub fn generator_row<F: RsField>(
    plan: &TransformPlan<F>,
    locator: &ErasureLocator<F>,
    point: usize,
    row: &mut [F::Elem],
) {
    assert_eq!(locator.size(), plan.size(), "locator covers another domain");
    assert!(point < plan.size(), "evaluation point out of range");
    assert!(
        row.len() <= point,
        "the evaluation point must lie outside the systematic prefix"
    );
    let tables = F::log_exp();
    let modulus = tables.order();
    let numerator = tables.log(locator.values()[point]);
    for (data, coefficient) in row.iter_mut().enumerate() {
        let difference = tables.log(plan.point_element(point ^ data));
        let derivative = tables.log(locator.derivatives()[data]);
        let exponent = (numerator + 2 * modulus - difference - derivative) % modulus;
        *coefficient = tables.exp(exponent);
    }
}

/// Bytes of workspace [`invert_square_into`] needs for a `size × size`
/// system, in elements.
#[must_use]
pub const fn inverse_scratch_elements(size: usize) -> usize {
    2 * size * size
}

/// Invert a dense `size × size` row-major matrix by Gauss–Jordan
/// elimination, returning whether it was invertible.
///
/// `augmented` is workspace of at least [`inverse_scratch_elements`]
/// elements; `inverse` receives the result and is untouched on a singular
/// input.
///
/// # Panics
/// Panics unless `matrix.len() >= size * size`,
/// `inverse.len() == size * size`, and `augmented` is large enough.
pub fn invert_square_into<E: Elem>(
    matrix: &[E],
    size: usize,
    augmented: &mut [E],
    inverse: &mut [E],
) -> bool {
    assert!(matrix.len() >= size * size, "matrix too small");
    assert_eq!(inverse.len(), size * size, "inverse has the wrong size");
    let stride = size * 2;
    let augmented = &mut augmented[..size * stride];
    augmented.fill(E::ZERO);
    for row in 0..size {
        augmented[row * stride..row * stride + size]
            .copy_from_slice(&matrix[row * size..(row + 1) * size]);
        augmented[row * stride + size + row] = E::ONE;
    }
    for column in 0..size {
        let Some(pivot) = (column..size).find(|&row| !augmented[row * stride + column].is_zero())
        else {
            return false;
        };
        if pivot != column {
            for entry in 0..stride {
                augmented.swap(column * stride + entry, pivot * stride + entry);
            }
        }
        let pivot_inverse = augmented[column * stride + column].inv();
        for entry in column..stride {
            augmented[column * stride + entry] =
                augmented[column * stride + entry].mul(pivot_inverse);
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = augmented[row * stride + column];
            if factor.is_zero() {
                continue;
            }
            for entry in column..stride {
                let pivot_value = augmented[column * stride + entry];
                augmented[row * stride + entry] =
                    augmented[row * stride + entry].add(factor.mul(pivot_value));
            }
        }
    }
    for row in 0..size {
        inverse[row * size..(row + 1) * size]
            .copy_from_slice(&augmented[row * stride + size..(row + 1) * stride]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::alloc::vec;
    use fff::field::Field;
    use fff::{Gf8, Gf16};

    use crate::rs::locator::SystematicLocators;

    type Gf16Elem = <Gf16 as Field>::Elem;

    fn elem(raw: u16) -> Gf16Elem {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(&raw.to_le_bytes());
        Gf16::read(&bytes)
    }

    #[test]
    fn generator_rows_reproduce_the_systematic_encoding() {
        // Ground truth: a codeword built from `systematic` novel-basis
        // coefficients and evaluated over the whole domain. The generator
        // row must be the linear map from its systematic prefix to any
        // other evaluation.
        for (log_size, systematic) in [(3usize, 5usize), (4, 9), (5, 20), (6, 33)] {
            let size = 1usize << log_size;
            let plan = TransformPlan::<Gf16>::new(size).unwrap();
            let locators = SystematicLocators::<Gf16>::new();
            let locator = locators.get(&plan, systematic).unwrap();

            let mut evaluations = vec![Gf16Elem::ZERO; size];
            for (index, slot) in evaluations[..systematic].iter_mut().enumerate() {
                let seed = u16::try_from(index).expect("small index");
                *slot = elem(seed.wrapping_mul(7_919).wrapping_add(3));
            }
            plan.forward(&mut evaluations).unwrap();
            let data = evaluations[..systematic].to_vec();

            let mut row = vec![Gf16Elem::ZERO; systematic];
            for (point, &expected) in evaluations.iter().enumerate().skip(systematic) {
                generator_row(&plan, &locator, point, &mut row);
                let combined = row
                    .iter()
                    .zip(data.iter())
                    .fold(Gf16Elem::ZERO, |accumulator, (&coefficient, &value)| {
                        accumulator.add(coefficient.mul(value))
                    });
                assert_eq!(combined, expected, "point {point} of {size}");
            }
        }
    }

    #[test]
    fn generator_rows_work_over_gf8() {
        let plan = TransformPlan::<Gf8>::new(8).unwrap();
        let locators = SystematicLocators::<Gf8>::new();
        let locator = locators.get(&plan, 5).unwrap();
        let mut row = vec![<Gf8 as Field>::Elem::ZERO; 5];
        for point in 5..8 {
            generator_row(&plan, &locator, point, &mut row);
            assert!(row.iter().all(|coefficient| !coefficient.is_zero()));
        }
    }

    #[test]
    #[should_panic(expected = "outside the systematic prefix")]
    fn rejects_a_systematic_point() {
        let plan = TransformPlan::<Gf16>::new(8).unwrap();
        let locators = SystematicLocators::<Gf16>::new();
        let locator = locators.get(&plan, 5).unwrap();
        let mut row = vec![Gf16Elem::ZERO; 5];
        generator_row(&plan, &locator, 4, &mut row);
    }

    #[test]
    fn inverse_times_matrix_is_the_identity() {
        let mut state = 0x9e37_79b9u32;
        for size in 1..=6 {
            let mut matrix = vec![Gf16Elem::ZERO; size * size];
            loop {
                for entry in &mut matrix {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *entry = elem(
                        state.to_le_bytes()[1..3]
                            .try_into()
                            .map(u16::from_le_bytes)
                            .unwrap(),
                    );
                }
                let mut augmented = vec![Gf16Elem::ZERO; inverse_scratch_elements(size)];
                let mut inverse = vec![Gf16Elem::ZERO; size * size];
                if !invert_square_into(&matrix, size, &mut augmented, &mut inverse) {
                    continue;
                }
                for row in 0..size {
                    for column in 0..size {
                        let entry = (0..size).fold(Gf16Elem::ZERO, |accumulator, index| {
                            accumulator
                                .add(matrix[row * size + index].mul(inverse[index * size + column]))
                        });
                        let expected = if row == column {
                            Gf16Elem::ONE
                        } else {
                            Gf16Elem::ZERO
                        };
                        assert_eq!(entry, expected, "size {size} at ({row},{column})");
                    }
                }
                break;
            }
        }
    }

    #[test]
    fn detects_singular_systems() {
        let size = 3;
        // Third row is the XOR of the first two.
        let matrix = [
            elem(1),
            elem(2),
            elem(3),
            elem(4),
            elem(8),
            elem(12),
            elem(5),
            elem(10),
            elem(15),
        ];
        let mut augmented = vec![Gf16Elem::ZERO; inverse_scratch_elements(size)];
        let mut inverse = vec![Gf16Elem::ONE; size * size];
        assert!(!invert_square_into(
            &matrix,
            size,
            &mut augmented,
            &mut inverse
        ));
        // The zero matrix is singular too.
        let zero = vec![Gf16Elem::ZERO; size * size];
        assert!(!invert_square_into(
            &zero,
            size,
            &mut augmented,
            &mut inverse
        ));
    }
}
