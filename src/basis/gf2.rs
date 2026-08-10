//! GF(2) linear algebra on field elements viewed as bit vectors.
//!
//! Every element of `GF(2^m)` is an `m`-bit vector over GF(2) in its stable
//! little-endian byte encoding, so change-of-basis and the additive
//! quadratic `x² + x = c` are ordinary linear algebra. `m ≤ 64` for every
//! field this crate supports, which is what lets a whole vector live in a
//! `u64` and a whole elimination step be one XOR.

use ::alloc::vec::Vec;

use fgf::field::{Elem, Field};

/// The element's GF(2) coordinate vector in the bit basis: exactly its
/// stable little-endian byte encoding, zero-extended.
pub(crate) fn bits_of<F: Field>(value: F::Elem) -> u64 {
    debug_assert!(F::BYTES <= 8);
    let mut bytes = [0u8; 8];
    F::write(&mut bytes[..F::BYTES], value);
    u64::from_le_bytes(bytes)
}

/// Inverse of [`bits_of`].
pub(crate) fn elem_of<F: Field>(bits: u64) -> F::Elem {
    debug_assert!(F::BYTES <= 8);
    let bytes = bits.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

/// A GF(2) linear system in reduced form, solved by XOR elimination.
///
/// Columns are supplied once; each is reduced against the pivots already
/// held and, if it survives, becomes the pivot for its own leading bit. The
/// companion word records which columns were combined, so a solve returns
/// the coefficient vector directly rather than just a membership answer.
pub(crate) struct XorSolver {
    /// `pivots[bit] = (value, combination)`, `value`'s leading bit is `bit`.
    pivots: [Option<(u64, u64)>; 64],
}

impl XorSolver {
    /// Reduce `columns` (at most 64) into pivot form.
    pub(crate) fn new(columns: &[u64]) -> Self {
        debug_assert!(columns.len() <= 64);
        let mut pivots = [None; 64];
        for (index, &column) in columns.iter().enumerate() {
            let mut value = column;
            let mut combination = 1u64 << index;
            while value != 0 {
                let leading = 63 - value.leading_zeros() as usize;
                match pivots[leading] {
                    None => {
                        pivots[leading] = Some((value, combination));
                        break;
                    }
                    Some((pivot_value, pivot_combination)) => {
                        value ^= pivot_value;
                        combination ^= pivot_combination;
                    }
                }
            }
        }
        Self { pivots }
    }

    /// The coefficient vector `v` with `⊕_{j ∈ v} columns[j] == target`, or
    /// `None` if `target` is outside the column span.
    pub(crate) fn solve(&self, target: u64) -> Option<u64> {
        let mut value = target;
        let mut combination = 0u64;
        while value != 0 {
            let leading = 63 - value.leading_zeros() as usize;
            let (pivot_value, pivot_combination) = self.pivots[leading]?;
            value ^= pivot_value;
            combination ^= pivot_combination;
        }
        Some(combination)
    }
}

/// Whether `elements` are GF(2)-linearly independent.
pub(crate) fn independent<F: Field>(elements: &[F::Elem]) -> bool {
    let columns: Vec<u64> = elements.iter().copied().map(bits_of::<F>).collect();
    let solver = XorSolver::new(&columns);
    // Independence ⇔ every column contributed a pivot ⇔ the pivot count
    // equals the column count.
    solver.pivots.iter().filter(|slot| slot.is_some()).count() == columns.len()
}

/// Solve `x² + x = value` over the field, returning the solution with the
/// smaller bit pattern, or `None` when the equation has no root.
///
/// `x ↦ x² + x` is GF(2)-linear in characteristic two (the cross terms
/// vanish), so this is a linear solve, not a root search. Its kernel is
/// `{0, 1}`, so solutions come in pairs differing by one.
pub(crate) fn solve_quadratic<F: Field>(value: F::Elem) -> Option<F::Elem> {
    let columns: Vec<u64> = (0..F::BITS as usize)
        .map(|bit| {
            let basis_element = elem_of::<F>(1u64 << bit);
            bits_of::<F>(basis_element.square().add(basis_element))
        })
        .collect();
    let solution = XorSolver::new(&columns).solve(bits_of::<F>(value))?;
    let root = elem_of::<F>(solution);
    let sibling = root.add(F::Elem::ONE);
    debug_assert_eq!(root.square().add(root), value);
    Some(if bits_of::<F>(root) <= bits_of::<F>(sibling) {
        root
    } else {
        sibling
    })
}
