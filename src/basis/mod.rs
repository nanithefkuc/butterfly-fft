//! Ordered field bases and coefficient-basis conversion.
//!
//! Two different things are called a "basis" here, and the module names them
//! apart deliberately.
//!
//! # Domain bases: ordered GF(2)-bases of the field
//!
//! An ordered basis `β = (β_0, …, β_{m-1})` fixes both the transform's
//! evaluation points (point `i` is the XOR of the `β_j` at the set bits of
//! `i`) and its subspace polynomials. Two providers ship:
//!
//! - [`BitBasis`] — `β_i` is the element with bit pattern `1 << i`. Point
//!   order is bit-pattern order; this is the default, and the layout every
//!   byte-row consumer expects.
//! - [`CantorBasis`] — `v_0 = 1` and `v_i² + v_i = v_{i-1}` (Cantor 1989),
//!   the basis of the LCH fast-FFT literature. Every normalizer is one and
//!   the coefficients of `W_k` are row `k` of Pascal's triangle modulo two.
//!   Thus `W_k = Σ_j C(k,j)·x^(2^j)`; it is the binomial
//!   `x^(2^k) + x` only when `k` is a power of two.
//!
//! [`CoordinateMap`] is the change of basis between the two views: an
//! element's bit pattern *is* its bit-basis coordinate vector, so converting
//! to any other ordered basis is one GF(2) matrix apply.
//!
//! # Coefficient bases: monomial ↔ novel
//!
//! [`crate::core`] speaks only the novel basis
//! `X_i(x) = ∏_j W̄_j(x)^{bit_j(i)}`. Interpolation-side consumers (GS
//! weighted-degree bookkeeping, Hasse derivatives) work in the monomial
//! basis. [`monomial_to_novel`] and [`novel_to_monomial`] convert between
//! them in `O(n log² n)`.

mod cantor;
mod convert;
mod gf2;

use ::alloc::vec::Vec;

use fgf::field::{Elem, Field};

pub use cantor::CantorBasis;
#[cfg(feature = "std")]
pub use cantor::cantor_basis;
pub use convert::{
    conversion_scratch_elements, monomial_to_novel, monomial_to_novel_bytes,
    monomial_to_novel_with_scratch, novel_to_monomial, novel_to_monomial_bytes,
    novel_to_monomial_with_scratch,
};

#[cfg(feature = "rs")]
pub(crate) use gf2::bits_of;
pub(crate) use gf2::independent;

/// An ordered GF(2)-basis of a field.
///
/// Implementors expose `β_i` for `i < bits()`; a transform of dimension `k`
/// consumes the prefix `β_0 … β_{k-1}`, whose span is the evaluation domain.
pub trait OrderedBasis<F: Field>: Send + Sync {
    /// The `index`-th basis element `β_index`.
    ///
    /// # Panics
    /// May panic if `index >= bits()`.
    fn element(&self, index: usize) -> F::Elem;

    /// Number of basis elements available, at most `F::BITS`.
    fn bits(&self) -> usize;

    /// The prefix `β_0 … β_{count-1}`, the form plan constructors take.
    ///
    /// # Panics
    /// Panics if `count > bits()`.
    fn prefix(&self, count: usize) -> Vec<F::Elem> {
        assert!(count <= self.bits(), "basis prefix longer than the basis");
        (0..count).map(|index| self.element(index)).collect()
    }
}

/// The default ordered basis: `β_i` is the element whose little-endian byte
/// representation is `1 << i`.
///
/// Under this basis transform point `i` is the field element whose bytes
/// hold the value `i`, which is the row order every byte-row consumer
/// assumes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitBasis;

impl<F: Field> OrderedBasis<F> for BitBasis {
    #[inline]
    fn element(&self, index: usize) -> F::Elem {
        assert!(index < F::BITS as usize, "bit basis index out of range");
        gf2::elem_of::<F>(1u64 << index)
    }

    #[inline]
    fn bits(&self) -> usize {
        F::BITS as usize
    }
}

/// The change of basis between an ordered basis and the field's bit pattern.
///
/// Holds both directions as GF(2) matrices over `F::BITS` bit-vector
/// columns, so each conversion is a fixed sequence of XORs rather than a
/// solve. The bit basis is the identity map, which is what makes this the
/// bit ↔ *anything* converter.
#[derive(Clone, Debug)]
pub struct CoordinateMap<F: Field> {
    /// Column `j` is the bit pattern of `β_j`: coordinates → element.
    forward: Vec<u64>,
    /// Column `b` holds the coordinates of the element with bit pattern
    /// `1 << b`: element → coordinates.
    inverse: Vec<u64>,
    marker: ::core::marker::PhantomData<fn() -> F>,
}

impl<F: Field> CoordinateMap<F> {
    /// Build the map for an ordered basis, given as its elements.
    ///
    /// Returns `None` unless `basis` is a full GF(2)-basis of the field:
    /// exactly `F::BITS` linearly independent elements. A shorter
    /// independent set spans a subspace and has no inverse map.
    #[must_use]
    pub fn new(basis: &[F::Elem]) -> Option<Self> {
        if basis.len() != F::BITS as usize {
            return None;
        }
        let forward: Vec<u64> = basis.iter().copied().map(gf2::bits_of::<F>).collect();
        let solver = gf2::XorSolver::new(&forward);
        let inverse: Vec<u64> = (0..F::BITS as usize)
            .map(|bit| solver.solve(1u64 << bit))
            .collect::<Option<_>>()?;
        Some(Self {
            forward,
            inverse,
            marker: ::core::marker::PhantomData,
        })
    }

    /// Build the map for an [`OrderedBasis`] provider.
    ///
    /// Returns `None` unless the provider offers a full basis of the field.
    #[must_use]
    pub fn of(basis: &impl OrderedBasis<F>) -> Option<Self> {
        if basis.bits() != F::BITS as usize {
            return None;
        }
        Self::new(&basis.prefix(F::BITS as usize))
    }

    /// The element with the given coordinates over this basis:
    /// `⊕_{j ∈ coordinates} β_j`.
    #[must_use]
    pub fn to_element(&self, coordinates: u64) -> F::Elem {
        gf2::elem_of::<F>(apply(&self.forward, coordinates))
    }

    /// The coordinates of `element` over this basis, the inverse of
    /// [`CoordinateMap::to_element`].
    #[must_use]
    pub fn to_coordinates(&self, element: F::Elem) -> u64 {
        apply(&self.inverse, gf2::bits_of::<F>(element))
    }
}

/// Apply a GF(2) matrix given as columns: XOR the columns selected by the
/// set bits of `vector`.
fn apply(columns: &[u64], vector: u64) -> u64 {
    let mut result = 0u64;
    let mut remaining = vector;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as usize;
        debug_assert!(bit < columns.len(), "coordinate outside the basis");
        result ^= columns[bit];
        remaining &= remaining - 1;
    }
    result
}

/// The transform point selected by `index` over an ordered basis: the XOR of
/// the basis elements at the set bits of `index`.
#[must_use]
pub fn point_of<F: Field>(basis: &[F::Elem], index: usize) -> F::Elem {
    let mut element = F::Elem::ZERO;
    let mut remaining = index;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as usize;
        debug_assert!(bit < basis.len());
        element = element.add(basis[bit]);
        remaining &= remaining - 1;
    }
    element
}

#[cfg(test)]
mod tests {
    use super::*;
    use fgf::{FanPaar16, Gf8, Gf16, Gf32, Gf64};

    fn bit_basis_is_identity_map<F: Field>() {
        let map = CoordinateMap::<F>::of(&BitBasis).expect("bit basis is a full basis");
        for bit in 0..F::BITS as usize {
            let pattern = 1u64 << bit;
            assert_eq!(gf2::bits_of::<F>(map.to_element(pattern)), pattern);
            assert_eq!(map.to_coordinates(gf2::elem_of::<F>(pattern)), pattern);
        }
    }

    #[test]
    fn bit_basis_coordinates_are_bit_patterns() {
        bit_basis_is_identity_map::<Gf8>();
        bit_basis_is_identity_map::<Gf16>();
        bit_basis_is_identity_map::<Gf32>();
        bit_basis_is_identity_map::<Gf64>();
        bit_basis_is_identity_map::<FanPaar16>();
    }

    #[test]
    fn short_or_dependent_bases_have_no_coordinate_map() {
        let short: Vec<_> = (0..4)
            .map(|i| OrderedBasis::<Gf8>::element(&BitBasis, i))
            .collect();
        assert!(CoordinateMap::<Gf8>::new(&short).is_none());

        let mut dependent: Vec<_> = (0..8)
            .map(|i| OrderedBasis::<Gf8>::element(&BitBasis, i))
            .collect();
        dependent[7] = dependent[0].add(dependent[1]);
        assert!(CoordinateMap::<Gf8>::new(&dependent).is_none());
    }

    #[test]
    fn point_of_matches_bit_pattern_under_the_bit_basis() {
        fn check<F: Field>() {
            let basis: Vec<_> = (0..F::BITS as usize)
                .map(|i| OrderedBasis::<F>::element(&BitBasis, i))
                .collect();
            for index in [0usize, 1, 2, 3, 5, 17, 63, 200] {
                let masked = index & ((1usize << basis.len().min(16)) - 1);
                assert_eq!(
                    point_of::<F>(&basis, masked),
                    gf2::elem_of::<F>(masked as u64)
                );
            }
        }
        check::<Gf8>();
        check::<Gf16>();
        check::<Gf32>();
    }
}
