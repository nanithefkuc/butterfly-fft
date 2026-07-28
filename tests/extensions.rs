#![cfg(feature = "std")]

//! The P4 acceptance scenario, end to end.
//!
//! A consumer holds a polynomial in the **monomial** basis and an
//! **arbitrary affine subspace** `α + span(δ)` of the field, and wants its
//! values at every point of that subspace in `O(n log n)`. cafft expresses
//! that as: convert the coefficients to the novel basis over `δ`, then run
//! a shifted forward transform. Nothing here evaluates point by point.
//!
//! Ground truth throughout is Horner over the monomial coefficients, which
//! shares no code with the transform.

use cafft::basis::{
    BitBasis, CantorBasis, CoordinateMap, OrderedBasis, monomial_to_novel, novel_to_monomial,
};
use cafft::core::transform::TransformPlan;
use cafft::shifted::ShiftedPlan;
use fff::field::{Elem, Field};
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

fn horner<E: Elem>(coefficients: &[E], point: E) -> E {
    coefficients
        .iter()
        .rev()
        .fold(E::ZERO, |accumulator, &coefficient| {
            accumulator.mul(point).add(coefficient)
        })
}

/// A random affine subspace: an independent direction basis of `dimension`
/// elements plus a shift. Built by picking elements and rejecting dependent
/// ones, so the directions are genuinely arbitrary — not a bit-basis prefix.
fn random_subspace<F: Field>(rng: &mut Rng, dimension: usize) -> (Vec<F::Elem>, F::Elem) {
    let map = CoordinateMap::<F>::of(&BitBasis).expect("bit basis is full");
    let mut directions: Vec<F::Elem> = Vec::new();
    let mut spanned: Vec<u64> = Vec::new();
    while directions.len() < dimension {
        let candidate = rng.elem::<F>();
        let mut reduced = map.to_coordinates(candidate);
        for &pivot in &spanned {
            reduced = reduced.min(reduced ^ pivot);
        }
        if reduced == 0 {
            continue;
        }
        for pivot in &mut spanned {
            *pivot = (*pivot).min(*pivot ^ reduced);
        }
        spanned.push(reduced);
        directions.push(candidate);
    }
    (directions, rng.elem::<F>())
}

/// **The acceptance scenario.** Evaluate a monomial-basis polynomial over an
/// arbitrary affine subspace via convert → shifted forward.
fn evaluate_over_affine_subspace<F: cafft::core::kernel::ButterflyKernels>(seed: u64) {
    let mut rng = Rng(seed);
    for dimension in 1..=7usize {
        let size = 1 << dimension;
        let (directions, shift) = random_subspace::<F>(&mut rng, dimension);
        let plan = ShiftedPlan::<F>::from_elements(size, &directions, shift).unwrap();

        // deg < size, in the monomial basis.
        let monomial = rng.elems::<F>(size);

        // Convert over the coset's direction basis, then transform.
        let mut values = monomial.clone();
        monomial_to_novel(&mut values, plan.plan()).unwrap();
        plan.forward(&mut values).unwrap();

        for (index, &value) in values.iter().enumerate() {
            let point = plan.point_element(index);
            assert_eq!(
                value,
                horner(&monomial, point),
                "{} dimension {dimension} point {index}",
                F::NAME
            );
        }

        // The point set really is the affine subspace: `size` distinct
        // points, all of the form shift ⊕ (GF(2) combination of directions).
        let mut points = plan.points();
        assert_eq!(points[0], shift);
        points.sort_unstable_by_key(|point| {
            let mut bytes = [0u8; 8];
            F::write(&mut bytes[..F::BYTES], *point);
            u64::from_le_bytes(bytes)
        });
        points.dedup();
        assert_eq!(points.len(), size, "coset points are not distinct");
    }
}

#[test]
fn monomial_polynomial_over_an_arbitrary_affine_subspace() {
    evaluate_over_affine_subspace::<Gf8>(0x1111_2222_3333_4444);
    evaluate_over_affine_subspace::<Gf16>(0x4444_3333_2222_1111);
}

/// Truncation composes with the shift: a degree-`active` polynomial padded
/// into a larger coset domain evaluates identically, without paying for the
/// padding.
#[test]
fn truncated_forward_composes_with_a_shift() {
    let mut rng = Rng(0xcafe_f00d_1234_5678);
    for dimension in 1..=6usize {
        let size = 1 << dimension;
        let (directions, shift) = random_subspace::<Gf16>(&mut rng, dimension);
        let plan = ShiftedPlan::<Gf16>::from_elements(size, &directions, shift).unwrap();
        let row_len = 2;

        for active in 1..=size {
            let mut monomial = rng.elems::<Gf16>(size);
            monomial[active..].fill(<Gf16 as Field>::Elem::ZERO);

            // Novel coefficients of a polynomial of degree < active occupy
            // exactly the first `active` slots: the novel basis is degree
            // triangular.
            let mut novel = monomial.clone();
            monomial_to_novel(&mut novel, plan.plan()).unwrap();
            assert!(
                novel[active..].iter().all(|value| value.is_zero()),
                "novel tail not zero for active {active}"
            );

            let mut rows = vec![0u8; size * row_len];
            for (index, &value) in novel.iter().enumerate() {
                <Gf16 as Field>::write(&mut rows[index * row_len..][..row_len], value);
            }
            plan.forward_bytes_trunc_range(&mut rows, row_len, active, 0..size)
                .unwrap();

            for index in 0..size {
                let value = <Gf16 as Field>::read(&rows[index * row_len..][..row_len]);
                assert_eq!(
                    value,
                    horner(&monomial, plan.point_element(index)),
                    "dimension {dimension} active {active} point {index}"
                );
            }
        }
    }
}

/// Interpolation direction: coset evaluations back to monomial coefficients.
#[test]
fn coset_interpolation_recovers_monomial_coefficients() {
    let mut rng = Rng(0x0bad_c0de_dead_10cc);
    for dimension in 1..=7usize {
        let size = 1 << dimension;
        let (directions, shift) = random_subspace::<Gf16>(&mut rng, dimension);
        let plan = ShiftedPlan::<Gf16>::from_elements(size, &directions, shift).unwrap();

        let monomial = rng.elems::<Gf16>(size);
        let mut values: Vec<_> = (0..size)
            .map(|index| horner(&monomial, plan.point_element(index)))
            .collect();

        plan.inverse(&mut values).unwrap();
        novel_to_monomial(&mut values, plan.plan()).unwrap();
        assert_eq!(values, monomial, "dimension {dimension}");
    }
}

/// The Cantor basis is a drop-in domain basis: same contracts, different
/// point order and cheaper tables.
#[test]
fn cantor_domain_basis_behaves_like_any_other() {
    let cantor = CantorBasis::<Gf16>::build().unwrap();
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    for dimension in 1..=8usize {
        let size = 1 << dimension;
        let plan = TransformPlan::<Gf16>::with_basis(size, &cantor.prefix(dimension)).unwrap();

        let monomial = rng.elems::<Gf16>(size);
        let mut values = monomial.clone();
        monomial_to_novel(&mut values, &plan).unwrap();
        plan.forward(&mut values).unwrap();
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(value, horner(&monomial, plan.point_element(index)));
        }
    }
}

/// Change of basis: bit ↔ Cantor coordinate maps invert each other, over
/// every element of a small field and a sample of a larger one.
#[test]
fn change_of_basis_matrices_invert() {
    let cantor8 = CantorBasis::<Gf8>::build().unwrap();
    let map = CoordinateMap::<Gf8>::of(&cantor8).unwrap();
    for pattern in 0..256u64 {
        let element = <Gf8 as Field>::read(&pattern.to_le_bytes()[..1]);
        assert_eq!(map.to_element(map.to_coordinates(element)), element);
        let coordinates = pattern;
        assert_eq!(map.to_coordinates(map.to_element(coordinates)), coordinates);
    }

    let cantor16 = CantorBasis::<Gf16>::build().unwrap();
    let map = CoordinateMap::<Gf16>::of(&cantor16).unwrap();
    let mut rng = Rng(0xa5a5_5a5a_a5a5_5a5a);
    for _ in 0..2048 {
        let element = rng.elem::<Gf16>();
        assert_eq!(map.to_element(map.to_coordinates(element)), element);
    }

    // Composing bit → Cantor coordinates with Cantor → element reproduces
    // the direct GF(2) combination of Cantor basis elements.
    for _ in 0..256 {
        let element = rng.elem::<Gf16>();
        let coordinates = map.to_coordinates(element);
        let mut rebuilt = <Gf16 as Field>::Elem::ZERO;
        for bit in 0..16 {
            if coordinates & (1 << bit) != 0 {
                rebuilt = rebuilt.add(OrderedBasis::<Gf16>::element(&cantor16, bit));
            }
        }
        assert_eq!(rebuilt, element);
    }
}
