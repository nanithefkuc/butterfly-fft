//! Multiplicative-transform acceptance: convention, inversion, lane
//! independence, noncanonical ingress, and rejected geometry.
//!
//! Ground truth for the forward direction is an independent `O(n²)` direct
//! DFT written in scalar [`fgf::field::Elem`] arithmetic from the plan's own
//! [`NttPlan::root`]. A round trip alone cannot pin the ordering or the sign
//! convention — bit-reversed output and `ω^-1` both round-trip — so the
//! direct comparison is the load-bearing check here.
// `as_chunks::<F::BYTES>()` needs a const generic argument depending on `F`,
// which stable Rust does not accept.
#![allow(clippy::chunks_exact_to_as_chunks)]

use butterfly_fft::ntt::{NttError, NttPlan, NttScratch};
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Gf8B, Gf16, Goldilocks, Mersenne31, QuadMersenne31, goldilocks, quad_mersenne31};

/// Deterministic xorshift, as in this crate's other test suites.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A canonical element: raw bytes are reduced through `add(ZERO)`.
    fn elem<F: Field>(&mut self) -> F::Elem {
        F::decode(&self.next_u64().to_le_bytes()[..F::BYTES]).add(F::Elem::ZERO)
    }

    fn elems<F: Field>(&mut self, count: usize) -> Vec<F::Elem> {
        (0..count).map(|_| self.elem::<F>()).collect()
    }
}

fn pack<F: Field>(values: &[F::Elem]) -> Vec<u8> {
    let mut bytes = vec![0u8; values.len() * F::BYTES];
    for (slot, &value) in bytes.chunks_exact_mut(F::BYTES).zip(values) {
        F::encode(slot, value);
    }
    bytes
}

fn unpack<F: Field>(bytes: &[u8]) -> Vec<F::Elem> {
    bytes.chunks_exact(F::BYTES).map(F::decode).collect()
}

/// Independent direct transform: `Y[k] = Σ_j x[j] · root^(j·k)`.
fn direct_dft<F: Field>(input: &[F::Elem], root: F::Elem) -> Vec<F::Elem> {
    let size = input.len();
    (0..size)
        .map(|k| {
            let mut total = F::Elem::ZERO;
            for (j, &value) in input.iter().enumerate() {
                let exponent = (j * k) % size;
                total = total.add(value.mul(root.pow(exponent as u64)));
            }
            total
        })
        .collect()
}

/// Forward matches the direct DFT, and the inverse restores the input.
fn check_size<F: FieldKernels>(size: usize, seed: u64) {
    let plan = NttPlan::<F>::new(size).expect("supported size");
    assert_eq!(plan.size(), size);

    let mut rng = Rng(seed);
    let input = rng.elems::<F>(size);
    let expected = direct_dft::<F>(&input, plan.root());

    let mut rows = pack::<F>(&input);
    let mut scratch = plan.scratch(F::BYTES).expect("scratch");
    plan.forward_bytes_scratch(&mut rows, F::BYTES, &mut scratch)
        .expect("forward");
    assert_eq!(
        unpack::<F>(&rows),
        expected,
        "{} size {size}: forward output disagrees with the direct DFT",
        F::NAME
    );

    plan.inverse_bytes_scratch(&mut rows, F::BYTES, &mut scratch)
        .expect("inverse");
    assert_eq!(
        unpack::<F>(&rows),
        input,
        "{} size {size}: inverse did not restore the input",
        F::NAME
    );
}

/// Three lanes per row transform independently and agree with three
/// separate single-lane transforms.
fn check_lanes<F: FieldKernels>(size: usize, seed: u64) {
    check_lanes_width::<F>(size, seed, 3);
}

fn check_lanes_width<F: FieldKernels>(size: usize, seed: u64, lanes_count: usize) {
    let mut rng = Rng(seed);
    let lanes: Vec<Vec<F::Elem>> = (0..lanes_count).map(|_| rng.elems::<F>(size)).collect();

    let row_len = lanes_count * F::BYTES;
    let plan = NttPlan::<F>::new(size).expect("supported size");
    let mut rows = vec![0u8; size * row_len];
    for point in 0..size {
        for (lane, values) in lanes.iter().enumerate() {
            let at = point * row_len + lane * F::BYTES;
            F::encode(&mut rows[at..at + F::BYTES], values[point]);
        }
    }

    let mut scratch = plan.scratch(row_len).expect("scratch");
    plan.forward_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("forward");

    for (lane, values) in lanes.iter().enumerate() {
        let mut single = pack::<F>(values);
        let mut narrow = plan.scratch(F::BYTES).expect("scratch");
        plan.forward_bytes_scratch(&mut single, F::BYTES, &mut narrow)
            .expect("forward");
        let expected = unpack::<F>(&single);
        let got: Vec<F::Elem> = (0..size)
            .map(|point| {
                let at = point * row_len + lane * F::BYTES;
                F::decode(&rows[at..at + F::BYTES])
            })
            .collect();
        assert_eq!(got, expected, "{} lane {lane} size {size}", F::NAME);
    }

    plan.inverse_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("inverse");
    for point in 0..size {
        for (lane, values) in lanes.iter().enumerate() {
            let at = point * row_len + lane * F::BYTES;
            assert_eq!(F::decode(&rows[at..at + F::BYTES]), values[point]);
        }
    }
}

fn forward_of<F: FieldKernels>(plan: &NttPlan<F>, raw: &[u8]) -> Vec<F::Elem> {
    let mut rows = raw.to_vec();
    let mut scratch = plan.scratch(F::BYTES).expect("scratch");
    plan.forward_bytes_scratch(&mut rows, F::BYTES, &mut scratch)
        .expect("forward");
    unpack::<F>(&rows)
}
/// Acceptance sizes: the 256-point direct DFT is quadratic and stays out
/// of the miri run, which checks the same contracts at small sizes.
fn sizes() -> &'static [usize] {
    if cfg!(miri) {
        &[1, 2, 4, 8, 32]
    } else {
        &[1, 2, 4, 8, 32, 256]
    }
}

#[test]
fn goldilocks_forward_matches_direct_dft() {
    for &size in sizes() {
        check_size::<Goldilocks>(size, 0x1234_5678_9abc_def1);
    }
}

#[test]
fn quad_mersenne31_forward_matches_direct_dft() {
    for &size in sizes() {
        check_size::<QuadMersenne31>(size, 0x0f0f_1e2d_3c4b_5a69);
    }
}

#[test]
fn goldilocks_lanes_are_independent() {
    for &size in sizes() {
        check_lanes::<Goldilocks>(size, 0xdead_beef_cafe_0001);
    }
}

#[test]
fn quad_mersenne31_lanes_are_independent() {
    for &size in sizes() {
        check_lanes::<QuadMersenne31>(size, 0x0bad_c0de_1357_9bdf);
    }
}
#[test]
fn quad_mersenne31_wide_rows_match_single_lane_transforms() {
    for lanes in [4usize, 16] {
        check_lanes_width::<QuadMersenne31>(32, 0x0def_ec7a_0000_0000 + lanes as u64, lanes);
    }
}

#[test]
fn goldilocks_noncanonical_input_matches_canonical() {
    let plan = NttPlan::<Goldilocks>::new(8).expect("supported size");
    let modulus = goldilocks::MODULUS;

    // Raw `p` and raw `p + 1` are noncanonical encodings of 0 and 1.
    let raw: Vec<goldilocks::Elem> = vec![
        goldilocks::Elem::from_raw(modulus),
        goldilocks::Elem::from_raw(modulus + 1),
        goldilocks::Elem::from_raw(modulus + 5),
        goldilocks::Elem::from_raw(u64::MAX),
        goldilocks::Elem::from_raw(0),
        goldilocks::Elem::from_raw(3),
        goldilocks::Elem::from_raw(modulus),
        goldilocks::Elem::from_raw(modulus + 2),
    ];
    let canonical: Vec<goldilocks::Elem> = raw.iter().map(|&e| e.add(Elem::ZERO)).collect();
    assert_ne!(
        pack::<Goldilocks>(&raw),
        pack::<Goldilocks>(&canonical),
        "fixture must actually be noncanonical"
    );

    assert_eq!(
        forward_of(&plan, &pack::<Goldilocks>(&raw)),
        forward_of(&plan, &pack::<Goldilocks>(&canonical))
    );
}

#[test]
fn quad_mersenne31_noncanonical_input_matches_canonical() {
    let plan = NttPlan::<QuadMersenne31>::new(8).expect("supported size");
    let modulus = quad_mersenne31::MODULUS;

    let raw: Vec<quad_mersenne31::Elem> = vec![
        quad_mersenne31::Elem::from_raw(modulus, modulus),
        quad_mersenne31::Elem::from_raw(modulus, 4),
        quad_mersenne31::Elem::from_raw(7, modulus),
        quad_mersenne31::Elem::from_raw(u32::MAX, u32::MAX),
        quad_mersenne31::Elem::from_raw(0, 0),
        quad_mersenne31::Elem::from_raw(modulus, 1),
        quad_mersenne31::Elem::from_raw(2, 3),
        quad_mersenne31::Elem::from_raw(modulus, modulus),
    ];
    let canonical: Vec<quad_mersenne31::Elem> = raw.iter().map(|&e| e.add(Elem::ZERO)).collect();
    assert_ne!(
        pack::<QuadMersenne31>(&raw),
        pack::<QuadMersenne31>(&canonical),
        "fixture must actually be noncanonical"
    );

    assert_eq!(
        forward_of(&plan, &pack::<QuadMersenne31>(&raw)),
        forward_of(&plan, &pack::<QuadMersenne31>(&canonical))
    );
}

#[test]
fn invalid_sizes_are_rejected() {
    assert_eq!(
        NttPlan::<Goldilocks>::new(3).unwrap_err(),
        NttError::InvalidSize { size: 3 }
    );
    assert_eq!(
        NttPlan::<Goldilocks>::new(0).unwrap_err(),
        NttError::InvalidSize { size: 0 }
    );
    assert_eq!(
        NttPlan::<QuadMersenne31>::new(3).unwrap_err(),
        NttError::InvalidSize { size: 3 }
    );
}

#[test]
fn oversized_domain_is_rejected() {
    let size = 1usize << 21;
    assert_eq!(
        NttPlan::<Goldilocks>::new(size).unwrap_err(),
        NttError::UnsupportedSize {
            size,
            field_order: Goldilocks::ORDER,
        }
    );
}

/// `|F| - 1` is odd for the binary fields and `2·(2^30 - 1)` for base
/// Mersenne31, so none of them has a fourth root of unity; all three still
/// accept the identity transform.
#[test]
fn fields_without_a_domain_reject_size_four() {
    fn reject<F: FieldKernels>() {
        assert_eq!(
            NttPlan::<F>::new(4).unwrap_err(),
            NttError::UnsupportedSize {
                size: 4,
                field_order: F::ORDER,
            },
            "{} unexpectedly admitted size 4",
            F::NAME
        );
    }
    reject::<Gf8B>();
    reject::<Gf16>();
    reject::<Mersenne31>();
}

#[test]
fn size_one_is_the_identity_everywhere() {
    fn identity<F: FieldKernels>(seed: u64) {
        let plan = NttPlan::<F>::new(1).expect("size one is always legal");
        assert_eq!(plan.size(), 1);
        let mut rng = Rng(seed);
        let input = rng.elems::<F>(1);
        let mut rows = pack::<F>(&input);
        let mut scratch = plan.scratch(F::BYTES).expect("scratch");
        plan.forward_bytes_scratch(&mut rows, F::BYTES, &mut scratch)
            .expect("forward");
        assert_eq!(unpack::<F>(&rows), input, "{} forward", F::NAME);
        plan.inverse_bytes_scratch(&mut rows, F::BYTES, &mut scratch)
            .expect("inverse");
        assert_eq!(unpack::<F>(&rows), input, "{} inverse", F::NAME);
    }
    identity::<Gf8B>(0x9e37_79b9_7f4a_7c15);
    identity::<Gf16>(0x517c_c1b7_2722_0a95);
    identity::<Mersenne31>(0x2545_f491_4f6c_dd1d);
    identity::<Goldilocks>(0x1405_7b7e_f767_814f);
    identity::<QuadMersenne31>(0x0123_4567_89ab_cdef);
}

/// Mersenne31 is the boundary case: `p - 1 = 2·(2^30 - 1)`, so size two is
/// the only nontrivial radix-two domain it has.
#[test]
fn mersenne31_admits_only_size_two() {
    check_size::<Mersenne31>(2, 0x3141_5926_5358_9793);
    assert!(NttPlan::<Mersenne31>::new(8).is_err());
}

#[test]
fn validation_errors_leave_the_destination_untouched() {
    let plan = NttPlan::<Goldilocks>::new(4).expect("supported size");
    let element = <Goldilocks as Field>::BYTES;
    let row_len = 2 * element;
    let mut scratch = plan.scratch(row_len).expect("scratch");

    let mut rng = Rng(0x7f4a_7c15_9e37_79b9);
    let input = rng.elems::<Goldilocks>(8);
    let pristine = pack::<Goldilocks>(&input);

    // Wrong buffer length.
    let mut rows = pristine.clone();
    assert_eq!(
        plan.forward_bytes_scratch(&mut rows[..3 * row_len], row_len, &mut scratch)
            .unwrap_err(),
        NttError::BufferLength {
            expected: 4 * row_len,
            actual: 3 * row_len,
        }
    );
    assert_eq!(rows, pristine, "buffer-length rejection mutated the rows");

    // Row length that is not a whole number of elements.
    let ragged = row_len + 1;
    let mut rows = pristine.clone();
    assert_eq!(
        plan.forward_bytes_scratch(&mut rows, ragged, &mut scratch)
            .unwrap_err(),
        NttError::InvalidRowLength {
            row_len: ragged,
            element_bytes: element,
        }
    );
    assert_eq!(rows, pristine, "row-length rejection mutated the rows");

    // Scratch built for a shorter row.
    let mut narrow = plan.scratch(element).expect("scratch");
    let mut rows = pristine.clone();
    assert_eq!(
        plan.forward_bytes_scratch(&mut rows, row_len, &mut narrow)
            .unwrap_err(),
        NttError::ScratchTooSmall {
            required: row_len,
            available: element,
        }
    );
    assert_eq!(rows, pristine, "scratch rejection mutated the rows");

    assert_eq!(
        plan.inverse_bytes_scratch(&mut rows, row_len, &mut narrow)
            .unwrap_err(),
        NttError::ScratchTooSmall {
            required: row_len,
            available: element,
        }
    );
    assert_eq!(rows, pristine, "inverse scratch rejection mutated the rows");

    // Plan and scratch survive the rejections: a valid call still runs, and
    // still agrees with the direct DFT lane by lane.
    plan.forward_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("forward after rejections");
    for lane in 0..2 {
        let lane_input: Vec<_> = (0..4).map(|point| input[point * 2 + lane]).collect();
        let expected = direct_dft::<Goldilocks>(&lane_input, plan.root());
        let got: Vec<_> = (0..4)
            .map(|point| {
                let at = point * row_len + lane * element;
                <Goldilocks as Field>::decode(&rows[at..at + element])
            })
            .collect();
        assert_eq!(got, expected);
    }
    plan.inverse_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("inverse after rejections");
    assert_eq!(rows, pristine);
}

#[test]
fn scratch_rejects_a_ragged_row_length() {
    let plan = NttPlan::<Goldilocks>::new(4).expect("supported size");
    assert_eq!(
        plan.scratch(5).unwrap_err(),
        NttError::InvalidRowLength {
            row_len: 5,
            element_bytes: <Goldilocks as Field>::BYTES,
        }
    );
}

#[test]
fn zero_row_length_is_rejected() {
    let plan = NttPlan::<Goldilocks>::new(4).expect("supported size");
    assert_eq!(
        plan.scratch(0).unwrap_err(),
        NttError::InvalidRowLength {
            row_len: 0,
            element_bytes: <Goldilocks as Field>::BYTES,
        }
    );
    let mut scratch = plan.scratch(8).expect("scratch");
    let mut rows: Vec<u8> = Vec::new();
    assert_eq!(
        plan.forward_bytes_scratch(&mut rows, 0, &mut scratch)
            .unwrap_err(),
        NttError::InvalidRowLength {
            row_len: 0,
            element_bytes: <Goldilocks as Field>::BYTES,
        }
    );
    assert_eq!(
        plan.inverse_bytes_scratch(&mut rows, 0, &mut scratch)
            .unwrap_err(),
        NttError::InvalidRowLength {
            row_len: 0,
            element_bytes: <Goldilocks as Field>::BYTES,
        }
    );
    assert!(rows.is_empty());
}

/// A scratch carries the element width it was built for, so handing a
/// `QuadMersenne31` plan a `Mersenne31` scratch is rejected even though the
/// row is long enough in bytes.
#[test]
fn scratch_from_a_different_field_is_rejected() {
    let m31 = NttPlan::<Mersenne31>::new(2).expect("supported size");
    let mut foreign: NttScratch = m31
        .scratch(8 * <Mersenne31 as Field>::BYTES)
        .expect("scratch");

    let plan = NttPlan::<QuadMersenne31>::new(4).expect("supported size");
    let element = <QuadMersenne31 as Field>::BYTES;
    let mut rows = vec![0u8; 4 * element];
    assert_eq!(
        plan.forward_bytes_scratch(&mut rows, element, &mut foreign)
            .unwrap_err(),
        NttError::ScratchFieldMismatch {
            scratch_element_bytes: <Mersenne31 as Field>::BYTES,
            field_element_bytes: element,
        }
    );
}
