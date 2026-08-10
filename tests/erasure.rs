//! Acceptance coverage for `butterfly_fft::rs`, through the public API only.
//!
//! The gate for the RS helpers is standalone property coverage on randomized
//! erasure patterns: encode a block, drop an arbitrary tolerable subset of
//! the transmitted symbols, and require exact recovery of every lost data
//! symbol — by the locator path and, where it applies, by the targeted dense
//! path, with both agreeing byte for byte.
//!
//! The locator itself is checked against the product definition
//! `Λ(x) = ∏_{e erased} (x ⊕ e)` computed directly in the field, so no stage
//! is its own oracle.

#![cfg(feature = "rs")]

use butterfly_fft::core::kernel::xor_scaled_bytes_rows;
use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::rs::{
    ErasureLocator, LocatorScratch, RecoveryScratch, RsField, StripEncoder, SystematicLocators,
    generator_row, inverse_scratch_elements, invert_square_into, recover_rows,
};
use fgf::field::{Elem, Field};
use fgf::{Gf8, Gf16};

/// xorshift64*, so patterns are varied but the failures are reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound as u64).expect("bound fits usize")
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next().to_le_bytes()[5]).collect()
    }

    /// A uniformly chosen `count`-subset of `0..bound`.
    fn subset(&mut self, bound: usize, count: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..bound).collect();
        for index in 0..count {
            let pick = index + self.below(bound - index);
            pool.swap(index, pick);
        }
        let mut chosen = pool[..count].to_vec();
        chosen.sort_unstable();
        chosen
    }
}

/// `Λ` and `Λ'` straight from the definition, in the field.
fn naive_locator<F: RsField>(
    plan: &TransformPlan<F>,
    known: &[bool],
) -> (Vec<F::Elem>, Vec<F::Elem>) {
    let size = plan.size();
    let mut values = vec![F::Elem::ZERO; size];
    let mut derivatives = vec![F::Elem::ZERO; size];
    for point in 0..size {
        let mut product = F::Elem::ONE;
        for (erased, &is_known) in known.iter().enumerate() {
            if is_known || erased == point {
                continue;
            }
            product = product.mul(plan.point_element(point ^ erased));
        }
        if known[point] {
            values[point] = product;
        } else {
            derivatives[point] = product;
        }
    }
    (values, derivatives)
}

#[test]
fn locator_matches_the_product_definition_on_random_patterns() {
    let mut rng = Rng::new(0x5eed_0001);
    for log_size in 1..=6usize {
        let plan = TransformPlan::<Gf16>::new(1 << log_size).unwrap();
        let size = plan.size();
        let mut locator = ErasureLocator::<Gf16>::for_domain(size);
        let mut scratch = LocatorScratch::for_domain(size);
        for _ in 0..24 {
            let count = rng.below(size + 1);
            let erased = rng.subset(size, count);
            let known: Vec<bool> = (0..size).map(|point| !erased.contains(&point)).collect();
            locator.recompute(&plan, &known, &mut scratch).unwrap();
            let (values, derivatives) = naive_locator(&plan, &known);
            assert_eq!(locator.values(), &values[..], "size {size} {erased:?}");
            assert_eq!(
                locator.derivatives(),
                &derivatives[..],
                "size {size} {erased:?}"
            );
        }
    }
    let plan = TransformPlan::<Gf8>::new(8).unwrap();
    for pattern in 0u32..256 {
        let known: Vec<bool> = (0..8).map(|index| pattern & (1 << index) != 0).collect();
        let locator = ErasureLocator::new(&plan, &known).unwrap();
        let (values, derivatives) = naive_locator(&plan, &known);
        assert_eq!(locator.values(), &values[..], "pattern {pattern:#04x}");
        assert_eq!(
            locator.derivatives(),
            &derivatives[..],
            "pattern {pattern:#04x}"
        );
    }
}

/// Recover `lost_data` through the locator path.
fn recover_by_locator<F: RsField>(
    encoder: &StripEncoder<F>,
    data: &[u8],
    repairs: &[u8],
    lost_data: &[usize],
    lost_repairs: &[usize],
) -> Vec<u8> {
    let row_len = encoder.row_len();
    let data_points = encoder.data_points();
    let repair_points = encoder.repair_points();
    let plan = encoder.evaluation_plan();
    let size = plan.size();

    let mut received = vec![0u8; size * row_len];
    let mut known = vec![false; size];
    for point in 0..data_points + repair_points {
        let source = if point < data_points {
            if lost_data.contains(&point) {
                continue;
            }
            &data[point * row_len..(point + 1) * row_len]
        } else {
            let repair = point - data_points;
            if lost_repairs.contains(&repair) {
                continue;
            }
            &repairs[repair * row_len..(repair + 1) * row_len]
        };
        received[point * row_len..(point + 1) * row_len].copy_from_slice(source);
        known[point] = true;
    }

    let locator = ErasureLocator::new(plan, &known).unwrap();
    let mut scratch = RecoveryScratch::for_geometry(size, row_len);
    let mut recovered = vec![0u8; lost_data.len() * row_len];
    recover_rows(
        plan,
        &locator,
        &received,
        row_len,
        lost_data,
        &mut scratch,
        &mut recovered,
    )
    .unwrap();
    recovered
}

/// Recover `lost_data` through the targeted dense path, using the first
/// `lost_data.len()` repair symbols.
fn recover_by_solve<F: RsField>(
    encoder: &StripEncoder<F>,
    locators: &SystematicLocators<F>,
    data: &[u8],
    repairs: &[u8],
    lost_data: &[usize],
) -> Vec<u8> {
    let row_len = encoder.row_len();
    let data_points = encoder.data_points();
    let plan = encoder.evaluation_plan();
    let locator = locators.get(plan, data_points).unwrap();
    let missing = lost_data.len();

    let mut generator = vec![F::Elem::ZERO; missing * data_points];
    for row in 0..missing {
        generator_row(
            plan,
            &locator,
            data_points + row,
            &mut generator[row * data_points..(row + 1) * data_points],
        );
    }
    let mut system = vec![F::Elem::ZERO; missing * missing];
    for row in 0..missing {
        for (column, &point) in lost_data.iter().enumerate() {
            system[row * missing + column] = generator[row * data_points + point];
        }
    }
    let mut augmented = vec![F::Elem::ZERO; inverse_scratch_elements(missing)];
    let mut inverse = vec![F::Elem::ZERO; missing * missing];
    assert!(
        invert_square_into(&system, missing, &mut augmented, &mut inverse),
        "the systematic sub-system is invertible for every erasure pattern"
    );

    let mut residuals = repairs[..missing * row_len].to_vec();
    let mut coefficients = vec![F::Elem::ZERO; missing];
    for point in 0..data_points {
        if lost_data.contains(&point) {
            continue;
        }
        for row in 0..missing {
            coefficients[row] = generator[row * data_points + point];
        }
        xor_scaled_bytes_rows::<F>(
            &mut residuals,
            row_len,
            &coefficients,
            &data[point * row_len..(point + 1) * row_len],
        );
    }

    let mut recovered = vec![0u8; missing * row_len];
    for residual in 0..missing {
        for row in 0..missing {
            coefficients[row] = inverse[row * missing + residual];
        }
        xor_scaled_bytes_rows::<F>(
            &mut recovered,
            row_len,
            &coefficients,
            &residuals[residual * row_len..(residual + 1) * row_len],
        );
    }
    recovered
}

fn check_expected(recovered: &[u8], data: &[u8], row_len: usize, lost: &[usize]) {
    for (row, &point) in lost.iter().enumerate() {
        assert_eq!(
            &recovered[row * row_len..(row + 1) * row_len],
            &data[point * row_len..(point + 1) * row_len],
            "data point {point}"
        );
    }
}

/// Randomized end-to-end acceptance over one code geometry.
fn round_trips<F: RsField>(data_points: usize, repair_points: usize, row_len: usize, seed: u64)
where
    F::Elem: Send + Sync,
{
    let mut rng = Rng::new(seed);
    let encoder = StripEncoder::<F>::new(data_points, repair_points, row_len).unwrap();
    let locators = SystematicLocators::<F>::new();
    let data = rng.bytes(data_points * row_len);
    let mut repairs = vec![0u8; repair_points * row_len];
    let mut scratch = encoder.scratch();
    encoder.encode(&data, &mut repairs, &mut scratch).unwrap();

    for _ in 0..16 {
        // Any subset of the transmitted symbols, as long as at most `m` are
        // lost: that is exactly the code's erasure tolerance.
        let losses = 1 + rng.below(repair_points);
        let dropped = rng.subset(data_points + repair_points, losses);
        let lost_data: Vec<usize> = dropped
            .iter()
            .copied()
            .filter(|&point| point < data_points)
            .collect();
        let lost_repairs: Vec<usize> = dropped
            .iter()
            .filter(|&&point| point >= data_points)
            .map(|&point| point - data_points)
            .collect();
        if lost_data.is_empty() {
            continue;
        }

        let recovered = recover_by_locator(&encoder, &data, &repairs, &lost_data, &lost_repairs);
        check_expected(&recovered, &data, row_len, &lost_data);

        // The targeted path consumes the first `|lost_data|` repairs, so it
        // applies only when those survived.
        if lost_repairs.iter().all(|&repair| repair >= lost_data.len()) {
            let solved = recover_by_solve(&encoder, &locators, &data, &repairs, &lost_data);
            assert_eq!(solved, recovered, "the two decode paths disagree");
        }
    }
}

#[test]
fn random_erasure_patterns_recover_exactly() {
    // Fused geometries (k a power of two, m <= k), padded geometries, single
    // and multi-element rows, both fields.
    round_trips::<Gf16>(8, 8, 2, 0xa1);
    round_trips::<Gf16>(8, 4, 6, 0xa2);
    round_trips::<Gf16>(5, 3, 2, 0xa3);
    round_trips::<Gf16>(9, 7, 4, 0xa4);
    round_trips::<Gf16>(17, 15, 2, 0xa5);
    round_trips::<Gf16>(100, 28, 8, 0xa6);
    round_trips::<Gf8>(4, 4, 1, 0xb1);
    round_trips::<Gf8>(5, 3, 3, 0xb2);
    round_trips::<Gf8>(20, 12, 2, 0xb3);
}

#[test]
fn maximal_erasure_is_still_exact() {
    // Losing exactly `m` data symbols is the worst tolerable pattern; the
    // locator path must still be exact, including for all-zero data (a
    // predicate-style check would pass on garbage here).
    for (data_points, repair_points, row_len) in [(9usize, 7usize, 4usize), (8, 8, 2)] {
        let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
        for zeroed in [false, true] {
            let data = if zeroed {
                vec![0u8; data_points * row_len]
            } else {
                Rng::new(0xc0de).bytes(data_points * row_len)
            };
            let mut repairs = vec![0u8; repair_points * row_len];
            let mut scratch = encoder.scratch();
            encoder.encode(&data, &mut repairs, &mut scratch).unwrap();
            let lost: Vec<usize> = (0..repair_points.min(data_points)).collect();
            let recovered = recover_by_locator(&encoder, &data, &repairs, &lost, &[]);
            check_expected(&recovered, &data, row_len, &lost);
            if zeroed {
                assert!(recovered.iter().all(|&byte| byte == 0));
            }
        }
    }
}

#[test]
fn locator_recovery_survives_a_shifted_domain() {
    // The locator only sees point differences, so an affine-coset plan
    // decodes with the same tables. Build a codeword over a coset, erase
    // half of it, recover.
    let shift = fgf::gf16::Elem(0x4d21);
    let shifted = butterfly_fft::shifted::ShiftedPlan::<Gf16>::new(16, shift).unwrap();
    let plan = shifted.plan();
    let row_len = 4;
    let active = 8;

    let mut rows = Rng::new(0xd1ce).bytes(active * row_len);
    rows.resize(16 * row_len, 0);
    shifted.forward_bytes(&mut rows, row_len).unwrap();
    let original = rows.clone();

    let erased: Vec<usize> = (0..16).filter(|point| point % 2 == 1).collect();
    let known: Vec<bool> = (0..16).map(|point| !erased.contains(&point)).collect();
    for &point in &erased {
        rows[point * row_len..(point + 1) * row_len].fill(0xA5);
    }
    let locator = ErasureLocator::new(plan, &known).unwrap();
    let mut scratch = RecoveryScratch::new();
    let mut recovered = vec![0u8; erased.len() * row_len];
    recover_rows(
        plan,
        &locator,
        &rows,
        row_len,
        &erased,
        &mut scratch,
        &mut recovered,
    )
    .unwrap();
    for (row, &point) in erased.iter().enumerate() {
        assert_eq!(
            &recovered[row * row_len..(row + 1) * row_len],
            &original[point * row_len..(point + 1) * row_len],
            "coset point {point}"
        );
    }
}

#[test]
fn strip_encoder_repairs_are_valid_codeword_evaluations() {
    // Independent of the truncated walkers: the repair at point `p` must be
    // the full forward transform of the interpolated coefficients, which we
    // obtain here by re-interpolating the received codeword with a full
    // inverse over the whole evaluation domain.
    for (data_points, repair_points, row_len) in [(8usize, 8usize, 2usize), (9, 7, 2), (16, 16, 2)]
    {
        let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
        let plan = encoder.evaluation_plan();
        let size = plan.size();
        assert_eq!(size, data_points + repair_points);
        let data = Rng::new(0xfeed).bytes(data_points * row_len);
        let mut repairs = vec![0u8; repair_points * row_len];
        let mut scratch = encoder.scratch();
        encoder.encode(&data, &mut repairs, &mut scratch).unwrap();

        let mut codeword = data.clone();
        codeword.extend_from_slice(&repairs);
        plan.inverse_bytes(&mut codeword, row_len).unwrap();
        // A systematic codeword has only `data_points` nonzero novel-basis
        // coefficients; the rest must vanish exactly.
        for point in data_points..size {
            for lane in 0..row_len / <Gf16 as Field>::BYTES {
                let start = point * row_len + lane * <Gf16 as Field>::BYTES;
                assert_eq!(
                    <Gf16 as Field>::read(&codeword[start..start + <Gf16 as Field>::BYTES]),
                    <Gf16 as Field>::Elem::ZERO,
                    "coefficient {point} of {size}"
                );
            }
        }
    }
}
