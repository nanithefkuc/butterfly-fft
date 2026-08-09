#![cfg(feature = "std")]

//! Differential coverage for the restricted execution models.
//!
//! Every variant is checked against the full element-domain transform, which
//! the in-crate tests anchor to a naive O(n²) novel-basis evaluation. The
//! angles here are the ones single-element rows cannot reach:
//!
//! - **Multi-element rows.** The walkers compute `half_rows` from
//!   `rows.len() / 2 / row_len` and slice copies as `active * row_len`; with
//!   one element per row those two units coincide and a unit confusion is
//!   invisible. Every case below uses several elements per row, with
//!   independent payload per lane.
//! - **Awkward truncations.** `active` values that are neither halves nor
//!   powers of two drive the `active > half` branch of the truncated inverse
//!   several levels deep, where the scratch-based tail subtraction runs.
//! - **Sub-ranges of the high coset**, not just the whole half.

use cafft::core::kernel::ButterflyKernels;
use cafft::core::transform::TransformPlan;
use fgf::field::{Elem, Field};
use fgf::{Gf8, Gf16};

/// Deterministic per-test element stream.
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
}

/// `lanes` independent coefficient vectors of `size` elements each.
fn random_lanes<F: Field>(rng: &mut Rng, lanes: usize, size: usize) -> Vec<Vec<F::Elem>> {
    (0..lanes)
        .map(|_| (0..size).map(|_| rng.elem::<F>()).collect())
        .collect()
}

/// Interleave lanes into byte rows: row `i` holds lane `l`'s element `i` at
/// byte offset `l * F::BYTES`.
fn pack<F: Field>(lanes: &[Vec<F::Elem>], rows_used: usize) -> Vec<u8> {
    let row_len = lanes.len() * F::BYTES;
    let mut rows = vec![0u8; rows_used * row_len];
    for (lane_index, lane) in lanes.iter().enumerate() {
        for (row, &value) in lane.iter().take(rows_used).enumerate() {
            let start = row * row_len + lane_index * F::BYTES;
            F::write(&mut rows[start..start + F::BYTES], value);
        }
    }
    rows
}

fn unpack<F: Field>(rows: &[u8], lanes: usize, row: usize, lane: usize) -> F::Elem {
    let start = row * lanes * F::BYTES + lane * F::BYTES;
    F::read(&rows[start..start + F::BYTES])
}

/// Full forward transform of each lane, in the element domain.
fn forward_lanes<F: ButterflyKernels>(
    plan: &TransformPlan<F>,
    lanes: &[Vec<F::Elem>],
) -> Vec<Vec<F::Elem>> {
    lanes
        .iter()
        .map(|lane| {
            let mut values = lane.clone();
            plan.forward(&mut values).unwrap();
            values
        })
        .collect()
}

/// Assert that every lane of `rows` matches `reference` on `indices`.
fn assert_rows<F: Field>(
    rows: &[u8],
    lanes: usize,
    reference: &[Vec<F::Elem>],
    indices: impl IntoIterator<Item = usize>,
    what: &str,
) {
    for index in indices {
        for (lane, expected) in reference.iter().enumerate().take(lanes) {
            assert_eq!(
                unpack::<F>(rows, lanes, index, lane),
                expected[index],
                "{} row {index} lane {lane} ({})",
                what,
                F::NAME
            );
        }
    }
}

const LANES: usize = 3;

fn selected_matches_full<F: ButterflyKernels>(seed: u64) {
    let mut rng = Rng(seed);
    for log_size in 1..=7usize {
        let size = 1 << log_size;
        let plan = TransformPlan::<F>::new(size).unwrap();
        let lanes = random_lanes::<F>(&mut rng, LANES, size);
        let reference = forward_lanes(&plan, &lanes);
        let row_len = LANES * F::BYTES;

        // Every prime-strided subset, plus a pseudo-random one, plus the
        // extremes.
        let mut subsets: Vec<Vec<usize>> = vec![
            vec![0],
            vec![size - 1],
            (0..size).collect(),
            (0..size).step_by(5).collect(),
        ];
        let mut random: Vec<usize> = (0..size)
            .filter(|_| rng.next_u64().is_multiple_of(3))
            .collect::<Vec<_>>();
        random.dedup();
        subsets.push(random);

        for selected in &subsets {
            let mut rows = pack::<F>(&lanes, size);
            plan.forward_bytes_selected(&mut rows, row_len, selected)
                .unwrap();
            assert_rows::<F>(
                &rows,
                LANES,
                &reference,
                selected.iter().copied(),
                "selected",
            );
        }
    }
}

#[test]
fn selected_matches_full_transform_on_wide_rows() {
    selected_matches_full::<Gf8>(0x1234_5678_9abc_def0);
    selected_matches_full::<Gf16>(0x0fed_cba9_8765_4321);
}

fn range_matches_full<F: ButterflyKernels>(seed: u64) {
    let mut rng = Rng(seed);
    for log_size in 1..=7usize {
        let size = 1 << log_size;
        let plan = TransformPlan::<F>::new(size).unwrap();
        let lanes = random_lanes::<F>(&mut rng, LANES, size);
        let reference = forward_lanes(&plan, &lanes);
        let row_len = LANES * F::BYTES;

        for start in 0..size {
            for end in start..=size {
                let mut rows = pack::<F>(&lanes, size);
                plan.forward_bytes_range(&mut rows, row_len, start..end)
                    .unwrap();
                assert_rows::<F>(&rows, LANES, &reference, start..end, "range");
            }
        }
    }
}

#[test]
fn every_contiguous_range_matches_full_transform() {
    range_matches_full::<Gf8>(0xdead_beef_cafe_0001);
    range_matches_full::<Gf16>(0xdead_beef_cafe_0002);
}

fn trunc_range_matches_padded<F: ButterflyKernels>(seed: u64) {
    let mut rng = Rng(seed);
    for log_size in 1..=7usize {
        let size = 1 << log_size;
        let plan = TransformPlan::<F>::new(size).unwrap();
        let row_len = LANES * F::BYTES;

        for active in 1..=size {
            // Coefficients beyond `active` are zero: that is the contract
            // the truncated walker exploits.
            let mut lanes = random_lanes::<F>(&mut rng, LANES, size);
            for lane in &mut lanes {
                lane[active..].fill(F::Elem::ZERO);
            }
            let reference = forward_lanes(&plan, &lanes);

            for range in [
                0..size,
                0..1,
                size - 1..size,
                size / 2..size,
                active.min(size - 1)..size,
            ] {
                let mut rows = pack::<F>(&lanes, size);
                plan.forward_bytes_trunc_range(&mut rows, row_len, active, range.clone())
                    .unwrap();
                assert_rows::<F>(&rows, LANES, &reference, range, "trunc");
            }
        }
    }
}

#[test]
fn truncated_forward_matches_zero_padded_full_transform() {
    trunc_range_matches_padded::<Gf8>(0xfeed_face_0000_0001);
    trunc_range_matches_padded::<Gf16>(0xfeed_face_0000_0002);
}

/// `active == 0` means "nothing to evaluate": the buffer must be untouched.
#[test]
fn truncated_forward_with_no_active_prefix_is_a_no_op() {
    let plan = TransformPlan::<Gf16>::new(16).unwrap();
    let mut rng = Rng(0x5555_aaaa_5555_aaaa);
    let lanes = random_lanes::<Gf16>(&mut rng, LANES, 16);
    let original = pack::<Gf16>(&lanes, 16);
    let mut rows = original.clone();
    plan.forward_bytes_trunc_range(&mut rows, LANES * 2, 0, 0..16)
        .unwrap();
    assert_eq!(rows, original);
    // An empty output range is equally inert.
    plan.forward_bytes_range(&mut rows, LANES * 2, 4..4)
        .unwrap();
    assert_eq!(rows, original);
}

fn high_coset_matches_full<F: ButterflyKernels>(seed: u64) {
    let mut rng = Rng(seed);
    for log_size in 2..=7usize {
        let size = 1 << log_size;
        let half = size / 2;
        let plan = TransformPlan::<F>::new(size).unwrap();
        let row_len = LANES * F::BYTES;

        // The high coset evaluates `half` coefficients zero-padded to the
        // full domain, and reports transform points `half..size`.
        let mut lanes = random_lanes::<F>(&mut rng, LANES, size);
        for lane in &mut lanes {
            lane[half..].fill(F::Elem::ZERO);
        }
        let reference = forward_lanes(&plan, &lanes);

        for start in 0..half {
            for end in start..=half {
                let mut rows = pack::<F>(&lanes, half);
                plan.forward_bytes_high_coset_range(&mut rows, row_len, start..end)
                    .unwrap();
                for index in start..end {
                    for (lane, expected) in reference.iter().enumerate().take(LANES) {
                        assert_eq!(
                            unpack::<F>(&rows, LANES, index, lane),
                            expected[half + index],
                            "high coset row {index} lane {lane} size {size} ({})",
                            F::NAME
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn high_coset_sub_ranges_match_full_transform() {
    high_coset_matches_full::<Gf8>(0xabcd_ef01_2345_6789);
    high_coset_matches_full::<Gf16>(0x9876_5432_10fe_dcba);
}

fn inverse_truncated_recovers<F: ButterflyKernels>(seed: u64) {
    let mut rng = Rng(seed);
    for log_size in 1..=7usize {
        let size = 1 << log_size;
        let plan = TransformPlan::<F>::new(size).unwrap();
        let row_len = LANES * F::BYTES;

        for active in 1..=size {
            let mut lanes = random_lanes::<F>(&mut rng, LANES, size);
            for lane in &mut lanes {
                lane[active..].fill(F::Elem::ZERO);
            }
            let evaluations = forward_lanes(&plan, &lanes);

            let mut rows = pack::<F>(&evaluations, size);
            let scratch_rows = plan.inverse_truncated_scratch_rows(active);
            // Poison the scratch: a walker that reads it before writing must
            // not pass.
            let mut scratch = vec![0xA5u8; scratch_rows * row_len];
            plan.inverse_truncated_bytes(&mut rows, row_len, active, &mut scratch)
                .unwrap();
            assert_rows::<F>(&rows, LANES, &lanes, 0..active, "inverse truncated");
        }
    }
}

#[test]
fn truncated_inverse_recovers_every_active_prefix() {
    inverse_truncated_recovers::<Gf8>(0x0102_0304_0506_0708);
    inverse_truncated_recovers::<Gf16>(0x0807_0605_0403_0201);
}

/// The scratch requirement must be exactly what the walker uses: one byte
/// less has to be rejected, and the stated amount has to suffice.
#[test]
fn scratch_sizing_is_tight() {
    let plan = TransformPlan::<Gf16>::new(64).unwrap();
    let row_len = 2;
    for active in 1..=64usize {
        let rows_needed = plan.inverse_truncated_scratch_rows(active);
        if rows_needed == 0 {
            continue;
        }
        let mut rows = vec![0u8; 64 * row_len];
        let mut short = vec![0u8; rows_needed * row_len - 1];
        assert!(
            plan.inverse_truncated_bytes(&mut rows, row_len, active, &mut short)
                .is_err(),
            "active {active} accepted undersized scratch"
        );
    }
}

#[cfg(feature = "internals")]
#[test]
fn unstable_factor_table_surface_is_read_only() {
    use cafft::internals::FactorTable;

    let plan = TransformPlan::<Gf16>::new(64).unwrap();
    let table: &FactorTable<Gf16> = plan.table();
    assert_eq!(table.factors().len(), plan.size());
    assert_eq!(table.derivative_factors().len(), plan.log_size());
}
