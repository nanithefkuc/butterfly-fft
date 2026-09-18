//! The `internals` surface: factor-table inspection and the tuning-only
//! derivative schedules, checked against the public derivative as oracle.

use butterfly_fft::internals::{self, FactorTable, FourStepPlan, FourStepScratch};
use butterfly_fft::ntt::{NttPlan, NttScratch};
use butterfly_fft::transform::TransformPlan;
use fgf::{Gf8B, Gf16, QuadMersenne31};

use butterfly_fft::error::TransformError;

type Schedule<F> = fn(&mut [u8], usize, &TransformPlan<F>, &[u8]) -> Result<(), TransformError>;

/// Deterministic per-test element stream, as in the crate's other suites.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64().to_le_bytes()[1]).collect()
    }
}

#[test]
fn factor_table_surface_is_read_only() {
    let plan = TransformPlan::<Gf16>::new(64).unwrap();
    let table: &FactorTable<Gf16> = internals::plan_table(&plan);
    assert_eq!(table.factors().len(), plan.size());
    assert_eq!(table.derivative_factors().len(), plan.log_size());
}

/// Every tuning schedule must agree with the production derivative, whose
/// crossover selection picks among these very schedules.
#[test]
fn tuning_derivative_schedules_match_production() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let schedules: [Schedule<Gf16>; 3] = [
        internals::derivative_into_bytes_sweep::<Gf16>,
        internals::derivative_into_bytes_gather::<Gf16>,
        internals::derivative_into_bytes_overwrite::<Gf16>,
    ];
    for (size, row_len) in [(16usize, 2usize), (64, 8), (256, 3 * 2)] {
        let plan = TransformPlan::<Gf16>::new(size).unwrap();
        let coefficients = rng.bytes(size * row_len);
        let mut expected = vec![0u8; coefficients.len()];
        plan.derivative_into_bytes(&mut expected, row_len, &coefficients)
            .unwrap();

        for schedule in schedules {
            // A garbage preset catches missing initialization.
            let mut actual = vec![0xA5u8; coefficients.len()];
            schedule(&mut actual, row_len, &plan, &coefficients).unwrap();
            assert_eq!(
                actual, expected,
                "schedule diverged at size {size} row length {row_len}"
            );
        }
    }
}

/// The tuning entry points validate geometry exactly like the public form.
#[test]
fn tuning_schedules_reject_bad_geometry() {
    let plan = TransformPlan::<Gf8B>::new(16).unwrap();
    let coefficients = vec![0u8; 16];
    let mut derivative = vec![0u8; 15];
    let schedules: [Schedule<Gf8B>; 3] = [
        internals::derivative_into_bytes_sweep::<Gf8B>,
        internals::derivative_into_bytes_gather::<Gf8B>,
        internals::derivative_into_bytes_overwrite::<Gf8B>,
    ];
    for schedule in schedules {
        assert!(schedule(&mut derivative, 1, &plan, &coefficients).is_err());
    }
}

/// Every NTT tuning schedule must agree with the public forward transform,
/// whose selector picks between these very schedules.
#[test]
fn tuning_ntt_schedules_match_production() {
    let mut rng = Rng(0x4552_7f4a_9e37_79b9);
    for (size, lanes) in [(16usize, 1usize), (64, 4), (256, 3)] {
        let row_len = lanes * <QuadMersenne31 as fgf::field::Field>::BYTES;
        let plan = NttPlan::<QuadMersenne31>::new(size).unwrap();
        let mut scratch: NttScratch = plan.scratch(row_len).unwrap();
        let rows = rng.bytes(size * row_len);

        let mut expected = rows.clone();
        plan.forward_bytes_scratch(&mut expected, row_len, &mut scratch)
            .unwrap();
        for schedule in [
            internals::ntt_forward_fused::<QuadMersenne31>,
            internals::ntt_forward_packed::<QuadMersenne31>,
            internals::ntt_forward_batched::<QuadMersenne31>,
        ] {
            // A garbage preset catches missing initialization.
            let mut actual = vec![0xA5u8; rows.len()];
            actual.copy_from_slice(&rows);
            schedule(&mut actual, row_len, &plan, &mut scratch).unwrap();
            assert_eq!(
                actual, expected,
                "schedule diverged at size {size} lanes {lanes}"
            );
        }
        let prepared = FourStepPlan::<QuadMersenne31>::new(size).unwrap();
        let mut fourstep_scratch: FourStepScratch = prepared.scratch(row_len).unwrap();
        let mut actual = rows.clone();
        prepared
            .forward_bytes_scratch(&mut actual, row_len, &mut fourstep_scratch)
            .unwrap();
        assert_eq!(
            actual, expected,
            "fourstep disagreed at size {size} lanes {lanes}"
        );
    }
}

/// The NTT tuning entry points validate geometry exactly like the public
/// form.
#[test]
fn tuning_ntt_schedules_reject_bad_geometry() {
    let plan = NttPlan::<QuadMersenne31>::new(16).unwrap();
    let mut rows = vec![0u8; 15 * 8];
    let mut scratch: NttScratch = plan.scratch(8).unwrap();
    for schedule in [
        internals::ntt_forward_fused::<QuadMersenne31>,
        internals::ntt_forward_packed::<QuadMersenne31>,
        internals::ntt_forward_batched::<QuadMersenne31>,
    ] {
        assert!(schedule(&mut rows, 8, &plan, &mut scratch).is_err());
    }
    let prepared = FourStepPlan::<QuadMersenne31>::new(16).unwrap();
    let mut fourstep_scratch = prepared.scratch(8).unwrap();
    let pristine = rows.clone();
    assert!(
        prepared
            .forward_bytes_scratch(&mut rows, 8, &mut fourstep_scratch)
            .is_err()
    );
    assert_eq!(rows, pristine);
}
