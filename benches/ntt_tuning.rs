//! NTT butterfly-schedule crossover: fused register butterflies against the
//! packed op-call control, the batched region-pass candidate, and the
//! cache-blocked four-step candidate, forced past production selection
//! through the `internals` facade.
//!
//! Every arm validates byte-for-byte agreement at every geometry before
//! anything is timed. Short targeted panels, as the crate's crossover

// `as_chunks_mut::<F::BYTES>()` needs a const generic argument depending on
// `F`, which stable Rust does not accept.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;

use butterfly_fft::internals::{
    FourStepPlan, FourStepScratch, ntt_forward_batched, ntt_forward_fused, ntt_forward_packed,
};
use butterfly_fft::ntt::{NttPlan, NttScratch};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Goldilocks, QuadMersenne31};

/// Full-product sizes spanning the cache hierarchy.
const SIZES: [usize; 6] = [64, 256, 1024, 4096, 16384, 65536];
/// Independent transform lanes packed into one row.
const LANES: [usize; 5] = [1, 2, 4, 8, 16];

/// Fixed-seed LCG, reduced into the field by the canonicalizing read.
fn fill<F: Field>(bytes: &mut [u8], seed: u64) {
    let mut state = seed;
    for slot in bytes.chunks_exact_mut(F::BYTES) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let value = F::decode(&state.to_le_bytes()[..F::BYTES]);
        F::encode(slot, value.add(F::Elem::ZERO));
    }
}

fn schedules<F: FieldKernels>(criterion: &mut Criterion) {
    for size in SIZES {
        let plan = NttPlan::<F>::new(size).expect("plan");
        let fourstep_plan = FourStepPlan::<F>::new(size).expect("fourstep plan");
        for lanes in LANES {
            let row_len = lanes * F::BYTES;
            let mut scratch: NttScratch = plan.scratch(row_len).expect("scratch");
            let mut fourstep_scratch: FourStepScratch =
                fourstep_plan.scratch(row_len).expect("fourstep scratch");

            let mut fused = vec![0u8; size * row_len];
            fill::<F>(&mut fused, 0x1234_5678_9abc_def1);
            // Every geometry is gated on an exact round trip: the fixed
            // fgf kernels make the packed and batched arms and the
            // production inverse exact at every width, so cross-schedule
            // agreement and the round trip can gate the timing
            // everywhere. Both alternative arms must also match the
            // fused arm byte for byte.
            let mut packed_probe = fused.clone();
            ntt_forward_packed(&mut packed_probe, row_len, &plan, &mut scratch).expect("geometry");
            let mut fused_probe = fused.clone();
            ntt_forward_fused(&mut fused_probe, row_len, &plan, &mut scratch).expect("geometry");
            assert_eq!(
                packed_probe,
                fused_probe,
                "{} p{size}_l{lanes} packed diverged from fused",
                F::NAME
            );
            let mut batched_probe = fused.clone();
            ntt_forward_batched(&mut batched_probe, row_len, &plan, &mut scratch)
                .expect("geometry");
            assert_eq!(
                batched_probe,
                fused_probe,
                "{} p{size}_l{lanes} batched diverged from fused",
                F::NAME
            );
            let mut fourstep_probe = fused.clone();
            fourstep_plan
                .forward_bytes_scratch(&mut fourstep_probe, row_len, &mut fourstep_scratch)
                .expect("geometry");
            assert_eq!(
                fourstep_probe,
                fused_probe,
                "{} p{size}_l{lanes} fourstep diverged from fused",
                F::NAME
            );
            let mut roundtrip = fused_probe.clone();
            plan.inverse_bytes_scratch(&mut roundtrip, row_len, &mut scratch)
                .expect("inverse");
            assert_eq!(
                roundtrip,
                fused,
                "{} p{size}_l{lanes} inverse(forward(x)) diverged",
                F::NAME
            );
            let mut packed = fused.clone();
            ntt_forward_packed(&mut packed, row_len, &plan, &mut scratch).expect("geometry");
            let mut batched = fused.clone();
            ntt_forward_batched(&mut batched, row_len, &plan, &mut scratch).expect("geometry");

            let case = format!("p{size}_l{lanes}");
            let mut group = criterion.benchmark_group(format!("ntt_tuning/{}", F::NAME));
            group.throughput(Throughput::Elements((size * lanes) as u64));
            group.sample_size(10);
            group.bench_function(format!("fused/{case}"), |bencher| {
                bencher.iter(|| {
                    ntt_forward_fused(
                        black_box(&mut fused),
                        row_len,
                        &plan,
                        black_box(&mut scratch),
                    )
                    .expect("validated geometry");
                });
            });
            group.bench_function(format!("packed/{case}"), |bencher| {
                bencher.iter(|| {
                    ntt_forward_packed(
                        black_box(&mut packed),
                        row_len,
                        &plan,
                        black_box(&mut scratch),
                    )
                    .expect("validated geometry");
                });
            });
            group.bench_function(format!("batched/{case}"), |bencher| {
                bencher.iter(|| {
                    ntt_forward_batched(
                        black_box(&mut batched),
                        row_len,
                        &plan,
                        black_box(&mut scratch),
                    )
                    .expect("validated geometry");
                });
            });
            let mut fourstep = fused.clone();
            group.bench_function(format!("fourstep/{case}"), |bencher| {
                bencher.iter(|| {
                    fourstep_plan
                        .forward_bytes_scratch(
                            black_box(&mut fourstep),
                            row_len,
                            black_box(&mut fourstep_scratch),
                        )
                        .expect("validated geometry");
                });
            });
            group.finish();
        }
    }
}

fn benches(criterion: &mut Criterion) {
    schedules::<Goldilocks>(criterion);
    schedules::<QuadMersenne31>(criterion);
}

criterion_group!(ntt_tuning, benches);
criterion_main!(ntt_tuning);
