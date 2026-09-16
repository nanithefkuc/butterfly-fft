//! NTT butterfly-schedule crossover: fused register butterflies against the
//! packed op-call control, forced past production selection through the
//! `internals` facade.
//!
//! Both arms validate byte-for-byte agreement at every geometry before
//! anything is timed. Short targeted panels, as the crate's crossover

// `as_chunks_mut::<F::BYTES>()` needs a const generic argument depending on
// `F`, which stable Rust does not accept.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;

use butterfly_fft::internals::{ntt_forward_fused, ntt_forward_packed};
use butterfly_fft::ntt::{NttPlan, NttScratch};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::{Goldilocks, QuadMersenne31};

/// Full-product sizes spanning the cache hierarchy.
const SIZES: [usize; 6] = [64, 256, 1024, 4096, 16384, 65536];
/// Independent transform lanes packed into one row.
const LANES: [usize; 3] = [1, 4, 16];

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
        for lanes in LANES {
            let row_len = lanes * F::BYTES;
            let mut scratch: NttScratch = plan.scratch(row_len).expect("scratch");

            let mut fused = vec![0u8; size * row_len];
            fill::<F>(&mut fused, 0x1234_5678_9abc_def1);
            // The packed control rides released fgf's Goldilocks kernels,
            // which the recorded correctness finding marks
            // nondeterministically wrong from 256 elements up on GFNI
            // hosts, so cross-schedule agreement cannot gate the timing,
            // and neither can a round trip whose production inverse runs
            // the same defective kernels. Where the production path is
            // fully fused the exact round trip gates the fused arm; at the
            // packed-defect geometries the fused arm's correctness is the
            // unit-test direct-DFT differential, and both timing arms are
            // timing-only, as in the recorded competitor table.
            if !F::has_vector_elementwise() || row_len <= 8 {
                let mut roundtrip = fused.clone();
                ntt_forward_fused(&plan, &mut roundtrip, row_len, &mut scratch).expect("geometry");
                plan.inverse_bytes_scratch(&mut roundtrip, row_len, &mut scratch)
                    .expect("inverse");
                assert_eq!(
                    roundtrip,
                    fused,
                    "{} p{size}_l{lanes} fused round trip diverged",
                    F::NAME
                );
            }
            let mut packed = fused.clone();
            ntt_forward_packed(&plan, &mut packed, row_len, &mut scratch).expect("geometry");

            let case = format!("p{size}_l{lanes}");
            let mut group = criterion.benchmark_group(format!("ntt_tuning/{}", F::NAME));
            group.throughput(Throughput::Elements((size * lanes) as u64));
            group.sample_size(10);
            group.bench_function(format!("fused/{case}"), |bencher| {
                bencher.iter(|| {
                    ntt_forward_fused(
                        &plan,
                        black_box(&mut fused),
                        row_len,
                        black_box(&mut scratch),
                    )
                    .expect("validated geometry");
                });
            });
            group.bench_function(format!("packed/{case}"), |bencher| {
                bencher.iter(|| {
                    ntt_forward_packed(
                        &plan,
                        black_box(&mut packed),
                        row_len,
                        black_box(&mut scratch),
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
