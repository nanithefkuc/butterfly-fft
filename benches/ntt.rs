//! Multiplicative-transform throughput.
//!
//! Plan construction (root search, permutation, twiddle preparation) and
//! scratch allocation are timed separately from the steady-state
//! `forward_bytes`/`inverse_bytes`, which allocate and prepare nothing.
//!
//! Only Goldilocks and `QuadMersenne31` appear. `Gf8B` and `Gf16` have odd
//! multiplicative group order (`2^m - 1`), and base `Mersenne31` has
//! `p - 1 = 2·(2^30 - 1)`, so none of the three has a radix-two domain at
//! these sizes — there is nothing to measure, and a faked panel would be
//! worse than an absent one. `Mersenne31` convolution embeds into
//! `QuadMersenne31`, which is measured here.
// `as_chunks_mut::<F::BYTES>()` needs a const generic argument depending on
// `F`, which stable Rust does not accept.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::hint::black_box;

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
        let value = F::read(&state.to_le_bytes()[..F::BYTES]);
        F::write(slot, value.add(F::Elem::ZERO));
    }
}

fn construction<F: FieldKernels>(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group(format!("ntt/{}/construct", F::NAME));
    for size in SIZES {
        group.bench_function(format!("plan/{size}"), |bencher| {
            bencher.iter(|| NttPlan::<F>::new(black_box(size)).expect("plan"));
        });
    }
    let plan = NttPlan::<F>::new(SIZES[0]).expect("plan");
    for lanes in LANES {
        let row_len = lanes * F::BYTES;
        group.bench_function(format!("scratch/{lanes}"), |bencher| {
            bencher.iter(|| plan.scratch(black_box(row_len)).expect("scratch"));
        });
    }
    group.finish();
}

fn execution<F: FieldKernels>(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group(format!("ntt/{}/execute", F::NAME));
    for size in SIZES {
        let plan = NttPlan::<F>::new(size).expect("plan");
        for lanes in LANES {
            let row_len = lanes * F::BYTES;
            let mut rows = vec![0u8; size * row_len];
            fill::<F>(&mut rows, 0x1234_5678_9abc_def1);
            let mut scratch: NttScratch = plan.scratch(row_len).expect("scratch");

            group.throughput(Throughput::Elements((size * lanes) as u64));
            group.bench_function(format!("forward/{size}/{lanes}"), |bencher| {
                bencher.iter(|| {
                    plan.forward_bytes(black_box(&mut rows), row_len, &mut scratch)
                        .expect("forward");
                });
            });
            group.bench_function(format!("inverse/{size}/{lanes}"), |bencher| {
                bencher.iter(|| {
                    plan.inverse_bytes(black_box(&mut rows), row_len, &mut scratch)
                        .expect("inverse");
                });
            });
        }
    }
    group.finish();
}

fn benches(criterion: &mut Criterion) {
    construction::<Goldilocks>(criterion);
    construction::<QuadMersenne31>(criterion);
    execution::<Goldilocks>(criterion);
    execution::<QuadMersenne31>(criterion);
}

criterion_group!(ntt, benches);
criterion_main!(ntt);
