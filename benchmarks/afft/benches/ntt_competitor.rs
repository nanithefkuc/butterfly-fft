//! Matched-geometry multiplicative-NTT comparison: `butterfly-fft`'s
//! Goldilocks plan against FastECC's MFA NTT over GF(0xFFF00001).
//!
//! Both arms transform the same point counts with the same independent
//! lane counts packed into each row/block. Throughput is field elements per
//! second, the matched metric: the arms carry different element widths
//! (`Goldilocks` packs 8-byte lanes, FastECC packs 4-byte lanes over its
//! default prime field). FastECC is compiled from its pinned AVX2
//! configuration without OpenMP, so both arms run one thread.
//!
//! Every arm is validated against its own round-trip contract before any
//! timing at every geometry: `butterfly-fft` inverts to its input exactly,
//! and FastECC's unnormalized inverse scales its input by the point count.
//! Where the `butterfly-fft` round trip fails on the active backend — on
//! GFNI hosts the Goldilocks packed kernels in released `fgf` compute
//! incorrect values through this transform for buffers of 256 elements and
//! up — its cells are skipped with a notice rather than timed, while the
//! FastECC arm (which does not use `fgf`) still covers every geometry. See
//! the "Known correctness finding" note in `BENCHMARKS.md`. Run one smoke
//! pass with `-- --test`.

use std::hint::black_box;
use std::time::Duration;

use butterfly_fft::ntt::{NttPlan, NttScratch};
use butterfly_fft_bench::FastEccNttBuffer;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fgf::Goldilocks;
use fgf::field::{Elem, Field};

const SIZES: [usize; 6] = [64, 256, 1024, 4096, 16384, 65536];
const LANES: [usize; 3] = [1, 4, 16];

const FASTECC_MODULUS: u64 = 0xFFF00001;

/// Fixed-seed LCG, reduced into the field by the canonicalizing read.
fn fill(bytes: &mut [u8], seed: u64) {
    let mut state = seed;
    for slot in bytes.chunks_exact_mut(<Goldilocks as Field>::BYTES) {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let value =
            <Goldilocks as Field>::decode(&state.to_le_bytes()[..<Goldilocks as Field>::BYTES]);
        <Goldilocks as Field>::encode(slot, value.add(<Goldilocks as Field>::Elem::ZERO));
    }
}

/// Validate both arms at one geometry before anything is timed. Returns
/// whether the `butterfly-fft` arm is exact at this geometry on the active
/// backend; the FastECC arm is asserted directly.
fn validate(size: usize, lanes: usize) -> bool {
    let row_len = lanes * <Goldilocks as Field>::BYTES;
    let plan = NttPlan::<Goldilocks>::new(size).expect("supported size");
    let mut scratch = plan.scratch(row_len).expect("scratch");
    let mut rows = vec![0u8; size * row_len];
    fill(&mut rows, 0x1234_5678);
    let original = rows.clone();
    plan.forward_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("geometry");
    plan.inverse_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("geometry");
    let butterfly_exact = rows == original;

    let mut fastecc = FastEccNttBuffer::new(size, lanes, 0x1234_5678);
    let reference = fastecc.data().to_vec();
    fastecc.forward();
    fastecc.inverse();
    for (&got, &input) in fastecc.data().iter().zip(&reference) {
        let expected = (u64::from(input) * size as u64) % FASTECC_MODULUS;
        assert_eq!(u64::from(got), expected, "FastECC round trip diverged");
    }
    butterfly_exact
}

fn comparison(criterion: &mut Criterion) {
    let bytes_per_elem = <Goldilocks as Field>::BYTES;
    for &size in SIZES.iter() {
        for &lanes in LANES.iter() {
            let butterfly_exact = validate(size, lanes);
            if !butterfly_exact {
                println!(
                    "note: butterfly-fft NTT round trip fails at p{size}_l{lanes} on this \
                     backend; timing the FastECC arm only (see BENCHMARKS.md)"
                );
            }

            let row_len = lanes * bytes_per_elem;
            let plan = NttPlan::<Goldilocks>::new(size).expect("supported size");
            let mut scratch = plan.scratch(row_len).expect("scratch");
            let mut rows = vec![0u8; size * row_len];
            fill(&mut rows, 0x243F_6A88);
            let mut fastecc = FastEccNttBuffer::new(size, lanes, 0x243F_6A88);

            let case = format!("p{size}_l{lanes}");
            for direction in ["forward", "inverse"] {
                let mut group = criterion.benchmark_group(format!("ntt_competitor/{direction}"));
                group.throughput(Throughput::Elements((size * lanes) as u64));
                group.sample_size(20);
                group.warm_up_time(Duration::from_secs(1));
                group.measurement_time(Duration::from_secs(3));

                if butterfly_exact {
                    group.bench_function(format!("butterfly-fft-goldilocks/{case}"), |bencher| {
                        bencher.iter(|| {
                            if direction == "forward" {
                                plan.forward_bytes_scratch(
                                    black_box(&mut rows),
                                    row_len,
                                    black_box(&mut scratch),
                                )
                            } else {
                                plan.inverse_bytes_scratch(
                                    black_box(&mut rows),
                                    row_len,
                                    black_box(&mut scratch),
                                )
                            }
                            .expect("validated geometry");
                        });
                    });
                }
                group.bench_function(format!("fastecc-gf(0xfff00001)/{case}"), |bencher| {
                    bencher.iter(|| {
                        if direction == "forward" {
                            fastecc.forward();
                        } else {
                            fastecc.inverse();
                        }
                        black_box(&fastecc);
                    });
                });
                group.finish();
            }
        }
    }
}

criterion_group!(benches, comparison);
criterion_main!(benches);
