//! Raw AFFT control measurements across butterfly-fft and the pinned comparison panel.
//!
//! The timed closures contain only one transform call. Input restoration,
//! allocation, initialization, and semantic checks happen outside timing.
//!
//! Run one smoke case:
//!
//! `cargo bench --manifest-path benchmarks/afft/Cargo.toml --bench raw_afft -- p32_r64 --test`
//!
//! Limit the full matrix by setting `BUTTERFLY_FFT_BENCH_MAX_BYTES` to the
//! largest payload buffer permitted per case. The default is 256 MiB.

use std::env;
use std::hint::black_box;
use std::time::Duration;

use additive_fft_reed_solomon::Gf2p8_11d;
use additive_fft_reed_solomon::kernel::Kernel;
use additive_fft_reed_solomon::kernel::gfni_kernel::GfniKernel;
use additive_fft_reed_solomon::kernel::lut_kernel::LutKernel;
use additive_fft_reed_solomon::poly_11d_lut::CantorBasisLut11d;
use butterfly_fft::basis::cantor_basis;
use butterfly_fft::core::kernel::backend as butterfly_fft_backend;
use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft_bench::{LeopardBuffer, NanorsBuffer, leopard_backend, nanors_backend};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fgf::{Gf8, Gf16};

const GF16_POINT_COUNTS: &[usize] = &[32, 128, 512, 2_048, 8_192, 32_768];
const GF8_POINT_COUNTS: &[usize] = &[32, 64, 128, 256];
const ROW_LENGTHS: &[usize] = &[64, 1_024, 65_536];
const DEFAULT_MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Case {
    points: usize,
    row_len: usize,
    bytes: usize,
}

impl Case {
    fn id(self) -> String {
        format!("p{}_r{}", self.points, self.row_len)
    }
}

fn max_bytes() -> usize {
    env::var("BUTTERFLY_FFT_BENCH_MAX_BYTES")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("BUTTERFLY_FFT_BENCH_MAX_BYTES must be a byte count")
        })
        .unwrap_or(DEFAULT_MAX_BYTES)
}

fn cases(point_counts: &[usize]) -> Vec<Case> {
    let max_bytes = max_bytes();
    point_counts
        .iter()
        .flat_map(|&points| {
            ROW_LENGTHS.iter().filter_map(move |&row_len| {
                let bytes = points.checked_mul(row_len)?;
                (bytes <= max_bytes).then_some(Case {
                    points,
                    row_len,
                    bytes,
                })
            })
        })
        .collect()
}

fn input_bytes(len: usize) -> Vec<u8> {
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn configure_group(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bytes: usize,
) {
    group.throughput(Throughput::Bytes(bytes as u64));
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn raw_gf16(c: &mut Criterion) {
    let butterfly_fft_name = format!("butterfly-fft-{}", butterfly_fft_backend().name());
    let leopard_name = format!("leopard-{}", leopard_backend());
    let nanors_name = format!("nanors-{}", nanors_backend());

    for case in cases(GF16_POINT_COUNTS) {
        let butterfly_fft =
            TransformPlan::<Gf16>::shared(case.points).expect("valid butterfly-fft plan");
        let input = input_bytes(case.bytes);

        let mut butterfly_fft_forward = input.clone();
        butterfly_fft
            .forward_bytes(&mut butterfly_fft_forward, case.row_len)
            .expect("valid butterfly-fft row geometry");

        let mut butterfly_fft_roundtrip = butterfly_fft_forward.clone();
        butterfly_fft
            .inverse_bytes(&mut butterfly_fft_roundtrip, case.row_len)
            .expect("valid butterfly-fft row geometry");
        assert_eq!(
            butterfly_fft_roundtrip,
            input,
            "butterfly-fft round trip for {}",
            case.id()
        );

        let mut leopard_roundtrip = LeopardBuffer::new(input.clone(), case.points);
        leopard_roundtrip.forward();
        leopard_roundtrip.inverse();
        assert_eq!(
            leopard_roundtrip.as_bytes(),
            input,
            "Leopard round trip for {}",
            case.id()
        );

        let mut nanors_roundtrip = NanorsBuffer::new(input.clone(), case.points);
        nanors_roundtrip.forward();
        nanors_roundtrip.inverse();
        assert_eq!(
            nanors_roundtrip.as_bytes(),
            input,
            "nanors round trip for {}",
            case.id()
        );

        let mut leopard_zero = LeopardBuffer::new(vec![0; case.bytes], case.points);
        leopard_zero.derivative();
        assert!(
            leopard_zero.as_bytes().iter().all(|&byte| byte == 0),
            "Leopard zero derivative for {}",
            case.id()
        );

        let case_id = case.id();
        let mut forward = c.benchmark_group("gf16/full_forward");
        configure_group(&mut forward, case.bytes);
        forward.bench_with_input(BenchmarkId::new(&butterfly_fft_name, &case_id), &case, |b, case| {
            b.iter_batched(
                || input.clone(),
                |mut rows| {
                    butterfly_fft
                        .forward_bytes(black_box(&mut rows), case.row_len)
                        .expect("valid butterfly-fft row geometry");
                    black_box(rows);
                },
                BatchSize::LargeInput,
            );
        });
        forward.bench_with_input(
            BenchmarkId::new(&leopard_name, &case_id),
            &case,
            |b, _case| {
                b.iter_batched(
                    || LeopardBuffer::new(input.clone(), case.points),
                    |mut rows| {
                        black_box(&mut rows).forward();
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        forward.bench_with_input(
            BenchmarkId::new(&nanors_name, &case_id),
            &case,
            |b, _case| {
                b.iter_batched(
                    || NanorsBuffer::new(input.clone(), case.points),
                    |mut rows| {
                        black_box(&mut rows).forward();
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        forward.finish();

        let mut inverse = c.benchmark_group("gf16/full_inverse");
        configure_group(&mut inverse, case.bytes);
        inverse.bench_with_input(BenchmarkId::new(&butterfly_fft_name, &case_id), &case, |b, case| {
            b.iter_batched(
                || input.clone(),
                |mut rows| {
                    butterfly_fft
                        .inverse_bytes(black_box(&mut rows), case.row_len)
                        .expect("valid butterfly-fft row geometry");
                    black_box(rows);
                },
                BatchSize::LargeInput,
            );
        });
        inverse.bench_with_input(
            BenchmarkId::new(&leopard_name, &case_id),
            &case,
            |b, _case| {
                b.iter_batched(
                    || LeopardBuffer::new(input.clone(), case.points),
                    |mut rows| {
                        black_box(&mut rows).inverse();
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        inverse.bench_with_input(
            BenchmarkId::new(&nanors_name, &case_id),
            &case,
            |b, _case| {
                b.iter_batched(
                    || NanorsBuffer::new(input.clone(), case.points),
                    |mut rows| {
                        black_box(&mut rows).inverse();
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        inverse.finish();

        let mut derivative = c.benchmark_group("gf16/derivative");
        configure_group(&mut derivative, case.bytes);
        derivative.bench_with_input(BenchmarkId::new(&butterfly_fft_name, &case_id), &case, |b, case| {
            b.iter_batched(
                || (input.clone(), vec![0; case.bytes]),
                |(coefficients, mut output)| {
                    butterfly_fft
                        .derivative_bytes(
                            black_box(&coefficients),
                            case.row_len,
                            black_box(&mut output),
                        )
                        .expect("valid butterfly-fft row geometry");
                    black_box(output);
                },
                BatchSize::LargeInput,
            );
        });
        derivative.bench_with_input(
            BenchmarkId::new(&leopard_name, &case_id),
            &case,
            |b, _case| {
                b.iter_batched(
                    || LeopardBuffer::new(input.clone(), case.points),
                    |mut rows| {
                        black_box(&mut rows).derivative();
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        derivative.finish();
    }
}

fn additive_gfni_supported() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("gfni")
}

fn raw_gf8(c: &mut Criterion) {
    let butterfly_fft_name = format!("butterfly-fft-cantor-{}", butterfly_fft_backend().name());
    let butterfly_fft_basis = cantor_basis::<Gf8>().expect("GF8 Cantor basis");
    let additive_basis = CantorBasisLut11d;
    let has_additive_gfni = additive_gfni_supported();

    for case in cases(GF8_POINT_COUNTS) {
        let butterfly_fft =
            TransformPlan::<Gf8>::with_basis(case.points, butterfly_fft_basis.elements())
                .expect("valid butterfly-fft Cantor plan");
        let input = input_bytes(case.bytes);
        let additive_input: Vec<Gf2p8_11d> = input.iter().copied().map(Gf2p8_11d::from).collect();
        let log_points = case.points.trailing_zeros() as u8;
        let zero = Gf2p8_11d::from(0);

        let mut butterfly_fft_roundtrip = input.clone();
        butterfly_fft
            .forward_bytes(&mut butterfly_fft_roundtrip, case.row_len)
            .expect("valid butterfly-fft GF8 geometry");
        butterfly_fft
            .inverse_bytes(&mut butterfly_fft_roundtrip, case.row_len)
            .expect("valid butterfly-fft GF8 geometry");
        assert_eq!(
            butterfly_fft_roundtrip,
            input,
            "butterfly-fft GF8 round trip for {}",
            case.id()
        );

        let mut lut_roundtrip = additive_input.clone();
        LutKernel::<Gf2p8_11d>::fft_sharded(
            &additive_basis,
            &mut lut_roundtrip,
            case.row_len,
            log_points,
            zero,
        );
        LutKernel::<Gf2p8_11d>::ifft_sharded(
            &additive_basis,
            &mut lut_roundtrip,
            case.row_len,
            log_points,
            zero,
        );
        assert_eq!(
            lut_roundtrip,
            additive_input,
            "additive LUT round trip for {}",
            case.id()
        );

        if has_additive_gfni {
            let mut gfni_roundtrip = additive_input.clone();
            GfniKernel::<Gf2p8_11d>::fft_sharded(
                &additive_basis,
                &mut gfni_roundtrip,
                case.row_len,
                log_points,
                zero,
            );
            GfniKernel::<Gf2p8_11d>::ifft_sharded(
                &additive_basis,
                &mut gfni_roundtrip,
                case.row_len,
                log_points,
                zero,
            );
            assert_eq!(
                gfni_roundtrip,
                additive_input,
                "additive GFNI round trip for {}",
                case.id()
            );
        }

        let case_id = case.id();
        let mut forward = c.benchmark_group("gf8/full_forward");
        configure_group(&mut forward, case.bytes);
        forward.bench_with_input(BenchmarkId::new(&butterfly_fft_name, &case_id), &case, |b, case| {
            b.iter_batched(
                || input.clone(),
                |mut rows| {
                    butterfly_fft
                        .forward_bytes(black_box(&mut rows), case.row_len)
                        .expect("valid butterfly-fft GF8 geometry");
                    black_box(rows);
                },
                BatchSize::LargeInput,
            );
        });
        forward.bench_with_input(
            BenchmarkId::new("additive-lut", &case_id),
            &case,
            |b, case| {
                b.iter_batched(
                    || additive_input.clone(),
                    |mut rows| {
                        LutKernel::<Gf2p8_11d>::fft_sharded(
                            &additive_basis,
                            black_box(&mut rows),
                            case.row_len,
                            log_points,
                            zero,
                        );
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        if has_additive_gfni {
            forward.bench_with_input(
                BenchmarkId::new("additive-avx512-gfni", &case_id),
                &case,
                |b, case| {
                    b.iter_batched(
                        || additive_input.clone(),
                        |mut rows| {
                            GfniKernel::<Gf2p8_11d>::fft_sharded(
                                &additive_basis,
                                black_box(&mut rows),
                                case.row_len,
                                log_points,
                                zero,
                            );
                            black_box(rows);
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
        forward.finish();

        let mut inverse = c.benchmark_group("gf8/full_inverse");
        configure_group(&mut inverse, case.bytes);
        inverse.bench_with_input(BenchmarkId::new(&butterfly_fft_name, &case_id), &case, |b, case| {
            b.iter_batched(
                || input.clone(),
                |mut rows| {
                    butterfly_fft
                        .inverse_bytes(black_box(&mut rows), case.row_len)
                        .expect("valid butterfly-fft GF8 geometry");
                    black_box(rows);
                },
                BatchSize::LargeInput,
            );
        });
        inverse.bench_with_input(
            BenchmarkId::new("additive-lut", &case_id),
            &case,
            |b, case| {
                b.iter_batched(
                    || additive_input.clone(),
                    |mut rows| {
                        LutKernel::<Gf2p8_11d>::ifft_sharded(
                            &additive_basis,
                            black_box(&mut rows),
                            case.row_len,
                            log_points,
                            zero,
                        );
                        black_box(rows);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        if has_additive_gfni {
            inverse.bench_with_input(
                BenchmarkId::new("additive-avx512-gfni", &case_id),
                &case,
                |b, case| {
                    b.iter_batched(
                        || additive_input.clone(),
                        |mut rows| {
                            GfniKernel::<Gf2p8_11d>::ifft_sharded(
                                &additive_basis,
                                black_box(&mut rows),
                                case.row_len,
                                log_points,
                                zero,
                            );
                            black_box(rows);
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
        inverse.finish();
    }
}

criterion_group!(benches, raw_gf16, raw_gf8);
criterion_main!(benches);
