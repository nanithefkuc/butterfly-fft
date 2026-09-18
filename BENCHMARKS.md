# Benchmarks

Historical public API and competitor measurements. These records predate the
current release-preparation changes; they are not measurements of the current
tree. The exact source revision was not retained in this record.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.4, Arch Linux | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | Linux 7.2.3, CachyOS | 1.98.1 | 8, isolated |

The recorded process-wide backend selection was `v3_gfni_crypto` on both
hosts. Actual field-specific NTT backends were not recorded; this selection
does not establish which prime-field kernels executed. Unless a section says
otherwise, paired cells are **Lunar Lake / Golden Cove**.

Reproduce from the crate root:

```sh
FEC_GOLDEN_CORE=<cpu> just _bench-run ntt          # multiplicative NTT
taskset -c <cpu> env BUTTERFLY_FFT_BENCH_MAX_BYTES=67108864 \
    cargo bench --manifest-path benchmarks/afft/Cargo.toml --bench raw_afft
taskset -c <cpu> cargo bench --manifest-path benchmarks/afft/Cargo.toml \
    --bench ntt_competitor
```

- The `afft` harness caps each case at 64 MiB.
- Plans and scratch are built outside timed execution regions; construction
  is measured separately where named.
- Inputs are deterministic. The historical harness validated each arm against
  its own output contract or an independent oracle before timing.

## Multiplicative NTT

Fields: `Goldilocks` and `QuadMersenne31`. Other supported fields do not admit
the tabulated radix-two domain sizes.

### Plan construction

| Field | 64 (µs) | 256 (µs) | 1024 (µs) | 4096 (µs) | 16384 (µs) | 65536 (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Goldilocks | 1.25/1.18 | 2.93/2.78 | 9.23/8.84 | 34.1/33.2 | 132/128 | 529/739 |
| QuadMersenne31 | 1.88/1.90 | 4.47/4.61 | 13.8/14.7 | 50.0/54.1 | 194/211 | 769/834 |
- Method: `NttPlan::new`; column sizes are point counts.
- Aggregation: Criterion median construction time.


### Execution

- Methods: `forward_bytes_scratch` / `inverse_bytes_scratch`.
- `fgf` 1.1.0. Cells are **forward/inverse**; the per-table host was not
  identified in the retained execution record.
- Aggregation: Criterion medians from one pinned run.

**Goldilocks**

| Size | lanes 1 (Melem/s) | lanes 4 (Melem/s) | lanes 16 (Melem/s) |
| ---: | ---: | ---: | ---: |
| 64 | 59.7/56.9 | 104/96.7 | 195/184 |
| 256 | 53.0/51.7 | 95.4/91.7 | 156/150 |
| 1024 | 49.2/47.9 | 81.8/79.6 | 120/115 |
| 4096 | 45.1/44.3 | 69.1/67.6 | 101/97.4 |
| 16384 | 39.0/38.3 | 60.2/59.1 | 71.6/69.5 |
| 65536 | 34.8/34.2 | 54.6/53.9 | 66.1/64.4 |

**QuadMersenne31**

| Size | lanes 1 (Melem/s) | lanes 4 (Melem/s) | lanes 16 (Melem/s) |
| ---: | ---: | ---: | ---: |
| 64 | 57.1/52.9 | 100/93.4 | 211/198 |
| 256 | 52.5/48.9 | 86.2/81.5 | 162/153 |
| 1024 | 47.9/45.1 | 74.9/71.6 | 120/113 |
| 4096 | 43.7/41.6 | 64.6/62.1 | 98.6/93.7 |
| 16384 | 38.8/37.1 | 57.3/55.2 | 90.0/86.2 |
| 65536 | 34.0/32.7 | 50.8/49.1 | 76.8/73.6 |

### FastECC comparison

FastECC is an NTT-based Reed-Solomon encoder over GF(0xFFF00001).
The harness interleaves both arms in one process over matched point counts
and lane counts. `Goldilocks` packs eight-byte lanes; FastECC packs four-byte
lanes. FastECC uses its pinned AVX2 configuration without OpenMP; both arms
run one thread.

```sh
taskset -c <cpu> cargo bench --manifest-path benchmarks/afft/Cargo.toml \
    --bench ntt_competitor
```

- Host: Lunar Lake; Golden Cove was unavailable for this campaign.
- Aggregation: Criterion medians from one pinned run; `fgf` 1.1.0.
- Cells are **forward/inverse**.

| Size | lanes | `butterfly-fft` (Melem/s) | FastECC (Melem/s) |
| ---: | ---: | ---: | ---: |
| 64 | 1 | 59.2/56.9 | 90.6/76.9 |
| 64 | 4 | 112/106 | 214/194 |
| 64 | 16 | 205/184 | 607/565 |
| 256 | 1 | 54.1/52.7 | 71.4/68.9 |
| 256 | 4 | 97.4/92.8 | 163/160 |
| 256 | 16 | 153/141 | 447/439 |
| 1024 | 1 | 48.6/47.9 | 55.4/55.5 |
| 1024 | 4 | 83.3/79.6 | 127/127 |
| 1024 | 16 | 121/112 | 336/334 |
| 4096 | 1 | 45.2/44.5 | 44.6/44.7 |
| 4096 | 4 | 73.4/70.7 | 99.9/99.2 |
| 4096 | 16 | 103/97.4 | 212/213 |
| 16384 | 1 | 39.4/38.8 | 33.9/34.1 |
| 16384 | 4 | 62.3/60.3 | 64.5/64.7 |
| 16384 | 16 | 87.3/82.9 | 174/174 |
| 65536 | 1 | 34.8/34.4 | 19.8/19.8 |
| 65536 | 4 | 56.4/54.8 | 51.4/51.3 |
| 65536 | 16 | 73.7/70.4 | 144/117 |

- FastECC's inverse is unnormalized: its round trip scales the input by the
  point count. Its forward output is natural only through the row-pointer
  table left permuted by its MFA. The inverse and output-layout contracts
  therefore differ between arms; each arm is validated against its own
  contract before timing.

## Additive transforms: competitor matrix

The `benchmarks/afft` harness interleaves the comparison panel in one process
over identical payloads. `butterfly-fft` uses the Cantor basis, matching
Leopard's FF16 and the GF(2^8) panel. Logical payload is `points × row_len`.

### GF(2^16) forward and inverse

| Case | `butterfly-fft` (GiB/s) | nanors (GiB/s) | Leopard (GiB/s) |
| --- | ---: | ---: | ---: |
| p32_r64 fwd | 9.56/10.51 | 8.89/9.03 | 7.86/7.98 |
| p32_r1024 fwd | 19.5/17.0 | 18.0/15.9 | 17.2/14.1 |
| p128_r64 fwd | 7.89/8.14 | 6.65/6.63 | 5.75/5.63 |
| p128_r1024 fwd | 12.4/12.7 | 10.0/9.3 | 10.5/9.1 |
| p512_r64 fwd | 6.68/6.93 | 5.27/5.33 | 4.14/4.13 |
| p512_r1024 fwd | 8.68/8.22 | 6.80/7.07 | 7.55/6.20 |
| p2048_r1024 fwd | 8.17/6.38 | 6.14/5.50 | 6.23/3.74 |
| p8192_r1024 fwd | 4.20/3.95 | 3.51/3.26 | 3.80/3.19 |
| p32768_r1024 fwd | 3.15/2.20 | 2.47/2.49 | 2.41/2.44 |
| p32_r64 inv | 9.28/10.00 | 8.76/8.90 | 7.51/8.03 |
| p32_r1024 inv | 20.7/16.9 | 19.2/15.6 | 15.5/14.4 |
| p512_r1024 inv | 9.03/8.55 | 7.06/7.32 | 7.84/6.44 |
| p8192_r1024 inv | 4.15/4.00 | 3.46/3.31 | 3.74/3.19 |
| p32768_r1024 inv | 3.13/2.26 | 2.47/2.55 | 2.42/2.47 |

### GF(2^16) derivative family

- `derivative_into_bytes` is the exact out-of-place formal derivative.
- `derivative_plus_identity_bytes` is the augmented in-place form `c + D(c)`.
- Leopard's native in-place sweep is augmented; its exact derivative arm
  subtracts the input copy.

| Case | exact `butterfly-fft` (GiB/s) | exact Leopard (GiB/s) | augmented `butterfly-fft` (GiB/s) | augmented Leopard (GiB/s) |
| --- | ---: | ---: | ---: | ---: |
| p32_r64 | 14.7/14.2 | 9.61/10.6 | 10.5/11.9 | 14.2/18.3 |
| p128_r1024 | 15.9/21.0 | 9.19/13.0 | 16.8/17.3 | 26.5/23.8 |
| p2048_r1024 | 9.40/7.85 | 5.80/4.99 | 13.5/9.92 | 9.11/6.50 |
| p8192_r1024 | 4.94/5.27 | 3.69/3.98 | 6.09/7.17 | 7.17/4.71 |
| p32768_r1024 | 3.62/3.64 | 2.69/2.69 | 4.36/4.62 | 4.26/3.95 |

### GF(2^8) forward and inverse

| Case | `butterfly-fft` fwd (GiB/s) | additive-fft-rs LUT fwd (GiB/s) | `butterfly-fft` inv (GiB/s) | additive-fft-rs LUT inv (GiB/s) |
| --- | ---: | ---: | ---: | ---: |
| p32_r64 | 11.0/10.9 | 1.18/1.36 | 11.0/11.5 | 1.19/1.37 |
| p128_r64 | 9.83/9.86 | 0.75/0.83 | 9.66/10.2 | 0.75/0.83 |
| p256_r64 | 8.99/9.14 | 0.63/0.70 | 9.03/9.61 | 0.62/0.70 |
| p256_r1024 | 10.8/10.1 | 0.68/0.74 | 11.1/9.26 | 0.68/0.74 |
| p128_r65536 | 6.90/5.28 | 0.77/0.80 | 6.72/5.22 | 0.78/0.80 |
| p256_r65536 | 5.63/4.27 | 0.66/0.67 | 5.61/4.85 | 0.65/0.69 |

## Named competitors

| Library | License | Pin | Role |
| --- | --- | --- | --- |
| Leopard | BSD-2-Clause | `catid/leopard` @ `6e5725e` | GF(2^16) FF16 transform and derivative, C++, AVX2 adapter |
| nanors | MIT | `sleepybishop/nanors` @ `593cba1` | GF(2^16) additive FFT, C, AVX2+GFNI adapter |
| additive-fft-reed-solomon | Apache-2.0 | `vkomenda/additive-fft-reed-solomon` 0.1.3 @ `f1bcc91` | GF(2^8) LUT-kernel additive FFT, Rust |
| FastECC | Apache-2.0 | `Bulat-Ziganshin/FastECC` @ `b8ca7db` | GF(0xFFF00001) MFA NTT, C++, AVX2 build without OpenMP |

All four are dev-only, built inside the excluded `benchmarks/afft` package,
and every arm validates against the library's own output before timing.
FastECC is the named competitor for the multiplicative NTT family; the
additive comparisons and the NTT comparison run from the same harness.
The Golden Cove host was unavailable when the FastECC campaign ran, so the
NTT comparison table carries Lunar Lake only.

## Aggregation and caveats

- Each cell is the Criterion median of one pinned run on the named host.
- The two-host campaigns used the same source tree, case cap, and pinned-core
  protocol; cross-host cells are independent measurements, not paired ratios.
- Short-row cases are timer-resolution-sensitive.
- Run-to-run variance and confidence intervals were not retained in this
  record. Re-measure before using these historical values for a decision.
