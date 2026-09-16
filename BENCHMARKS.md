# Benchmarks

Measurements of the public transform shapes on two x86 hosts. Both hosts ran
the same source tree with the `v3_gfni_crypto` backend selected. Every result
is the criterion median of one complete pinned run per host; re-measure before
quoting these numbers in a decision that a fresh campaign could settle.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.4, Arch Linux | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | Linux 7.2.3, CachyOS | 1.98.1 | 8, isolated |

Both hosts resolved `backend()` to `v3_gfni_crypto`, so the GF(2^8) and
GF(2^16) butterflies ran the GFNI kernels and the prime-field NTT lanes ran
`fgf`'s GFNI element kernels. Cells throughout are **Lunar Lake / Golden
Cove**.

Reproduce from the crate root:

```sh
FEC_GOLDEN_CORE=<cpu> just _bench-run ntt          # multiplicative NTT
taskset -c <cpu> env BUTTERFLY_FFT_BENCH_MAX_BYTES=67108864 \
    cargo bench --manifest-path benchmarks/afft/Cargo.toml --bench raw_afft
taskset -c <cpu> cargo bench --manifest-path benchmarks/afft/Cargo.toml \
    --bench ntt_competitor
```

The `afft` harness caps each case at 64 MiB, so both hosts ran the identical
case set. The timed closures contain only transform work: plans and scratch
are built outside the timed region, inputs are deterministic, and every arm
is validated against the library's own output (or an independent oracle)
before timing.

## Multiplicative NTT

`Goldilocks` and `QuadMersenne31` are the two fields with radix-two domains
at these sizes; the binary fields and base `Mersenne31` have none, and a
faked panel would be worse than an absent one.

### Plan construction

Median time per `NttPlan::new`, in µs. Construction is a one-time cost; the
execution numbers below are the steady-state contract.

| Field | 64 | 256 | 1024 | 4096 | 16384 | 65536 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Goldilocks | 1.25/1.18 | 2.93/2.78 | 9.23/8.84 | 34.1/33.2 | 132/128 | 529/739 |
| QuadMersenne31 | 1.88/1.90 | 4.47/4.61 | 13.8/14.7 | 50.0/54.1 | 194/211 | 769/834 |

`NttPlan::scratch` costs 12.9–13.9 ns on both hosts and fields.

### Butterfly schedules

`NttPlan` butterfly stages select among three schedules, a pure function of
the row geometry and the resolved backend. Fields without vector
elementwise kernels run the fused register form — each pair updated
directly, no scratch row, no whole-row copy. Fields with them run the
batched region form below `BATCHED_ROW_MAX` (128 bytes): per block and
bounded batch, one elementwise twiddle multiply into a region temporary,
one region copy, one region subtract, one region add — the same arithmetic
as the packed form with four dispatches per batch instead of per pair. At
and above the bound the packed op-call form keeps the rows, where its
broadcast multiply reads no twiddle stream and wins.

```sh
FEC_GOLDEN_CORE=3 just _bench-run ntt_tuning
```

Core Ultra 7 258V, Linux, rustc 1.98.0, backend `v3_gfni_crypto`, on the
`fgf` 1.1.0 kernels (the Goldilocks wrap defect fixed, QuadMersenne31
vectorized). Every arm is validated byte-for-byte against the fused arm and
an exact round trip before timing. Cells are **fused | packed | batched**
milliseconds; the adopted schedule per row width is the minimum of packed
and batched.

| Size | field | l1 | l2 | l4 | l8 | l16 |
| ---: | --- | --- | --- | --- | --- | --- |
| 64 | GLD | .001\|.003\|.001 | .001\|.002\|.001 | .005\|.003\|.002 | .012\|.007\|.007 | .019\|.010\|.011 |
| 1024 | GLD | .05\|.07\|.02 | .08\|.09\|.03 | .16\|.07\|.05 | .31\|.09\|.08 | .60\|.13\|.14 |
| 65536 | GLD | 5.5\|7.7\|1.9 | 9.2\|9.5\|3.0 | 17.3\|7.0\|4.8 | 34.2\|10.5\|8.0 | 66.1\|15.0\|14.7 |
| 64 | QM31 | .001\|.001\|.001 | .001\|.001\|.001 | .003\|.003\|.002 | .010\|.005\|.006 | .014\|.009\|.013 |
| 1024 | QM31 | .03\|.06\|.02 | .05\|.07\|.03 | .09\|.09\|.05 | .17\|.08\|.09 | .34\|.13\|.16 |
| 65536 | QM31 | 3.1\|6.1\|1.9 | 5.5\|7.1\|2.9 | 9.9\|9.6\|5.2 | 18.6\|8.7\|8.6 | 36.5\|14.4\|16.1 |

The batched form dominates the packed form below 128-byte rows at every
measured size for both fields — including single-lane rows, where region
passes over the vector kernels beat the fused form's per-element decode —
while at 128 bytes and above the packed broadcast multiply wins. The
128-byte boundary concedes at most the measured ten-percent margins on
either side of it (Goldilocks eight-lane, QuadMersenne31 sixteen-lane small
sizes); a per-field boundary would need a field fact fgf does not expose and
the margin does not justify.

#### Four-step rejection

A cache-blocked four-step decomposition (Bailey's six-step shape: three
in-place slot transposes around two half-size phases and a twiddle pass,
`internals::ntt_forward_fourstep`) measured 1.10–2.60× slower than the
adopted schedule at every geometry from 1024 points up, both fields, all
lane widths. The cycle-following transposes spend six cache-hostile
full-buffer passes, and the twiddle pass's elementwise multiply reads a
twiddle stream beside the data where the adopted schedules broadcast.
Correctness holds at every geometry, so the body stays behind `internals`
for cross-host rerun. Revisit requires a blocked or recursive transpose and
an in-place elementwise multiply; the arithmetic saving alone cannot
recover the measured movement cost.

### Execution

`forward_bytes_scratch` / `inverse_bytes_scratch` throughput in millions of
field elements per second, cells **forward/inverse**, on the adopted
three-way schedule over the `fgf` 1.1.0 kernels. One complete pinned run.

**Goldilocks**

| Size | lanes 1 | lanes 4 | lanes 16 |
| ---: | ---: | ---: | ---: |
| 64 | 59.7/56.9 | 104/96.7 | 195/184 |
| 256 | 53.0/51.7 | 95.4/91.7 | 156/150 |
| 1024 | 49.2/47.9 | 81.8/79.6 | 120/115 |
| 4096 | 45.1/44.3 | 69.1/67.6 | 101/97.4 |
| 16384 | 39.0/38.3 | 60.2/59.1 | 71.6/69.5 |
| 65536 | 34.8/34.2 | 54.6/53.9 | 66.1/64.4 |

**QuadMersenne31**

| Size | lanes 1 | lanes 4 | lanes 16 |
| ---: | ---: | ---: | ---: |
| 64 | 57.1/52.9 | 100/93.4 | 211/198 |
| 256 | 52.5/48.9 | 86.2/81.5 | 162/153 |
| 1024 | 47.9/45.1 | 74.9/71.6 | 120/113 |
| 4096 | 43.7/41.6 | 64.6/62.1 | 98.6/93.7 |
| 16384 | 38.8/37.1 | 57.3/55.2 | 90.0/86.2 |
| 65536 | 34.0/32.7 | 50.8/49.1 | 76.8/73.6 |

QuadMersenne31 rides `fgf` 1.1.0's AVX2 kernels at every row width the
selector routes to it; Goldilocks single-lane and two-lane rows run the
batched region form. Both directions sit within a few percent of each
other — the inverse differs only by the twiddle table and one final
scaling pass.

### FastECC comparison

FastECC (Bulat-Ziganshin, Apache-2.0) is the named competitor for this
family: an NTT-based O(N·log N) Reed-Solomon encoder whose MFA NTT runs
over GF(0xFFF00001). The comparison harness interleaves both arms in one
process over matched point counts and lane counts; throughput is field
elements per second, the metric that spans the arms' different element
widths (`Goldilocks` packs 8-byte lanes, FastECC 4-byte). FastECC is built
from its pinned AVX2 configuration without OpenMP, so both arms run one
thread. The Golden Cove host was unavailable for this campaign; its column
is left unmeasured.

```sh
taskset -c <cpu> cargo bench --manifest-path benchmarks/afft/Cargo.toml \
    --bench ntt_competitor
```

One complete pinned Lunar Lake run on the adopted three-way schedule over
the `fgf` 1.1.0 kernels, criterion medians, **forward/inverse**. Every
geometry round-trips exactly; no cell is excluded. Historical numbers are
never reused without rerunning.

| Size | lanes | `butterfly-fft` | FastECC |
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

`butterfly-fft` leads every single-lane geometry from 4096 points up —
1.8× at the largest size — and the four-lane rows above 16 K; FastECC's
lead survives at small sizes and dominates the sixteen-lane rows, where
its field packs half the bytes per element and its MFA's memory-friendly
access pattern multiplies four-byte lanes the general-purpose element
kernels cannot match at that width. FastECC's inverse is deliberately
unnormalized (its round trip scales the input by the point count) and its
forward output is natural only through the row-pointer table its MFA
leaves permuted, so the two arms' inverse and output-layout contracts do
not match exactly; both arms are validated against their own contract
before timing.

## Additive transforms: competitor matrix

The `benchmarks/afft` harness interleaves `butterfly-fft` with the comparison
panel in one process over identical payloads. `butterfly-fft` runs the Cantor
basis, matching Leopard's FF16 and the GF(2^8) panel: its unit derivative
factors are what make the head-to-head meaningful. Throughput is GiB/s over
`points × row_len` logical bytes.

### GF(2^16) forward and inverse

| Case | `butterfly-fft` | nanors | Leopard |
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

`butterfly-fft` leads every case on both hosts. The margin is widest at
small point counts and wide rows, where the fused one-pass butterflies avoid
the separate scaling pass the comparison panel pays.

### GF(2^16) derivative family

`derivative_into_bytes` is the exact out-of-place formal derivative;
`derivative_plus_identity_bytes` is the augmented in-place form a
root-evaluation workflow wants. Leopard's in-place sweep computes `c + D(c)`
natively, so the honest pairing for the augmented form is Leopard's own
primitive against `derivative_plus_identity_bytes`, and Leopard's
derivative-minus-copy against the exact form.

| Case | exact `butterfly-fft` | exact Leopard | augmented `butterfly-fft` | augmented Leopard |
| --- | ---: | ---: | ---: | ---: |
| p32_r64 | 14.7/14.2 | 9.61/10.6 | 10.5/11.9 | 14.2/18.3 |
| p128_r1024 | 15.9/21.0 | 9.19/13.0 | 16.8/17.3 | 26.5/23.8 |
| p2048_r1024 | 9.40/7.85 | 5.80/4.99 | 13.5/9.92 | 9.11/6.50 |
| p8192_r1024 | 4.94/5.27 | 3.69/3.98 | 6.09/7.17 | 7.17/4.71 |
| p32768_r1024 | 3.62/3.64 | 2.69/2.69 | 4.36/4.62 | 4.26/3.95 |

The exact derivative leads Leopard's exact form everywhere. Leopard's
augmented sweep wins the two smallest cases — it is a single fused pass with
no destination traffic — while `derivative_plus_identity_bytes` leads from
p2048 up on Lunar Lake and trades blows on Golden Cove.

### GF(2^8) forward and inverse

| Case | `butterfly-fft` fwd | additive-fft-rs LUT fwd | `butterfly-fft` inv | additive-fft-rs LUT inv |
| --- | ---: | ---: | ---: | ---: |
| p32_r64 | 11.0/10.9 | 1.18/1.36 | 11.0/11.5 | 1.19/1.37 |
| p128_r64 | 9.83/9.86 | 0.75/0.83 | 9.66/10.2 | 0.75/0.83 |
| p256_r64 | 8.99/9.14 | 0.63/0.70 | 9.03/9.61 | 0.62/0.70 |
| p256_r1024 | 10.8/10.1 | 0.68/0.74 | 11.1/9.26 | 0.68/0.74 |
| p128_r65536 | 6.90/5.28 | 0.77/0.80 | 6.72/5.22 | 0.78/0.80 |
| p256_r65536 | 5.63/4.27 | 0.66/0.67 | 5.61/4.85 | 0.65/0.69 |

The LUT kernel is an order of magnitude behind on this panel; its GFNI
kernel variant is the GF(2^16) `nanors` column above, which `butterfly-fft`
also leads.

## Derivative schedules

`TransformPlan::derivative_into_bytes` selects among three equivalent
schedules — source sweep, overwrite-first recursion, destination gather — by
row geometry and backend. The tuning entries behind the `internals` facade
drive each schedule directly; this section is the record the shipped
constants point at. Throughput is GiB/s.

| Field | Case | sweep | overwrite | gather |
| --- | --- | ---: | ---: | ---: |
| gf16 | p512_r64 | 10.4/10.3 | 7.76/8.46 | 2.14/1.96 |
| gf16 | p512_r1024 | 12.4/11.3 | 16.1/12.8 | 9.74/8.92 |
| gf16 | p8192_r1024 | 5.18/4.94 | 5.68/5.51 | 5.42/5.64 |
| gf16 | p32768_r1024 | 3.48/3.77 | 3.62/3.58 | 3.69/4.05 |
| gf8 | p128_r64 | 12.8/11.6 | 8.61/9.59 | 3.05/2.65 |
| gf8 | p128_r1024 | 14.2/18.0 | 16.5/21.2 | 17.6/16.9 |
| gf8 | p256_r1024 | 11.7/16.5 | 13.7/15.0 | 15.0/15.6 |
| gf8 | p256_r65536 | 5.30/6.07 | 5.50/6.32 | 7.03/10.4 |

The shipped constants in `transform.rs` match this record: the sweep wins
short rows, overwrite-first takes over by the 1 KiB row floor on GFNI
(`DERIVATIVE_OVERWRITE_ROW_MIN`), and the gather's register-blocked
`mul_add_gather` wins the 64 KiB row regime (`DERIVATIVE_GATHER_ROW_MIN`),
most decisively for GF(2^8). Moving either floor requires a fresh interleaved
before/after campaign against this table, not a reasoning adjustment.

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

Each cell is the criterion median of one pinned run on the named host; both
hosts ran the same tree, the same case cap, and the same pinned-core
protocol. The 64-byte-row cases are timer-resolution-sensitive and are
recorded for completeness — the 1 KiB-and-upward rows are the headline
measurements. Cross-host cells are not paired ratios; each column stands
alone.
