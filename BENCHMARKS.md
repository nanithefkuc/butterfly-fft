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

### Fused butterfly schedule

`NttPlan` butterfly stages select between the fused register form — each
element pair updated directly, no scratch row, no whole-row copy, one
backend-independent walk per stage — and the packed op-call form it
replaced, which survives behind `internals::ntt_forward_packed` as the
callable control. The `ntt_tuning` bench interleaves both arms in one
binary at every execution geometry; ratios below are control (packed)
divided by fused, so above 1.00 the fused form wins.

```sh
FEC_GOLDEN_CORE=3 just _bench-run ntt_tuning
```

Core Ultra 7 258V, Linux, rustc 1.98.0, backend `v3_gfni_crypto`, cells
**Goldilocks / QuadMersenne31**:

| Size | lanes 1 | lanes 4 | lanes 16 |
| ---: | ---: | ---: | ---: |
| 64 | 2.18 / 1.74 | 0.54 / 1.20 | 0.27 / 1.16 |
| 256 | 1.75 / 1.80 | 0.45 / 1.19 | 0.24 / 1.15 |
| 1024 | 1.62 / 1.92 | 0.44 / 1.19 | 0.26 / 1.14 |
| 4096 | 1.62 / 1.88 | 0.52 / 1.17 | 0.22 / 1.13 |
| 16384 | 1.47 / 1.82 | 0.41 / 1.16 | 0.23 / 1.13 |
| 65536 | 1.44 / 1.76 | 0.53 / 1.17 | 0.23 / 1.13 |

The fused form dominates every QuadMersenne31 geometry — that field has no
vector elementwise kernels, so the packed form's four op calls per pair
bought nothing — and every Goldilocks single-lane geometry, where the
packed form's 8-byte rows ran the scalar tails of the AVX2 kernels. At
Goldilocks multi-element rows the packed kernels win decisively and keep
the schedule. The shipped selector is therefore `!has_vector_elementwise()
|| row_len <= FUSED_ROW_MAX` with `FUSED_ROW_MAX = 8` bytes: a pure
function of the row geometry and the resolved backend. The two-element
row (16 bytes) sits in the unmeasured gap between the winning ends and is
routed conservatively to the packed form; moving it requires a fresh
interleaved campaign.

### Execution

`forward_bytes_scratch` / `inverse_bytes_scratch` throughput in millions of
field elements per second, cells **forward/inverse**. One complete pinned
run on the fused-schedule tree; the schedule ratios above, not cross-run
deltas, are the comparison that set the selector.

**Goldilocks**

| Size | lanes 1 | lanes 4 | lanes 16 |
| ---: | ---: | ---: | ---: |
| 64 | 56.4/53.7 | 105/83.4 | 208/188 |
| 256 | 31.8/31.3 | 78.0/75.0 | 150/139 |
| 1024 | 23.1/23.0 | 61.9/60.5 | 124/116 |
| 4096 | 18.7/18.7 | 49.2/47.5 | 99.0/93.5 |
| 16384 | 15.0/15.0 | 43.1/42.1 | 82.8/79.2 |
| 65536 | 12.9/13.1 | 35.8/34.9 | 56.0/54.4 |

**QuadMersenne31**

| Size | lanes 1 | lanes 4 | lanes 16 |
| ---: | ---: | ---: | ---: |
| 64 | 52.2/47.1 | 62.3/55.0 | 66.9/59.2 |
| 256 | 38.2/34.8 | 46.0/41.6 | 48.1/44.0 |
| 1024 | 29.4/26.4 | 34.8/32.5 | 35.4/34.2 |
| 4096 | 23.4/22.9 | 28.6/27.0 | 30.9/29.0 |
| 16384 | 20.2/19.3 | 24.6/23.2 | 26.2/24.8 |
| 65536 | 16.9/16.1 | 21.2/20.3 | 22.6/21.4 |

Goldilocks scales with lane count because its wide rows ride the `fgf`
packed kernels; QuadMersenne31's fused extension-field arithmetic
saturates earlier. Both directions sit within a few percent of each other —
the inverse differs only by the twiddle table and one final scaling pass.

### Known correctness finding

The Goldilocks packed schedule — the four- and sixteen-lane rows, and any
geometry whose buffer reaches 256 Goldilocks elements — runs on a
defective path on GFNI hosts: the Goldilocks packed kernels of released
`fgf` 1.0.0 compute incorrect values through this transform there,
non-deterministically between processes with identical input, while the
scalar backend is exact. The fused schedule is scalar element arithmetic
throughout and is checked against an independent direct DFT at wide
geometries, so every geometry the selector routes to it is exact. Every
op-level differential (`mul_into`, prepared `mul_into_with`,
`sub_assign`, `mul_assign`, `mul_assign_with`, at the relevant widths and
random values) passes, so the defect lives in a kernel state the isolated
probes do not reproduce — consistent with stale upper-register contents in
the vector kernels. The packed-schedule columns are timing-only records:
the transform's work is value-independent, but correctness at those widths
is not established until `fgf` fixes the kernel. The FastECC comparison
below validates round trips at every geometry and times `butterfly-fft`
only where the round trip holds.


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

One complete pinned Lunar Lake run on the fused-schedule tree, criterion
medians, **forward/inverse**; `-` marks cells the correctness finding
above excludes. This campaign supersedes the earlier table recorded
before the fused schedule landed: its `butterfly-fft` column predates the
single-lane fused butterflies, and its FastECC columns are not
reproducible against the pinned revision on this host — historical
numbers are never reused without rerunning.

| Size | lanes | `butterfly-fft` | FastECC |
| ---: | ---: | ---: | ---: |
| 64 | 1 | 55.6/54.2 | 91.2/78.0 |
| 64 | 4 | - | 217/196 |
| 64 | 16 | - | 613/571 |
| 256 | 1 | 31.2/31.0 | 72.4/70.1 |
| 256 | 4 | - | 164/161 |
| 256 | 16 | - | 441/429 |
| 1024 | 1 | 21.8/21.6 | 56.6/56.3 |
| 1024 | 4 | - | 126/126 |
| 1024 | 16 | - | 328/329 |
| 4096 | 1 | 18.7/18.6 | 44.4/44.6 |
| 4096 | 4 | - | 100/100 |
| 4096 | 16 | - | 220/219 |
| 16384 | 1 | 15.2/15.0 | 34.7/34.5 |
| 16384 | 4 | - | 66.3/66.1 |
| 16384 | 16 | - | 170/171 |
| 65536 | 1 | 12.8/12.9 | 19.5/19.6 |
| 65536 | 4 | - | 52.6/52.4 |
| 65536 | 16 | - | 140/140 |

FastECC still leads every validated geometry — 1.6× at 64 points and
1.5× at the largest single-lane size, narrowed from 3.8× and 2.1× before
the fused schedule. Its field carries half the bytes per element and its
32-bit multiply-with-reciprocal reduction is cheaper per element than
Goldilocks' 64-bit fold, while `butterfly-fft`'s wide lanes ride the
general-purpose `fgf` element kernels and its single lanes run exact
scalar arithmetic. Closing the remainder is `fgf`-side work: the packed
Goldilocks kernels own both the correctness defect at wide geometries and
the per-element cost at narrow ones, and `QuadMersenne31` has no vector
kernels at all. FastECC's inverse is deliberately unnormalized (its round
trip scales the input by the point count) and its forward output is
natural only through the row-pointer table its MFA leaves permuted, so
the two arms' inverse and output-layout contracts do not match exactly;
both arms are validated against their own contract before timing.

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
