> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

> [!WARNING]
> `butterfly-fft` makes no constant-time guarantee. Its field arithmetic comes
> from variable-time `fgf`; handling secret elements, coefficients, or buffers
> requires a separate audit.

# butterfly-fft — Additive FFT and NTT over Finite Fields

`butterfly-fft` provides reusable transform plans over finite fields. It
supports additive FFTs over binary fields and multiplicative number-theoretic
transforms over prime and prime-extension fields, for polynomial evaluation,
interpolation, and the transform layer underneath erasure coding.

| Main property | What it provides |
| --- | --- |
| Checked public geometry | Plans validate domains; execution checks buffer lengths and byte-row geometry before transforming. |
| Reusable preparation | Plans own precomputed tables; additive execution and NTT execution with reusable scratch allocate nothing in steady state. |
| Runtime SIMD | Dedicated additive byte-row butterflies for `Gf8B` and `Gf16` on x86 and `AArch64`, with portable fallbacks. |
| Restricted evaluation | Selected-output, range, truncated, and high-coset additive transforms. |
| Basis conversion | Bit and Cantor bases, coordinate maps, and monomial/novel coefficient conversion. |
| Portable builds | `no_std` with `alloc`, without architecture-specific compiler flags. |

`butterfly-fft` is a transform library, not a codec. It owns transform-buffer
layouts and butterfly kernels; [`fgf`](https://github.com/nanithefkuc/fgf) owns
field arithmetic and packed field encodings. Consumers own wire formats,
recovery protocols, and evaluation-point-to-wire-index mappings.

## Installation

The minimum supported Rust version is 1.93, edition 2024. Examples use the
same `fgf` release as the library so field types match:

```toml
[dependencies]
butterfly-fft = "=1.0.0"
fgf = "=1.1.0"
```

For portable `no_std` execution, plans and tables still require `alloc`:

```toml
[dependencies]
butterfly-fft = { version = "=1.0.0", default-features = false }
fgf = { version = "=1.1.0", default-features = false }
```

## Quick start

An additive plan consumes novel-basis coefficients. A polynomial supplied in
the monomial basis is converted before evaluation; interpolation and the
reverse conversion recover the original coefficients:

```rust
use butterfly_fft::{TransformPlan, basis::{monomial_to_novel, novel_to_monomial}};
use fgf::{Gf16, gf16};

let plan = TransformPlan::<Gf16>::new(4).unwrap();
// p(x) = 3 + 2x.
let coefficients = [3, 2, 0, 0].map(gf16::Elem::from_raw);
let mut values = coefficients;
monomial_to_novel(&mut values, &plan).unwrap();
plan.forward(&mut values).unwrap();
for (index, value) in values.iter().enumerate() {
    let expected = gf16::Elem::from_raw(3)
        .add(gf16::Elem::from_raw(2).mul(plan.point_element(index)));
    assert_eq!(*value, expected);
}
plan.inverse(&mut values).unwrap();
novel_to_monomial(&mut values, &plan).unwrap();
assert_eq!(values, coefficients);
```

Byte-row transforms apply the same operation independently to each element
column. Buffers hold `size` contiguous rows of `row_len` bytes, with packed
little-endian field elements inside each row:

```rust
use butterfly_fft::TransformPlan;
use fgf::Gf16;

let plan = TransformPlan::<Gf16>::new(16).unwrap();
let mut rows = [0x5au8; 16 * 64];
let coefficients = rows;

plan.forward_bytes(&mut rows, 64).unwrap();
plan.inverse_bytes(&mut rows, 64).unwrap();
assert_eq!(rows, coefficients);
```

## Supported fields and domains

Both transform families require power-of-two sizes and cap the domain at
`2^transform::MAX_LOG_SIZE` points. Additive domains are also bounded by the
binary field's dimension; NTT domains require a primitive root of the requested
order.

| Transform | Field markers | Domain | Acceleration |
| --- | --- | --- | --- |
| Additive FFT | `Gf8B`, `Gf16` | binary subspace or affine coset | dedicated x86 and `AArch64` byte-row butterflies |
| Additive FFT | `Gf8D`, `Gf32`, `Gf64`, `FanPaar8`, `FanPaar16`, `FanPaar32`, `FanPaar64` | binary subspace or affine coset | portable butterflies |
| NTT | `Goldilocks`, `QuadMersenne31` | every power-of-two size up to the plan cap | `fgf` field kernels |
| NTT | `Mersenne31` | sizes one and two | `fgf` field kernels |
| NTT | binary fields | identity only (size one) | portable |

## Additive transforms

`TransformPlan::new` uses the bit basis. `with_basis` accepts an independent
ordered basis, and `with_shift` translates its span into an affine coset.
With `std`, `TransformPlan::shared` caches unshifted bit-basis plans per field
and size process-wide. `transform::PlanCache` provides an owned, droppable
cache without requiring `std`.

| Operation | Contract |
| --- | --- |
| `forward` / `inverse` | in-place evaluation / interpolation over typed elements |
| `forward_bytes` / `inverse_bytes` | the same operations over independent packed element columns |
| `forward_bytes_selected` / `forward_bytes_range` | only the requested output rows are final |
| `forward_bytes_truncated_range` | evaluate a coefficient prefix with a zero-filled tail, restricted to an output range |
| `forward_bytes_high_coset_range` | evaluate half-domain coefficients at high-coset points using a half-domain buffer |
| `inverse_bytes_truncated_scratch` | recover an active coefficient prefix using caller-owned scratch; the represented polynomial must have a zero coefficient tail |
| `derivative_into` / `derivative_into_bytes` | write the formal derivative to a separate destination |
| `derivative_plus_identity_bytes` | replace coefficients with the sum of the polynomial and its derivative, not the derivative alone |

Restricted transforms define only their documented output rows; other rows
may hold intermediate values. The
[transform module](https://github.com/nanithefkuc/butterfly-fft/blob/main/src/transform.rs)
owns the per-operation input and scratch requirements.

The [basis module](https://github.com/nanithefkuc/butterfly-fft/blob/main/src/basis.rs)
provides `BitBasis`, `CantorBasis`, and `CoordinateMap` for domain coordinates.
`monomial_to_novel` and `novel_to_monomial` convert polynomial coefficients;
their `_scratch` forms borrow workspace instead of allocating it. Domain basis
selection and polynomial coefficient conversion are distinct operations.

## Number-theoretic transforms

`NttPlan` consumes and produces natural-order rows. Its forward transform uses
powers of `plan.root()`; the inverse uses the reciprocal root and normalizes
by the inverse domain size. Construction prepares the tables, while
`plan.scratch(row_len)` allocates reusable execution workspace:

```rust
use butterfly_fft::NttPlan;
use fgf::Goldilocks;

let plan = NttPlan::<Goldilocks>::new(4).unwrap();
let mut rows = [0u8; 32];
rows[..8].copy_from_slice(&1u64.to_le_bytes());
let original = rows;
let mut scratch = plan.scratch(8).unwrap();

plan.forward_bytes_scratch(&mut rows, 8, &mut scratch).unwrap();
for lane in rows.chunks_exact(8) {
    assert_eq!(lane, 1u64.to_le_bytes());
}
plan.inverse_bytes_scratch(&mut rows, 8, &mut scratch).unwrap();
assert_eq!(rows, original);
```

Prime-field input lanes are canonicalized before execution. Round trips
preserve field values; encoded bytes return unchanged when the input was
canonical. Consumers choose padding and truncation for convolution; an NTT
plan transforms exactly its declared domain. The
[NTT module](https://github.com/nanithefkuc/butterfly-fft/blob/main/src/ntt.rs)
defines the transform convention and workspace requirements.

Four-step NTT experiments use `internals::FourStepPlan` and
`internals::FourStepScratch`, with explicit construction and scratch preparation.
Stable `NttPlan` and `NttScratch` do not prepare or retain that experimental
state. These experiment types are not part of the stable API.

## Features

| Feature | Effect |
| --- | --- |
| default | enables `std` and runtime-dispatched `simd` |
| `std` | shared plan and basis caches, plus `fgf` runtime support |
| `simd` | architecture-specific butterfly and field kernels; implies `std` |
| `internals` | re-export-only facade of factor tables, tuning APIs, and experimental four-step plans |

Without default features, the crate uses `no_std` plus `alloc` and portable
scalar execution. There is no separate `alloc` feature: plans always allocate.
Nothing behind `internals` is a compatibility promise; those APIs may change
or disappear in any release.

## Platforms and backends

| Additive butterfly backend | Target |
| --- | --- |
| `v3_gfni_crypto` | `x86`/`x86_64` with AVX2, GFNI, and crypto capabilities |
| `v3` | `x86`/`x86_64` AVX2 shuffle kernels |
| `v2` | `x86`/`x86_64` SSSE3/SSE4.2 shuffle kernels |
| `neon_aes` | `AArch64` NEON with AES/PMULL capability; uses the same butterflies as `neon` |
| `neon` | `AArch64` NEON shuffle kernels |
| `scalar` | portable fallback, including WebAssembly |

`kernel::backend()` reports the process-wide additive butterfly selection;
`kernel::backend_for::<F>()` narrows it to the tiers implemented for a field.
Only `Gf8B` and `Gf16` have dedicated SIMD butterflies. NTTs use `fgf` field
kernels rather than this additive backend ladder.

[`simdispatch`](https://github.com/nanithefkuc/simdispatch) owns detection and
the downgrade-only `SIMD_BACKEND` override. It may request a weaker backend
at process startup; unsupported upgrades are ignored. Builds need no global
target-feature flags.

## Layout and safety

Additive coefficients use the novel basis, not the monomial basis. Evaluation
point `i` is the coset shift plus the XOR of the ordered basis elements at the
set bits of `i`. Under an unshifted bit-basis plan, point order is field
bit-pattern order. Byte rows use `fgf`'s little-endian element encodings.

The public API is safe but not constant-time. Additive execution and basis
conversion return `TransformError` for invalid buffer, workspace, selection,
range, active prefix, or byte-row geometry. NTT methods return `NttError`.
Rejected execution leaves destination and scratch unchanged. All error types implement
`core::error::Error`, including without `std`, and their enums are
non-exhaustive. Checked geometry does not validate semantic premises such as
a truncated transform's zero coefficient tail.

## Performance

[BENCHMARKS.md](https://github.com/nanithefkuc/butterfly-fft/blob/main/BENCHMARKS.md)
records public transform measurements and the in-process competitor matrix.
The `ntt` benchmark target covers the public NTT surface:

```sh
just bench-save ntt
just bench ntt
```

The separate `benchmarks/afft` project provides the raw additive-transform
competitor harness. It requires `git`, network access, and a C/C++ toolchain:

```sh
cargo bench --manifest-path benchmarks/afft/Cargo.toml --bench raw_afft -- p32_r64 --test
BUTTERFLY_FFT_BENCH_MAX_BYTES=67108864 cargo bench --manifest-path benchmarks/afft/Cargo.toml
```

## Building

From the repository, `just` provides the build and verification commands:

```sh
just build       # release build with all features
just features    # no-default, default, and all-feature tests
just test
just doc
```

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/butterfly-fft/blob/main/LICENSE).
