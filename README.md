> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things may break. Audit the code yourself, or with
> your own agent before using.

# butterfly-fft — Additive FFT and NTT over Finite Fields

`butterfly-fft` provides reusable additive-FFT plans over binary fields,
multiplicative NTT plans over prime and prime-extension fields, basis
conversion, affine-coset execution, and runtime-dispatched butterfly kernels.

The crate owns transform mathematics, transform-buffer layouts, factor tables,
and butterfly kernels. [`fgf`](https://github.com/nanithefkuc/fgf) owns field
arithmetic and byte-buffer field primitives. FEC consumers own wire formats,
codec shells, and evaluation-point-to-wire-index mappings.

## Usage

The MSRV is Rust 1.93, edition 2024. Field arithmetic uses `fgf` 1.0.0 from
crates.io.

`butterfly-fft` is distributed through git only; it is not published to
[crates.io](https://crates.io).

```toml
[dependencies]
butterfly-fft = { git = "https://github.com/nanithefkuc/butterfly-fft" }
```

Portable `no_std` builds are available. They use `alloc` for plans and tables
and the portable scalar backend:

```toml
[dependencies]
butterfly-fft = { git = "https://github.com/nanithefkuc/butterfly-fft", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| default (`std`, `simd`) | shared plan caches and runtime-dispatched butterfly kernels |
| `std` without `simd` | allocation-backed plans with portable scalar execution |
| `simd` | runtime-selected vector butterflies; implies `std` |
| `internals` | unstable factor tables and implementation APIs for experiments |
| `--no-default-features` | `no_std` plus `alloc`, portable scalar execution |

### Platforms

| Platform | Result |
| --- | --- |
| `x86`/`x86_64` | GFNI/AVX2/SSSE3 butterfly dispatch for GF(2^8) and GF(2^16) |
| `AArch64` | NEON butterfly dispatch for GF(2^8) and GF(2^16) |
| wasm32 and other targets | portable scalar butterflies |
| wider `fgf` fields | portable scalar butterflies |

## Quick start

A plan is built once for a field and power-of-two domain size, then reused.
Element transforms operate in place and do not allocate:

```rust
use butterfly_fft::TransformPlan;
use fgf::{Gf16, gf16};

let plan = TransformPlan::<Gf16>::new(4).expect("valid transform plan");
let mut values = [
    gf16::Elem::from_raw(0x1234),
    gf16::Elem::from_raw(0xabcd),
    gf16::Elem::from_raw(0x0108),
    gf16::Elem::from_raw(0xffff),
];
let coefficients = values;

plan.forward(&mut values).expect("valid transform input");
plan.inverse(&mut values).expect("valid transform input");
assert_eq!(values, coefficients);
```

Payloads use `size` rows of `row_len` bytes. Each element column is transformed
in place, allowing the butterfly kernels to process independent columns at
vector width:

```rust
use butterfly_fft::TransformPlan;
use fgf::Gf16;

let plan = TransformPlan::<Gf16>::new(16).expect("valid transform plan");
let mut rows = vec![0u8; 16 * 64];
plan.forward_bytes(&mut rows, 64)
    .expect("valid byte-row geometry");
plan.inverse_bytes(&mut rows, 64)
    .expect("valid byte-row geometry");
```

With the `std` feature, `TransformPlan::shared` caches one plan per field and
size process-wide, and `PlanCache` holds an owned, droppable cache.

Byte-row APIs reject zero row lengths, partial elements, and checked-geometry
overflows before execution. Restricted selected, range, and truncated walkers
define only their documented output rows; other rows may contain intermediate
values.

## Transform modules

| Module | Result |
| --- | --- |
| `transform` | [`TransformPlan`] forward, inverse, derivative, selected-output, range, truncated, and high-coset execution |
| `internals` | unstable factor tables and tuning schedules (feature-gated) |
| `kernel` | fused butterfly kernels and runtime SIMD dispatch |
| `basis` | [`basis::BitBasis`], [`basis::CantorBasis`], [`basis::CoordinateMap`], and monomial/novel conversion |
| `ntt` | [`NttPlan`] radix-two multiplicative (number-theoretic) transforms over prime and prime-extension fields |

## Building

`butterfly-fft` builds on stable Rust without target-feature flags; SIMD kernels
are selected at runtime:

```sh
cargo build                        # default: std + simd
cargo build --no-default-features  # portable no_std + alloc
cargo test --all-features
cargo doc --all-features --no-deps
```

## Backends

`kernel::backend()` reports the process-wide backend. The backend ladder
and downgrade-only `SIMD_BACKEND` override come from
[`simdispatch`](https://github.com/nanithefkuc/simdispatch). The supported
ordering is exposed as `kernel::BUTTERFLY_FFT_TIERS`.

| Identifier | Target and requirements | Butterfly lane width |
| --- | --- | --- |
| `v3_gfni_crypto` | x86 AVX2 + GFNI + crypto | 32 bytes |
| `v3` | x86 AVX2 shuffle | 32 bytes |
| `v2` | x86 SSSE3/SSE4.2 shuffle | 16 bytes |
| `neon_aes` | `AArch64` NEON + AES/PMULL | 16 bytes |
| `neon` | `AArch64` NEON split-nibble shuffle | 16 bytes |
| `scalar` | portable fallback | scalar |

`SIMD_BACKEND=v3_gfni_crypto|v3|v2|neon_aes|neon|scalar` requests a backend at
process startup. The override is downgrade-only; an unsupported upgrade is
ignored. Backends are re-exported from `simdispatch`.

## Benchmarks

`BENCHMARKS.md` records the pinned measurements for the public transform
shapes, the derivative-schedule crossovers, and the in-process competitor
comparison on the project's benchmark hosts.

The `benchmarks/afft` project contains the standalone raw-transform harness
behind the competitor matrix. It requires `git`, network access, and a working
C/C++ toolchain:

```sh
cargo bench --manifest-path benchmarks/afft/Cargo.toml --bench raw_afft -- p32_r64 --test
BUTTERFLY_FFT_BENCH_MAX_BYTES=67108864 cargo bench --manifest-path benchmarks/afft/Cargo.toml
```

## License

MIT - see [LICENSE](https://github.com/nanithefkuc/butterfly-fft/blob/main/LICENSE)
