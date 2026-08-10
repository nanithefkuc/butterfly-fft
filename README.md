> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things may break. Audit the code yourself, or with
> your own agent before using.

# cafft

**Common Additive Fast Fourier Transform** — a shared additive-FFT engine over
binary fields.

`cafft` is the transform layer that erasure codecs, polynomial commitment
schemes, and other GF(2^m) consumers keep re-implementing: subspace twiddle
tables, SIMD-batched butterfly kernels, and in-place execution models, plus the
basis, shifted-coset, and truncated variants those consumers actually need.

Field arithmetic and byte-buffer vector primitives come from
[`fgf`](https://github.com/nanithefkuc/fgf); this crate never re-implements
field arithmetic. Wire formats, codec shells, and evaluation-point ↔ wire-index
maps stay with the consumer.

## Status

Pre-1.0. The API is usable and covered by tests, but not yet stable. `cafft`
is not on crates.io because it depends on `fgf` by git; depend on it the same
way.

```toml
[dependencies]
cafft = { git = "https://github.com/nanithefkuc/cafft.git" }
```

## Quick start

```rust
use cafft::core::transform::TransformPlan;
use fgf::{Gf16, gf16};

// A plan is built once per (field, size) and reused. Execution allocates
// nothing.
let plan = TransformPlan::<Gf16>::new(4)?;

let mut values = [
    gf16::Elem(0x1234),
    gf16::Elem(0xabcd),
    gf16::Elem(0x0108),
    gf16::Elem(0xffff),
];
let coefficients = values;

// Forward: novel-basis coefficients -> evaluations at points 0..4.
plan.forward(&mut values)?;
// Inverse: back to coefficients.
plan.inverse(&mut values)?;
assert_eq!(values, coefficients);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Real payloads are not single elements. The byte-row API transforms `size` rows
of `row_len` bytes each, one transform per column of elements, which is what
lets the SIMD kernels run at width:

```rust
use cafft::core::transform::TransformPlan;
use fgf::Gf16;

let plan = TransformPlan::<Gf16>::shared(256)?; // process-wide cached plan
let mut rows = vec![0u8; 256 * 4096];
plan.forward_bytes(&mut rows, 4096)?;
plan.inverse_bytes(&mut rows, 4096)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## What's in the box

| Module | Contents |
| --- | --- |
| [`core::transform`] | [`TransformPlan`]: forward, inverse, formal derivative, selected-output, range-restricted, truncated, and high-coset execution — element and byte-row flavors. Plus [`PlanCache`] and a process-wide shared cache. |
| [`core::factors`] | Subspace twiddle and derivative factor tables. |
| [`core::kernel`] | Fused butterfly kernels with runtime SIMD dispatch. |
| [`basis`] | Ordered GF(2)-bases ([`BitBasis`], [`CantorBasis`]), [`CoordinateMap`] change of basis, and monomial ↔ novel coefficient conversion. |
| [`shifted`] | [`ShiftedPlan`]: the same execution models over an affine coset `α + V`, at identical cost. |
| [`rs`] | RS-facing erasure algebra: locator evaluation, Forney-style recovery, systematic-locator caches, dense targeted solve, strip-blocked encode. |

[`core::transform`]: https://github.com/nanithefkuc/cafft/blob/main/src/core/transform.rs
[`core::factors`]: https://github.com/nanithefkuc/cafft/blob/main/src/core/factors.rs
[`core::kernel`]: https://github.com/nanithefkuc/cafft/blob/main/src/core/kernel/mod.rs
[`basis`]: https://github.com/nanithefkuc/cafft/blob/main/src/basis/mod.rs
[`shifted`]: https://github.com/nanithefkuc/cafft/blob/main/src/shifted.rs
[`rs`]: https://github.com/nanithefkuc/cafft/blob/main/src/rs/mod.rs

Full API documentation:

```sh
cargo doc --all-features --no-deps --open
```

## Design guarantees

- **Allocation-free execution.** A plan owns its tables. Every walker runs in
  place over caller-provided buffers; validation and backend dispatch happen
  once at the public boundary, never per butterfly. Enforced by
  `tests/zero_alloc.rs`.
- **Checked geometry.** Every derived byte length or offset (`size * row_len`,
  row starts, scratch sizes) is computed with checked arithmetic before any
  slice is taken. `row_len == 0` and partial trailing elements are rejected at
  the boundary.
- **Explicit output rows.** Restricted walkers (selected, range, truncated)
  document exactly which rows are final outputs. Every other row is an
  undefined intermediate — never rely on it.
- **Differentially tested kernels.** Each SIMD forward and inverse kernel is
  tested against portable scalar element arithmetic, including tails and
  zero/one/nontrivial factors. Round-trip tests alone are not accepted.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | yes | Runtime CPU detection and shared plan caches. |
| `simd` | yes | Vector butterfly backends. Implies `std`. |
| `rs` | no | RS erasure algebra (`cafft::rs`). |
| `internals` | no | Unstable implementation APIs. Exempt from compatibility guarantees. |

With `--no-default-features` the crate is `no_std` (it still needs `alloc`) and
runs the portable scalar backend.

## Backends

GF(2^8) and GF(2^16) have dedicated kernels on x86 (`v3_gfni_crypto`,
`v3`, `v2`) and AArch64 (`neon`; PMULL hosts resolve `neon_aes`). AVX-512
hosts run the 32-byte GFNI kernels. WebAssembly and wider fields run scalar.

Backends and their detection are a re-export of
[`simdispatch::Backend`](https://docs.rs/simdispatch), resolved once per
process over `cafft::core::kernel::CAFFT_TIERS`. Set the one stack-wide
`SIMD_BACKEND` env var to `v3_gfni_crypto`, `v3`, `v2`, `neon_aes`, `neon`,
or `scalar` to force a *weaker* backend than the host supports — useful for
testing. Requests for a backend the host cannot execute are ignored.

```sh
SIMD_BACKEND=scalar cargo test --all-features
```

## Minimum supported Rust version

1.89, edition 2024. An MSRV bump is a minor-version change.

## Benchmarks

`benchmarks/afft` is a standalone crate (its own workspace, excluded from the
published package) comparing `cafft` against Leopard, nanors, and
`additive-fft-reed-solomon`. Its build script clones the pinned upstream C/C++
sources and compiles native adapters, so it needs `git`, network access, and a
working host C/C++ toolchain.

```sh
# one smoke case
cargo bench --manifest-path benchmarks/afft/Cargo.toml --bench raw_afft -- p32_r64 --test
# full matrix, capped payload size
CAFFT_BENCH_MAX_BYTES=67108864 cargo bench --manifest-path benchmarks/afft/Cargo.toml
```

## A note on `core`

This crate has a module named `core`. Inside crate code, sysroot paths must be
written absolutely (`::core::…`, `::std::…`); a relative `core::…` resolves to
the local module. CI enforces this.

## License

MIT. See [LICENSE](LICENSE).
