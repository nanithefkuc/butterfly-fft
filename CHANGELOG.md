# Changelog

All notable changes to butterfly-fft are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.1] - 2026-09-24

### Changed

- The `fgf` dependency resolves the registry release `=1.2.0`, matching every
  other consumer in the closure, so one `fgf` and one `FieldKernels` trait
  compile in a graph that also contains the polynomial crates.

## [1.0.0] - 2026-09-18

This dated entry describes the prepared 1.0.0 changes; it does not assert
that the package has been published.

### Changed (checked execution)

- **Breaking:** additive byte-row execution and coefficient conversion return
  `TransformError` for invalid row lengths and overflowing geometry instead
  of panicking. Selected, range, and truncated operations also return errors
  for invalid selections, ranges, and active prefixes. Handle or propagate
  these errors rather than catching a panic; rejected calls leave destination
  and scratch buffers unchanged.
- **Breaking:** `inverse_bytes_truncated_scratch_rows(active)` now returns
  `Result<usize, TransformError>`. Use
  `plan.inverse_bytes_truncated_scratch_rows(active)?` before allocating
  workspace, and handle an invalid active prefix.
- **Breaking:** `TransformLengthError` is replaced by the non-exhaustive
  `TransformError` enum. Replace struct construction or field access with
  `TransformError::BufferLength { expected, got }`, and handle geometry and
  workspace variants separately. Add a wildcard arm when matching any public
  error enum. `PlanError`, `TransformError`, and `NttError` implement
  `core::error::Error` with or without `std`.
- **Breaking:** four-step NTT experiments use `internals::FourStepPlan` and
  `internals::FourStepScratch`. Replace `internals::ntt_forward_fourstep`
  with explicit `FourStepPlan::<F>::new(size)`, `.scratch(row_len)`, and
  `.forward_bytes_scratch(rows, row_len, &mut scratch)`. Stable `NttPlan`
  preparation no longer builds experimental subplans, and `NttScratch` no
  longer contains experimental transpose state. The implementation compiles
  regardless of `internals`; the feature only exposes the facade.

### Changed (kernels)

- The SIMD butterfly kernels are safe `archmage` capability-token functions:
  every `#[target_feature] unsafe fn` and pointer-loop in `kernel/x86.rs`,
  `kernel/x86/gfni.rs`, and `kernel/aarch64.rs` is now a `#[arcane]`
  function taking the exact token its instructions require
  (`X64V3GfniCryptoToken`, `X64V3Token`, `X64V2Token`, `NeonToken`), walking
  the halves with reference-based loads and stores over chunk arrays. The
  crate-private `RawDispatch` proof is gone — the tokens are the proof — and
  the crate is now `#![deny(unsafe_code)]` with no unsafe surface and no
  residue. `archmage` 0.9.29 joins as a dependency of the `simd` feature,
  at the release `fgf` resolves.

### Fixed

- `basis::interpolate_bytes_scratch` validates the scratch geometry before
  transforming, so a rejected call leaves the input rows unchanged instead of
  half-transformed.

### Changed (NTT execution schedule)

- `NttPlan` selects fused, batched, or packed butterfly execution from the
  row geometry and field backend. Explicit controls remain available through
  `internals::ntt_forward_fused`, `ntt_forward_batched`, and
  `ntt_forward_packed`. These free functions take destination, row length,
  plan, and scratch in that order.
- **Breaking:** `NttError::ScratchTooSmall` is replaced by
  `ScratchRowTooSmall`, `ScratchBatchTooSmall`, and `ScratchTransposeTooSmall`.
  Match the relevant capacity variant rather than interpreting every shortage
  as row bytes. `NttScratch` includes bounded batch buffers; reuse with another
  plan requires sufficient row and batch capacity. Four-step transpose capacity
  is checked separately by `FourStepScratch`.

### Changed (naming and layout, fgf alignment)

- **Breaking:** the derivative pair takes the destination first with `_into`
  overwrite semantics: `derivative(coefficients, derivative)` becomes
  `derivative_into(derivative, coefficients)`, and
  `derivative_bytes(coefficients, row_len, derivative)` becomes
  `derivative_into_bytes(derivative, row_len, coefficients)`. The tuning
  schedules follow as `internals::derivative_into_bytes_sweep`,
  `derivative_into_bytes_gather`, and `derivative_into_bytes_overwrite`,
  all `(derivative, row_len, plan, coefficients)`.
- **Breaking:** `_scratch` marks every workspace-taking operation:
  `NttPlan::forward_bytes_scratch` and `inverse_bytes_scratch` (from
  `forward_bytes` / `inverse_bytes`),
  `novel_to_monomial_bytes_scratch` and `monomial_to_novel_bytes_scratch`
  (from `novel_to_monomial_bytes` / `monomial_to_novel_bytes`),
  `interpolate_bytes_scratch` (from `inverse_interpolate_bytes`), and
  `TransformPlan::inverse_bytes_truncated_scratch` (from
  `inverse_truncated_bytes`, with the sizing query
  `inverse_bytes_truncated_scratch_rows`).
- **Breaking:** restriction markers are spelled out and follow the layout
  marker: `forward_bytes_trunc_range` becomes
  `forward_bytes_truncated_range`.
- The crate-private `xor_scaled_bytes` wrapper is deleted; its call sites
  use `fgf::ops::mul_add`, whose zero/one fast paths and panic contract it
  duplicated.
- The crate-private backend-parameterized butterflies are renamed
  `fused_forward_backend` / `fused_inverse_backend`, keeping the `_with`
  suffix reserved for prepared state.
- Per-field kernel wiring moves to `kernel/fields.rs` and the recursive
  transform walkers to `transform/walk.rs`; `kernel.rs` and `transform.rs`
  keep the shared contracts. No public paths change.
- Crate documentation comes from `README.md` through
  `#![doc = include_str!]`, so the README examples compile as doctests; the
  README and package description cover the NTT family beside the additive
  FFT.
- `missing_docs` is denied rather than warned.
- `BENCHMARKS.md` contains public API and competitor measurements only.

### Changed (API reshape)

- **Breaking:** the `core` module is gone. `core::transform` is now
  [`transform`](butterfly_fft::transform), `core::kernel` is
  [`kernel`](butterfly_fft::kernel), and the factor tables live privately
  inside the transform subtree. `TransformPlan`, `NttPlan`, and the three
  error types are re-exported at the crate root.
- **Breaking:** `ShiftedPlan` is folded into `TransformPlan`:
  `TransformPlan::with_shift(size, basis, shift)` builds a coset plan,
  `point_element` is shift-aware (`shift ⊕ point`) for every plan, and
  `points()` lists the evaluation points. The `shifted` module is removed.
- **Breaking:** `NttError` lives in `error` (re-exported from `ntt`).
- **Breaking:** the per-tier `unsafe` kernel entries are no longer public
  API. `ButterflyKernels` keeps only `BUTTERFLY_TIERS`; the kernel entries
  moved to a crate-private trait whose safe functions require genuine
  `archmage` capability tokens. The public surface is fully safe.
- `xor_scaled_bytes` and `xor_scaled_bytes_rows` are no longer public:
  row scaling is `fgf::ops::mul_add`'s contract, and this crate's public
  kernels are the fused butterflies it owns.
- The scratch-taking basis conversions are renamed
  `novel_to_monomial_scratch` / `monomial_to_novel_scratch`, reserving the
  `_with` suffix for reusable prepared state.
- `CoordinateMap::of` is renamed `from_basis`, and `to_element` documents
  its panic on out-of-range coordinates.

### Changed (dependency sweep)

- **Breaking:** `NttPlan::forward_bytes_scratch`, `inverse_bytes_scratch`,
  and `scratch` reject a zero row length with `NttError::InvalidRowLength`,
  matching the additive byte-row APIs and the documented crate rule; they
  were previously accepted no-ops. A scratch built for a different element
  width now reports the new `NttError::ScratchFieldMismatch` instead of a
  self-contradictory `InvalidRowLength`, and `UnsupportedSize`'s message no
  longer blames a missing root when only the table cap rejected the size.
- **Breaking:** the `basis` module (`OrderedBasis`, `BitBasis`,
  `CoordinateMap`, `CantorBasis`, `point_of`) is bounded by
  `ButterflyKernels`, sealing it to binary extension fields; its GF(2)
  coordinate algebra silently misbehaved over prime fields.
- `Gf8D` (the `0x11D` Reed–Solomon interop field) implements
  `ButterflyKernels`, so additive plans and basis conversion work over it on
  the portable scalar backend.
- The `internals` surface is a single re-export-only facade. Tuning derivative
  functions take `(derivative, row_len, plan, coefficients)`;
  `internals::plan_table` takes the plan. Their implementation no longer
  depends on the feature being enabled.
- `TransformPlan::inverse_bytes_truncated_scratch` documents its
  precondition: the evaluations must be the forward transform of a
  polynomial whose novel-basis coefficients `active..` vanish.

- Removed the `rs` feature and RS-specific encoding, locator, recovery, and
  targeted-solve helpers. Systematic Reed-Solomon codec algorithms now live in
  `srs`; this crate exposes only codec-neutral transform operations.


### Added

- `ntt` module: a second transform family over the same object, a radix-two
  Cooley-Tukey multiplicative (number-theoretic) transform. `NttPlan`
  precomputes the root of unity, bit-reversal permutation, both twiddle
  directions, and the `n^-1` scale, so `forward_bytes_scratch` and
  `inverse_bytes_scratch` run over packed byte rows with no allocation and
  no coefficient preparation.
  Goldilocks and `QuadMersenne31` admit sizes to `2^20`; base `Mersenne31`
  admits sizes one and two, and binary fields admit only the identity.
- `TransformPlan::vanishing_polynomial` returns the dense monomial
  coefficients of the domain vanishing polynomial `G(X)` for both subspace and
  affine-coset domains, and `TransformPlan::shift` exposes the coset shift.
- `basis::interpolate_bytes_scratch` composes inverse transform with
  novel-to-monomial conversion for allocation-free received-word
  interpolation, and `TransformPlan::derivative_plus_identity_bytes`
  computes `c + D(c)` in place for workflows that evaluate only at roots
  of `c`.

### Changed

- **Breaking:** field types come from the exact registry dependency
  `fgf = "=1.1.0"` instead of the git dependency. Consumers must use the same
  package source and version to share field types with transform plans.
  Serialization uses `Field::decode`/`Field::encode`, and tower decomposition
  uses `Elem::to_components`. The minimum supported Rust version is 1.93.
- Updated the AArch64 butterfly helpers and README example to use public
  element constructors and raw-value accessors. The standalone AFFT
  comparison project uses the same registry field dependency.
- Renamed the crate from `cafft` to `butterfly-fft`, including its package,
  library identifier, repository URL, and dependent feature paths.
- Replaced the former `fff` field dependency with `fgf`, preserving transform,
  basis-conversion, dispatch, and allocation behavior.
- Took up fgf's tightened field-element encapsulation: the x86 butterfly
  kernels and the kernel unit tests construct and read `gf8b`/`gf16`
  elements through `Elem::from_raw`/`Elem::to_raw` instead of the tuple
  field, which fgf narrowed to crate visibility. No behavior change.
- `TransformPlan::derivative_into_bytes` selects source-sweep,
  overwrite-first, or destination-gather execution by row geometry and backend.
- Transform byte walkers use fused base cases for small dimensions.
- Fused butterflies gained a unit-coefficient fast path alongside the
  zero-coefficient path: `c = 1` runs as two XOR passes with no field
  multiply.
- The benchmark harness tracks the crate's `fgf` pin again (`Gf8B` rename)
  and compares GF(2^16) against leopard on the same Cantor basis. Exact `D(c)`
  and decoder-equivalent `c + D(c)` are now separate persistent-buffer series;
  the prior comparison mislabeled Leopard's in-place augmented derivative as
  an exact derivative and biased cache setup toward its single buffer.
