# Changelog

All notable changes to butterfly-fft are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `basis::interpolate_bytes_scratch` validates the scratch geometry before
  transforming, so a rejected call leaves the input rows unchanged instead of
  half-transformed.

### Changed (NTT execution schedule)

- `NttPlan` butterfly stages select among three measured schedules: the
  fused register form (fields without vector elementwise kernels, every
  row width), a batched region form — one elementwise twiddle multiply,
  one region copy, one region subtract and add per bounded batch instead
  of four op calls per pair — and the packed op-call form at and above
  128-byte rows, where its broadcast multiply reads no twiddle stream and
  wins. The pinned fgf moves to 1.1.0, whose QuadMersenne31 vector
  kernels and Goldilocks correctness fix the wide-row paths ride. All
  three schedules stay reachable through `internals::ntt_forward_fused`,
  `internals::ntt_forward_batched`, and `internals::ntt_forward_packed`,
  and the campaign record, including a measured rejection of a four-step
  cache-blocked decomposition, lives in `BENCHMARKS.md`. `NttScratch`
  grows the batched schedule's two bounded region buffers, and scratch
  reused across plans of different sizes is rejected with
  `NttError::ScratchTooSmall` before anything is written.

### Changed (naming and layout, fgf alignment)

- **Breaking:** the derivative pair takes the destination first with `_into`
  overwrite semantics: `derivative(coefficients, derivative)` becomes
  `derivative_into(derivative, coefficients)`, and
  `derivative_bytes(coefficients, row_len, derivative)` becomes
  `derivative_into_bytes(derivative, row_len, coefficients)`. The tuning
  schedules follow as `internals::derivative_into_bytes_sweep`,
  `derivative_into_bytes_gather`, and `derivative_into_bytes_overwrite`,
  all `(plan, derivative, row_len, coefficients)`.
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
- `BENCHMARKS.md` records the derivative-schedule crossovers and the pinned
  public-transform and competitor measurements.

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
  moved to a crate-private trait behind a private dispatch proof, after
  `fgf`'s `KernelDispatch`/`RawDispatch` pattern. The public surface is
  fully safe.
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
- The `internals` surface is now a single facade: the tuning derivative
  schedules and factor-table inspection moved from `#[cfg(feature =
  "internals")]` methods to `internals::derivative_into_bytes_sweep`,
  `derivative_into_bytes_gather`, `derivative_into_bytes_overwrite`, and
  `internals::plan_table`, taking the plan as their first argument.
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
  admits only size 2 and the binary fields only the identity, all decided by
  the `size | |F| - 1` rule rather than a field special case.
- `TransformPlan::vanishing_polynomial` returns the dense monomial
  coefficients of the domain vanishing polynomial `G(X)` for both subspace and
  affine-coset domains, and `TransformPlan::shift` exposes the coset shift.
- `basis::interpolate_bytes_scratch` composes inverse transform with
  novel-to-monomial conversion for allocation-free received-word
  interpolation, and `TransformPlan::derivative_plus_identity_bytes`
  computes `c + D(c)` in place for workflows that evaluate only at roots
  of `c`.

### Changed

- **Breaking:** Field types come from the crates.io release `fgf` 1.0.0
  instead of the git dependency. Consumers must use the same registry
  dependency to share field types with transform plans. Serialization uses
  `Field::decode`/`Field::encode`, and tower decomposition uses
  `Elem::to_components`. The minimum supported Rust version is 1.93.
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
- `TransformPlan::derivative_bytes` selects measured source-sweep,
  overwrite-first, and destination-gather schedules by row geometry and
  backend. Wide-row Cantor derivatives improved by up to 36%, while short rows
  retain the lower-overhead sweep.
- Transform byte walkers execute dimensions of three or fewer as explicit
  fused base cases instead of per-level recursion, cutting per-call kernel
  setup roughly fourfold at the bottom of the tree. Measured up to 34%
  faster on mid-size payloads.
- Fused butterflies gained a unit-coefficient fast path alongside the
  zero-coefficient path: `c = 1` runs as two XOR passes with no field
  multiply.
- The benchmark harness tracks the crate's `fgf` pin again (`Gf8B` rename)
  and compares GF(2^16) against leopard on the same Cantor basis. Exact `D(c)`
  and decoder-equivalent `c + D(c)` are now separate persistent-buffer series;
  the prior comparison mislabeled Leopard's in-place augmented derivative as
  an exact derivative and biased cache setup toward its single buffer.
