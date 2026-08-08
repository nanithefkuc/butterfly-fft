# CAFFT Engineering Invariants

These rules apply to the entire repository. They encode decisions that are
expensive to rediscover; violating one is a bug even when the tests pass.

## Scope

`cafft` is a transform engine, not a codec. Field arithmetic and byte-buffer
vector primitives come from `fff` — never re-implement them here. Wire
formats, receipt bookkeeping, and evaluation-point ↔ wire-index maps belong to
consumers.

## Mathematics

- Every mathematical identity in public documentation MUST have an independent
  behavioral check. Never use the implementation under test as its own oracle.
- The Cantor chain is `v_0 = 1`, `v_i² + v_i = v_{i-1}`. Its subspace
  polynomials follow Pascal coefficients modulo two:
  `W_k = Σ_j C(k,j) x^(2^j)`. `W_k = x^(2^k) + x` holds only when `k` is a
  power of two.
- Formal-derivative coverage MUST compare through monomial-basis
  differentiation, not only restate the novel-basis formula.

## Public buffer geometry

- Public byte-row APIs MUST reject `row_len == 0` and partial elements
  consistently before entering a walker.
- Every derived byte length or offset (`size * row_len`, `active * row_len`,
  scratch sizes, row starts) MUST use checked arithmetic before slicing.
- Public documentation MUST state which rows are final outputs. Restricted
  selected/range walkers may leave all other rows as undefined intermediate
  values; never promise they remain untouched.
- Execution walkers remain allocation-free. Validation and dispatch occur once
  at the public boundary, never per butterfly. `tests/zero_alloc.rs` enforces
  this; extend it when adding an execution model.

## Plans and execution surfaces

- `ShiftedPlan` MUST expose the same applicable execution models as
  `TransformPlan`; additions to one require an explicit shifted-facade
  decision and coverage.
- Constructors MUST share size/basis validation so equivalent invalid inputs
  return equivalent errors.
- Do not build and discard a factor table to obtain basis metadata.
- `FactorTable` and other unstable tuning data belong behind the `internals`
  feature; keep stable public APIs opaque.

## Kernels and unsafe dispatch

- `src/core/kernel` owns every intrinsic in the crate. The rest of the crate
  keeps `#![warn(unsafe_code)]`.
- Unsafe target-feature calls MUST be guarded by explicit detection of every
  feature named by `#[target_feature]`; never rely on one feature or backend
  tier implying another.
- Every SIMD forward and inverse kernel MUST be independently
  differential-tested against scalar element arithmetic, including tails,
  zero/one factors, and nontrivial factors. Round trips alone are
  insufficient.
- CI MUST execute each supported runtime backend where hardware permits.
  Cross-compilation alone does not establish architecture-kernel correctness.

## Crate hygiene

- This crate has a module named `core`. Crate code MUST write sysroot paths
  absolutely (`::core::…`, `::std::…`); CI rejects `use core::` and
  `use std::` in `src/`.
- The crate is `no_std` + `alloc` without default features. New code MUST NOT
  reach for `std` outside `#[cfg(feature = "std")]`.
- Public items MUST be documented (`#![warn(missing_docs)]`) and
  `cargo doc --all-features` MUST be warning-free.
- Public documentation and comments MUST NOT reference private or unpublished
  downstream projects.

## Testing changes

- A bug fix MUST include a regression that fails for the observed bug.
- Tests MUST assert exact field values (`ZERO`, `ONE`, or the expected
  element), not predicates that allow unintended nonzero values.
- Run focused regressions first, then the full matrix:

  ```sh
  cargo test --all-features
  cargo test --no-default-features
  SIMD_BACKEND=scalar cargo test --all-features    # and each weaker backend
  # SIMD_BACKEND is owned by simdispatch (Level 0), shared with fff.
  cargo fmt --check
  cargo clippy --all-features --all-targets
  cargo doc --all-features --no-deps
  ```
