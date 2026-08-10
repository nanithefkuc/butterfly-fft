# AGENTS.md

Working rules for `butterfly-fft`: the operational summary of how the crate is
built, tested, and extended.

## What this crate is

`butterfly-fft` is the shared additive-FFT transform layer for binary fields. It
owns subspace and affine-coset transform mathematics, transform-buffer layouts,
factor tables, basis conversion, and fused butterfly kernels.

It is not a codec and does not own wire formats, receipt bookkeeping, or
mapping between evaluation points and wire indexes. Field arithmetic and
byte-buffer vector primitives come from [`fgf`](https://github.com/nanithefkuc/fgf);
never re-implement them here.

Rust imports use the package's library identifier, `butterfly_fft`.

## Hard rules

1. **Independent mathematics checks.** Every identity documented publicly needs
   an independent behavioral check. Never use the implementation under test as
   its own oracle. Formal derivatives are checked through monomial-basis
   differentiation, not only by restating a novel-basis formula.
2. **Checked public geometry.** Byte-row APIs reject `row_len == 0` and partial
   elements before entering a walker. Every derived byte length, offset, row
   start, and scratch size uses checked arithmetic before slicing.
3. **Explicit output contracts.** Restricted selected, range, and truncated
   walkers document which rows are final outputs. Other rows may contain
   undefined intermediate values and must not be consumed.
4. **Allocation-free execution.** Plans own their tables and walkers operate in
   place over caller-provided buffers. Validation and backend dispatch happen at
   the public boundary, never per butterfly. `tests/zero_alloc.rs` enforces
   this invariant.
5. **Shared plan validation.** `ShiftedPlan` exposes the same applicable
   execution models as `TransformPlan`. Constructors share size and basis
   validation so equivalent invalid inputs return equivalent errors.
6. **Stable API boundaries.** Factor tables and tuning data remain behind the
   `internals` feature. Do not widen the stable API to expose implementation
   details.
7. **Kernel ownership.** Every intrinsic belongs under `src/core/kernel`.
   Unsafe target-feature calls must check every feature named by their
   `#[target_feature]`. Every SIMD forward and inverse kernel is differentially
   tested against scalar field arithmetic, including tails, zero/one factors,
   and nontrivial factors.
8. **Sysroot paths.** The crate has a module named `core`; sysroot paths in
   crate code must be absolute (`::core::...`, `::std::...`).
9. **Feature discipline.** The crate is `no_std` plus `alloc` without default
   features. New code must not reach for `std` outside `#[cfg(feature = "std")]`.
10. **Documentation.** Public items stay documented and
    `cargo doc --all-features` remains warning-free.

## Backend selection

`simdispatch` is the single source for backend detection, ordering, and the
stack-wide downgrade-only `SIMD_BACKEND` override. `core::kernel::BUTTERFLY_FFT_TIERS`
contains the tiers implemented by this crate. Do not add a crate-local CPU
probe or environment override.

## Testing

A bug fix includes a regression that fails for the observed bug. Tests assert
exact field values and use an independent oracle. Run focused regressions first,
then the full matrix:

```sh
cargo test --all-features
cargo test --no-default-features
SIMD_BACKEND=scalar cargo test --all-features
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

The excluded `benchmarks/afft` project is separate from the published package.
Performance changes require a recorded benchmark comparison; do not commit a
crossover threshold based only on reasoning.
