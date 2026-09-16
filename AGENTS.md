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
5. **Coset plans share validation.** `TransformPlan::with_shift` runs the
   same size and basis validation as the subspace constructors, so
   equivalent invalid inputs return equivalent errors.
6. **Stable API boundaries.** Factor tables and tuning data remain behind the
   `internals` feature. Do not widen the stable API to expose implementation
   details.
7. **Kernel ownership.** Every intrinsic belongs under `src/kernel`. The
   per-tier kernel entries live on the crate-private `TierButterflies`
   trait, sealed behind the private `RawDispatch` proof; the public
   `ButterflyKernels` trait carries no `unsafe` surface. Every SIMD forward
   and inverse kernel is differentially tested against scalar field
   arithmetic, including tails, zero/one factors, and nontrivial factors.
8. **Sysroot paths.** Sysroot paths in crate code are written absolutely
   (`::core::...`, `::std::...`); CI enforces the form.
9. **Feature discipline.** The crate is `no_std` plus `alloc` without default
   features. New code must not reach for `std` outside `#[cfg(feature = "std")]`.
10. **Documentation.** Public items stay documented and
    `cargo doc --all-features` remains warning-free.

## Backend selection

`simdispatch` is the single source for backend detection, ordering, and the
stack-wide downgrade-only `SIMD_BACKEND` override. `kernel::BUTTERFLY_FFT_TIERS`
contains the tiers implemented by this crate. Do not add a crate-local CPU
probe or environment override.

## Tooling

`just validate` is the pull-request gate; the shared recipe surface is
documented once in the umbrella's root `AGENTS.md`. Crate specifics:

- The `internals` facade is the crate's single `feature = "internals"` site.
  The facade-only items in `src/tuning.rs`, `TransformPlan::table`, and the
  `FactorTable` accessors carry per-item `#[allow(dead_code)]`: they are
  alive whenever the facade is compiled and unreachable otherwise, which is
  the visibility-only exception the mechanism permits.
- Test target `factor_table` requires `internals`; it exercises the facade
  surface against the public derivative as oracle.

- **`TIERS = v3_gfni_crypto v3 v2 scalar`**, matching the tiers
  `kernel::BUTTERFLY_FFT_TIERS` declares. Dispatch resolves one backend
  per process, so `just test-tiers` and `just cover` re-run the suite once per
  tier; a single host run only ever exercises the strongest.
- **`MIRI = --no-default-features`.** The `no_std` + `alloc` closure is what
  miri can execute, and the run covers the safe wrappers over `src/kernel`
  — walker geometry, scratch sizing, and the checked byte-row
  kernel tests. Superlinear test sweeps and large transform magnitudes
  shrink under `cfg!(miri)` (see the `log_cap` / `max_log` / `max_dimension`
  helpers and the geometry lists in `tests/zero_alloc.rs`): miri validates
  geometry and safety, not throughput, and every boundary class —
  multi-element rows, partial-element tails, awkward truncations — stays
  represented at the smaller sizes. The full run sits near twenty-six
  minutes; growing it past that needs a reason.
- **`COV_IGNORE` is empty**: every line counts toward the 95% gate.
- **Bench target:** `ntt` — `just bench-save ntt`, then `just bench ntt`.
  The `ntt_tuning` target (requires `internals`) drives the fused,
  batched, packed, and four-step NTT butterfly schedules past production
  selection for crossover measurement, validating every arm against the
  fused arm and an exact round trip before timing; its record is the
  "Butterfly schedules" section of `BENCHMARKS.md`, which also carries the
  four-step rejection and its revisit bar.
- `justfile` is a byte-identical vendored copy; never edit it here or the
  umbrella's `just drift` check fails. Crate-specific values and recipes belong
  in `crate.just`.

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
