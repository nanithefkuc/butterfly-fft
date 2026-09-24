# AGENTS.md

Working rules for `butterfly-fft`: the operational summary of how the crate is
built, tested, and extended.

## What this crate is

`butterfly-fft` provides additive FFTs over binary fields and multiplicative
NTTs over supported finite fields. It owns transform mathematics,
transform-buffer layouts, factor tables, basis conversion, and fused butterflies.

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
2. **Checked public geometry.** Additive execution returns `TransformError`
   for invalid buffers, workspace, selections, ranges, active prefixes, and
   byte-row geometry before modifying any destination or scratch. NTT methods
   return `NttError`. Every derived byte length and offset is checked before
   slicing. All public error enums are non-exhaustive and implement
   `core::error::Error`, including without `std`.
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
6. **Stable API boundaries.** Factor tables and tuning APIs are reachable only
   through the re-export-only `internals` facade. The underlying code compiles
   unconditionally. Experimental four-step state belongs to `FourStepPlan` and
   `FourStepScratch`, not stable `NttPlan` or `NttScratch` preparation.
7. **Kernel ownership.** Every intrinsic belongs under `src/kernel`. The
   per-tier kernel entries live on the crate-private `TierButterflies`
   trait, sealed by the private `Sealed` supertrait bound, and take the
   genuine `archmage` capability token their instructions require —
   summoned once per process into the cached token bank (`X86_TOKENS`,
   `NEON_TOKEN`) and never forged. Every kernel is a safe `#[arcane]`
   function over reference-based loads and stores; the public
   `ButterflyKernels` trait carries no unsafe surface. Every SIMD forward
   and inverse kernel is differentially tested against scalar field
   arithmetic, including tails, zero/one factors, and nontrivial factors.

8. **Sysroot paths.** Sysroot paths in crate code are written absolutely
   (`::core::...`, `::std::...`); CI enforces the form.
9. **Feature discipline.** The crate is `no_std` plus `alloc` without default
   features. New code must not reach for `std` outside `#[cfg(feature = "std")]`.
10. **Documentation.** Public items stay documented and
    `just doc` remains warning-free.


## Unsafe residue ledger

The crate is `#![deny(unsafe_code)]` and owns no unsafe surface: every SIMD
kernel is a safe `archmage` capability-token function, and nothing in `src/`
falls into the sanctioned residue classes (offset-addressed rows,
uninitialized scratch, non-temporal or aligned-only stores, provider
callbacks). The only unsafe in the repository is the `GlobalAlloc` impl in
`tests/zero_alloc.rs` (trait-mandated, test-only) and the C-FFI bindings in
the excluded `benchmarks/afft` project.

When a change introduces a survivor, it carries a per-item
`#[allow(unsafe_code)]`, a full SINCE–THUS proof, and an entry here.

## Backend selection

`simdispatch` is the single source for backend detection, ordering, and the
stack-wide downgrade-only `SIMD_BACKEND` override. `kernel::BUTTERFLY_FFT_TIERS`
contains the tiers implemented by this crate. Do not add a crate-local CPU
probe or environment override.

The user-approved exception to the ecosystem's git-until-1.0 dependency rule
is `simdispatch = { version = "=0.1.0", default-features = false }` from the
registry. It must resolve the same package as `fgf` so both crates share one
`Backend` type. This exact source/version exception does not authorize other
pre-1.0 registry dependencies or local implementations of lower-layer algebra.

## Tooling

`just validate` is the pull-request gate; the shared recipe surface is
documented once in the umbrella's root `AGENTS.md`. Crate specifics:

- The `internals` facade is the crate's single `feature = "internals"` site.
  Its implementation items compile without that feature; narrowly scoped
  `#[allow(dead_code)]` attributes cover items otherwise unreachable.
- Test target `factor_table` requires `internals` and compares tuning entries
  with their public counterparts.
- **`TIERS` follows the host architecture:** `v3_gfni_crypto v3 v2 scalar` on
  x86, `neon_aes neon scalar` on AArch64, and `scalar` elsewhere.
  `just test-tiers` and `just cover` start a process per requested tier.
  Unsupported upgrades fall back; a requested tier is not execution evidence.
  The harness-free `backend_report` test prints the requested override, resolved
  additive backend, and per-field additive and NTT backends. Inspect that report
  when deciding which paths the host actually exercised.
- **`MIRI` is empty.** The library forbids unsafe. The no-default scalar
  interpreter path is ordinary checked-code coverage, not owned-unsafe Miri
  coverage.
- **`COV_IGNORE` is empty.** GFNI source is not excluded from coverage.
  Direct-kernel differential tests run when the host can summon the required
  token; unsupported hosts skip those kernels. A passing run does not imply
  that every backend was exercised.
- **Bench target:** `ntt` — `just bench-save ntt`, then `just bench ntt`.
  The `ntt_tuning` target (requires `internals`) compares fused, batched,
  packed, and explicitly prepared four-step execution outside production
  selection. Keep internal schedule measurements in ignored local storage.
  `BENCHMARKS.md` contains public API and competitor measurements only.
- `justfile` is a byte-identical vendored copy; never edit it here or the
  umbrella's `just drift` check fails. Crate-specific values and recipes belong
  in `crate.just`.

## Testing

A bug fix includes a regression that fails for the observed bug. Tests assert
exact field values and use an independent oracle. Run focused regressions first,
then the full matrix:

```sh
just test <regression-filter>
just features
just test-tiers
just lint
just doc
```

The excluded `benchmarks/afft` project is separate from the published package.
Performance changes require a recorded benchmark comparison; do not commit a
crossover threshold based only on reasoning.
