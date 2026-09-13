# Changelog

All notable changes to butterfly-fft are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed

- Removed the `rs` feature and RS-specific encoding, locator, recovery, and
  targeted-solve helpers. Systematic Reed-Solomon codec algorithms now live in
  `srs`; this crate exposes only codec-neutral transform operations.


### Added

- `ntt` module: a second transform family over the same object, a radix-two
  Cooley-Tukey multiplicative (number-theoretic) transform. `NttPlan`
  precomputes the root of unity, bit-reversal permutation, both twiddle
  directions, and the `n^-1` scale, so `forward_bytes`/`inverse_bytes` run
  over packed byte rows with no allocation and no coefficient preparation.
  Goldilocks and `QuadMersenne31` admit sizes to `2^20`; base `Mersenne31`
  admits only size 2 and the binary fields only the identity, all decided by
  the `size | |F| - 1` rule rather than a field special case.
- `TransformPlan::vanishing_polynomial` returns the dense monomial
  coefficients of the domain vanishing polynomial `G(X)` for both subspace and
  affine-coset domains, and `TransformPlan::shift` exposes the coset shift.
- `basis::inverse_interpolate_bytes` composes inverse transform with
  novel-to-monomial conversion for allocation-free received-word
  interpolation.
- `TransformPlan::derivative_plus_identity_bytes` and the matching
  `ShiftedPlan` method compute `c + D(c)` in place for workflows that evaluate
  only at roots of `c`.

### Changed

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
