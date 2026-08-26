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

- `TransformPlan::vanishing_polynomial` returns the dense monomial
  coefficients of the domain vanishing polynomial `G(X)` for both subspace and
  affine-coset domains, and `TransformPlan::shift` exposes the coset shift.
- `basis::inverse_interpolate_bytes` composes inverse transform with
  novel-to-monomial conversion for allocation-free received-word
  interpolation.
### Changed

- Renamed the crate from `cafft` to `butterfly-fft`, including its package,
  library identifier, repository URL, and dependent feature paths.
- Replaced the former `fff` field dependency with `fgf`, preserving transform,
  basis-conversion, dispatch, and allocation behavior.
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
