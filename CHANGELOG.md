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
- `TransformPlan::derivative_bytes` computes the formal derivative in a
  single ascending sweep over row blocks, coupling each block with its
  pristine source rows in one pass; plans whose derivative factors are all
  zero or one (the Cantor basis) run the sweep on plain XORs with no field
  multiplies. Measured up to 51% faster on bit-basis plans and within 8% of
  leopard's hand-tuned decoder loop on small-row Cantor domains.
- Transform byte walkers execute dimensions of three or fewer as explicit
  fused base cases instead of per-level recursion, cutting per-call kernel
  setup roughly fourfold at the bottom of the tree. Measured up to 34%
  faster on mid-size payloads.
- Fused butterflies gained a unit-coefficient fast path alongside the
  zero-coefficient path: `c = 1` runs as two XOR passes with no field
  multiply.
