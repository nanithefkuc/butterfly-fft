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
