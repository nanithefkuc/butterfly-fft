# Changelog

All notable changes to butterfly-fft are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).


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
  basis-conversion, Reed-Solomon, dispatch, and allocation behavior.
