# Changelog

All notable changes to butterfly-fft are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Renamed the crate from `cafft` to `butterfly-fft`, including its package,
  library identifier, repository URL, and dependent feature paths.
- Replaced the former `fff` field dependency with `fgf`, preserving transform,
  basis-conversion, Reed-Solomon, dispatch, and allocation behavior.
