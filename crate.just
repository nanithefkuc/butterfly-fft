# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to butterfly-fft belong in this file and are documented in `AGENTS.md`.

CRATE := 'butterfly-fft'
MSRV := '1.93'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := if arch() == 'aarch64' { 'neon_aes neon scalar' } else if arch() == 'x86_64' { 'v3_gfni_crypto v3 v2 scalar' } else if arch() == 'x86' { 'v3_gfni_crypto v3 v2 scalar' } else { 'scalar' }

# This crate owns no library unsafe; no Miri target is listed.
MIRI := ''

# Coverage exclusions for code that cannot execute on a CI host. Empty means
# every line of the crate counts.
COV_IGNORE := ''
