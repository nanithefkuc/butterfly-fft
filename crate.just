# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to butterfly-fft belong in this file and are documented in `AGENTS.md`.

CRATE := 'butterfly-fft'
MSRV := '1.93'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := 'v3_gfni_crypto v3 v2 scalar'

# `cargo miri test` argument sets, one per line. Empty when the crate declares
# no unsafe surface, in which case `just unsafe-check` reports and skips.
MIRI := '--no-default-features'

# Coverage exclusions for code that cannot execute on a CI host. Empty means
# every line of the crate counts.
# GFNI butterfly kernels: only GFNI-capable hosts execute a line of
# kernel/x86/gfni.rs, and the hosted CI fleet mixes Xeon generations, so
# counting it made the gate depend on which runner the job landed on (the
# same runner-dependence fgf documents for its GFNI files). The
# direct-kernel differential tests cover the file wherever the host can
# summon the tier; everywhere else they skip with a printed notice.
COV_IGNORE := 'kernel/x86/gfni\.rs$'
