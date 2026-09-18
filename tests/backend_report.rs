use butterfly_fft::kernel::{backend, backend_for};
use fgf::{Gf8B, Gf16, Goldilocks, QuadMersenne31};

fn main() {
    println!(
        "backend_report requested={} additive={} gf8={} gf16={} goldilocks={} quad_mersenne31={}",
        std::env::var("SIMD_BACKEND").unwrap_or_else(|_| "auto".into()),
        backend(),
        backend_for::<Gf8B>(),
        backend_for::<Gf16>(),
        fgf::kernel::backend_for::<Goldilocks>(),
        fgf::kernel::backend_for::<QuadMersenne31>(),
    );
}
