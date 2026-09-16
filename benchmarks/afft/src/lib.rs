//! Raw-transform adapters used only by the standalone AFFT benchmark harness.

use std::ffi::{CStr, c_char, c_void};
use std::sync::LazyLock;

unsafe extern "C" {
    fn butterfly_fft_leopard_init() -> bool;
    fn butterfly_fft_leopard_backend() -> *const c_char;
    fn butterfly_fft_leopard_forward(rows: *mut *mut c_void, points: u32, row_len: u64);
    fn butterfly_fft_leopard_inverse(rows: *mut *mut c_void, points: u32, row_len: u64);
    fn butterfly_fft_leopard_derivative(rows: *mut *mut c_void, points: u32, row_len: u64);
    fn butterfly_fft_leopard_derivative_exact(
        output: *mut *mut c_void,
        coefficients: *const *mut c_void,
        points: u32,
        row_len: u64,
    );

    fn butterfly_fft_nanors_init();
    fn butterfly_fft_nanors_backend() -> *const c_char;
    fn butterfly_fft_nanors_forward(rows: *mut u8, log_points: u32, row_len: u32);
    fn butterfly_fft_nanors_inverse(rows: *mut u8, log_points: u32, row_len: u32);

    fn butterfly_fft_fastecc_new(points: usize, lanes: usize) -> *mut FastEccNtt;
    fn butterfly_fft_fastecc_fill(state: *mut FastEccNtt, seed: u32);
    fn butterfly_fft_fastecc_forward(state: *mut FastEccNtt);
    fn butterfly_fft_fastecc_inverse(state: *mut FastEccNtt);
    fn butterfly_fft_fastecc_data(state: *const FastEccNtt) -> *const u32;
    fn butterfly_fft_fastecc_free(state: *mut FastEccNtt);
}

static LEOPARD_INITIALIZED: LazyLock<()> = LazyLock::new(|| {
    // SAFETY: initialization takes no pointers and is serialized by LazyLock.
    assert!(
        unsafe { butterfly_fft_leopard_init() },
        "Leopard FF16 self-test failed"
    );
});
static NANORS_INITIALIZED: LazyLock<()> = LazyLock::new(|| {
    // SAFETY: initialization takes no pointers and is serialized by LazyLock.
    unsafe { butterfly_fft_nanors_init() };
});

fn initialize_leopard() {
    LazyLock::force(&LEOPARD_INITIALIZED);
}

fn initialize_nanors() {
    LazyLock::force(&NANORS_INITIALIZED);
}

fn backend_name(pointer: *const c_char) -> &'static str {
    assert!(
        !pointer.is_null(),
        "native adapter returned a null backend name"
    );
    // SAFETY: both adapters return process-static NUL-terminated string literals.
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .expect("native backend name must be UTF-8")
}

/// Selected catid/leopard FF16 backend.
#[must_use]
pub fn leopard_backend() -> &'static str {
    initialize_leopard();
    // SAFETY: initialization completed and the function returns a static string.
    backend_name(unsafe { butterfly_fft_leopard_backend() })
}

/// Flat payload plus the row-pointer table required by catid/leopard.
pub struct LeopardBuffer {
    bytes: Vec<u8>,
    row_len: usize,
    pointers: Vec<*mut c_void>,
}

impl LeopardBuffer {
    /// Construct a reusable pointer view over `points` equal-sized rows.
    #[must_use]
    pub fn new(bytes: Vec<u8>, points: usize) -> Self {
        assert!(points.is_power_of_two() && points <= 65_536);
        assert_eq!(bytes.len() % points, 0);
        let row_len = bytes.len() / points;
        assert!(row_len > 0 && row_len.is_multiple_of(64));

        let base = bytes.as_ptr().cast_mut();
        let pointers = (0..points)
            .map(|index| {
                // SAFETY: each offset names the beginning of a row in `bytes`.
                unsafe { base.add(index * row_len) }.cast::<c_void>()
            })
            .collect();
        Self {
            bytes,
            row_len,
            pointers,
        }
    }

    /// Payload bytes in point-major row order.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Apply Leopard's full raw FF16 forward transform.
    pub fn forward(&mut self) {
        initialize_leopard();
        // SAFETY: the pointer table has one valid row pointer per point and all
        // rows have Leopard's required 64-byte-multiple length.
        unsafe {
            butterfly_fft_leopard_forward(
                self.pointers.as_mut_ptr(),
                self.pointers.len() as u32,
                self.row_len as u64,
            )
        };
    }

    /// Apply Leopard's full raw FF16 inverse transform.
    pub fn inverse(&mut self) {
        initialize_leopard();
        // SAFETY: the same invariants as `forward` hold.
        unsafe {
            butterfly_fft_leopard_inverse(
                self.pointers.as_mut_ptr(),
                self.pointers.len() as u32,
                self.row_len as u64,
            )
        };
    }

    /// Add the formal derivative to the coefficient workspace in place.
    pub fn derivative_plus_identity(&mut self) {
        initialize_leopard();
        // SAFETY: the same invariants as `forward` hold.
        unsafe {
            butterfly_fft_leopard_derivative(
                self.pointers.as_mut_ptr(),
                self.pointers.len() as u32,
                self.row_len as u64,
            )
        };
    }

    /// Compute the exact formal derivative out of place.
    pub fn derivative_exact(&mut self, coefficients: &Self) {
        assert_eq!(self.row_len, coefficients.row_len);
        assert_eq!(self.pointers.len(), coefficients.pointers.len());
        initialize_leopard();
        // SAFETY: both pointer tables contain the same number of disjoint,
        // equal-length rows, and `self` and `coefficients` are distinct borrows.
        unsafe {
            butterfly_fft_leopard_derivative_exact(
                self.pointers.as_mut_ptr(),
                coefficients.pointers.as_ptr(),
                self.pointers.len() as u32,
                self.row_len as u64,
            )
        };
    }
}

impl Clone for LeopardBuffer {
    fn clone(&self) -> Self {
        Self::new(self.bytes.clone(), self.pointers.len())
    }
}

/// Selected nanors FF16 backend.
#[must_use]
pub fn nanors_backend() -> &'static str {
    initialize_nanors();
    // SAFETY: initialization completed and the function returns a static string.
    backend_name(unsafe { butterfly_fft_nanors_backend() })
}

/// Flat point-major payload accepted by nanors' raw FF16 walkers.
#[derive(Clone)]
pub struct NanorsBuffer {
    bytes: Vec<u8>,
    points: usize,
    row_len: usize,
}

impl NanorsBuffer {
    /// Validate and retain one raw transform payload.
    #[must_use]
    pub fn new(bytes: Vec<u8>, points: usize) -> Self {
        assert!(points.is_power_of_two() && points <= 65_536);
        assert_eq!(bytes.len() % points, 0);
        let row_len = bytes.len() / points;
        assert!(row_len > 0 && row_len.is_multiple_of(2));
        assert!(u32::try_from(row_len).is_ok());
        Self {
            bytes,
            points,
            row_len,
        }
    }

    /// Payload bytes in point-major row order.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Apply nanors' full raw FF16 forward transform.
    pub fn forward(&mut self) {
        initialize_nanors();
        // SAFETY: construction established a writable point-major array of
        // complete u16 lanes.
        unsafe {
            butterfly_fft_nanors_forward(
                self.bytes.as_mut_ptr(),
                self.points.trailing_zeros(),
                self.row_len as u32,
            )
        };
    }

    /// Apply nanors' full raw FF16 inverse transform.
    pub fn inverse(&mut self) {
        initialize_nanors();
        // SAFETY: the same invariants as `forward` hold.
        unsafe {
            butterfly_fft_nanors_inverse(
                self.bytes.as_mut_ptr(),
                self.points.trailing_zeros(),
                self.row_len as u32,
            )
        };
    }
}

// Opaque C++ state; only ever handled by pointer across the FFI.
enum FastEccNtt {}

/// FastECC's MFA NTT over GF(0xFFF00001): `points` blocks of `lanes`
/// `u32` field elements each, transformed in place.
pub struct FastEccNttBuffer {
    raw: *mut FastEccNtt,
    points: usize,
    lanes: usize,
}

impl FastEccNttBuffer {
    /// Allocate and fill with deterministic field elements.
    #[must_use]
    pub fn new(points: usize, lanes: usize, seed: u32) -> Self {
        // SAFETY: the sizes match the allocation contract; the fill runs
        // before any other pointer escapes.
        let raw = unsafe { butterfly_fft_fastecc_new(points, lanes) };
        assert!(!raw.is_null(), "FastECC allocation failed");
        // SAFETY: `raw` is a live allocation owned by this buffer.
        unsafe { butterfly_fft_fastecc_fill(raw, seed) };
        Self { raw, points, lanes }
    }

    /// Number of transform points (blocks).
    #[must_use]
    pub fn points(&self) -> usize {
        self.points
    }

    /// Independent transform lanes packed into each block.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// The flat `points × lanes` element buffer, row-major.
    #[must_use]
    pub fn data(&self) -> &[u32] {
        // SAFETY: `data` returns a pointer to the live `points × lanes`
        // allocation owned by this buffer, unchanged for its lifetime.
        unsafe {
            core::slice::from_raw_parts(
                butterfly_fft_fastecc_data(self.raw),
                self.points * self.lanes,
            )
        }
    }

    /// Forward MFA NTT, in place.
    pub fn forward(&mut self) {
        // SAFETY: `raw` is a live allocation of this geometry.
        unsafe { butterfly_fft_fastecc_forward(self.raw) };
    }

    /// Inverse MFA NTT, in place and unnormalized: the composition with
    /// `forward` multiplies every lane by `points`.
    pub fn inverse(&mut self) {
        // SAFETY: the same invariants as `forward` hold.
        unsafe { butterfly_fft_fastecc_inverse(self.raw) };
    }
}

impl Drop for FastEccNttBuffer {
    fn drop(&mut self) {
        // SAFETY: `raw` is freed exactly once, here.
        unsafe { butterfly_fft_fastecc_free(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(len: usize) -> Vec<u8> {
        (0..len)
            .map(|index| (index as u8).wrapping_mul(73).wrapping_add(19))
            .collect()
    }

    #[test]
    fn leopard_raw_round_trip() {
        let original = input(32 * 64);
        let mut rows = LeopardBuffer::new(original.clone(), 32);
        rows.forward();
        rows.inverse();
        assert_eq!(rows.as_bytes(), original);

        let mut zero = LeopardBuffer::new(vec![0; 32 * 64], 32);
        zero.derivative_plus_identity();
        assert!(zero.as_bytes().iter().all(|&byte| byte == 0));

        let coefficients = input(32 * 64);
        let mut expected = vec![0u8; coefficients.len()];
        for source in 1usize..32 {
            let mut bits = source;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let destination = source ^ (1 << bit);
                for offset in 0..64 {
                    expected[destination * 64 + offset] ^= coefficients[source * 64 + offset];
                }
                bits &= bits - 1;
            }
        }
        let input_rows = LeopardBuffer::new(coefficients, 32);
        let mut exact = LeopardBuffer::new(vec![0xA5; 32 * 64], 32);
        exact.derivative_exact(&input_rows);
        assert_eq!(exact.as_bytes(), expected);
        assert!(!leopard_backend().is_empty());
    }

    #[test]
    fn fastecc_ntt_round_trip_is_scaled_identity() {
        let points = 64;
        let mut buffer = FastEccNttBuffer::new(points, 2, 0x243F_6A88);
        let original = buffer.data().to_vec();
        buffer.forward();
        buffer.inverse();
        // FastECC's inverse leaves the 1/N normalization to the caller, so
        // the round trip multiplies every lane by `points`.
        let modulus: u64 = 0xFFF00001;
        for (&got, &input) in buffer.data().iter().zip(&original) {
            let expected = (u64::from(input) * points as u64) % modulus;
            assert_eq!(u64::from(got), expected);
        }
    }
}
