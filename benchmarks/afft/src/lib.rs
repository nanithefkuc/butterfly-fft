//! Raw-transform adapters used only by the standalone AFFT benchmark harness.

use std::ffi::{CStr, c_char, c_void};
use std::sync::LazyLock;

unsafe extern "C" {
    fn cafft_leopard_init() -> bool;
    fn cafft_leopard_backend() -> *const c_char;
    fn cafft_leopard_forward(rows: *mut *mut c_void, points: u32, row_len: u64);
    fn cafft_leopard_inverse(rows: *mut *mut c_void, points: u32, row_len: u64);
    fn cafft_leopard_derivative(rows: *mut *mut c_void, points: u32, row_len: u64);

    fn cafft_nanors_init();
    fn cafft_nanors_backend() -> *const c_char;
    fn cafft_nanors_forward(rows: *mut u8, log_points: u32, row_len: u32);
    fn cafft_nanors_inverse(rows: *mut u8, log_points: u32, row_len: u32);
}

static LEOPARD_INITIALIZED: LazyLock<()> = LazyLock::new(|| {
    // SAFETY: initialization takes no pointers and is serialized by LazyLock.
    assert!(
        unsafe { cafft_leopard_init() },
        "Leopard FF16 self-test failed"
    );
});
static NANORS_INITIALIZED: LazyLock<()> = LazyLock::new(|| {
    // SAFETY: initialization takes no pointers and is serialized by LazyLock.
    unsafe { cafft_nanors_init() };
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
    backend_name(unsafe { cafft_leopard_backend() })
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
            cafft_leopard_forward(
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
            cafft_leopard_inverse(
                self.pointers.as_mut_ptr(),
                self.pointers.len() as u32,
                self.row_len as u64,
            )
        };
    }

    /// Apply the formal-derivative loop used by Leopard's FF16 decoder.
    pub fn derivative(&mut self) {
        initialize_leopard();
        // SAFETY: the same invariants as `forward` hold.
        unsafe {
            cafft_leopard_derivative(
                self.pointers.as_mut_ptr(),
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
    backend_name(unsafe { cafft_nanors_backend() })
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
            cafft_nanors_forward(
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
            cafft_nanors_inverse(
                self.bytes.as_mut_ptr(),
                self.points.trailing_zeros(),
                self.row_len as u32,
            )
        };
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
        zero.derivative();
        assert!(zero.as_bytes().iter().all(|&byte| byte == 0));
        assert!(!leopard_backend().is_empty());
    }

    #[test]
    fn nanors_raw_round_trip() {
        let original = input(32 * 64);
        let mut rows = NanorsBuffer::new(original.clone(), 32);
        rows.forward();
        rows.inverse();
        assert_eq!(rows.as_bytes(), original);
        assert!(!nanors_backend().is_empty());
    }
}
