#![cfg(feature = "std")]

//! Execution must allocate nothing.
//!
//! Plans own every table they need; forward, inverse and derivative walk them
//! in place. This is a hard contract for codec hot paths, so it is checked
//! with a counting global allocator rather than by inspection.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use butterfly_fft::basis::{
    conversion_scratch_elements, interpolate_bytes_scratch, monomial_to_novel_bytes_scratch,
    monomial_to_novel_scratch, novel_to_monomial_bytes_scratch, novel_to_monomial_scratch,
};
use butterfly_fft::transform::TransformPlan;
use fgf::{Gf8B, Gf16};

struct Counting;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator unchanged; the
// counters are the only added effect.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `body` with allocation counting armed, returning the count.
///
/// Single-threaded by construction: this binary holds one test, so the
/// global arming flag cannot race.
fn count_allocations(body: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    body();
    ARMED.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

fn check_field<F: butterfly_fft::kernel::ButterflyKernels>(log_size: usize, row_len: usize) {
    let plan = TransformPlan::<F>::new(1 << log_size).expect("valid plan");
    let mut rows = vec![0x5Au8; plan.size() * row_len];
    let mut derivative = vec![0u8; rows.len()];
    let mut values = vec![F::Elem::default(); plan.size()];
    let mut element_derivative = vec![F::Elem::default(); plan.size()];
    let mut conversion_scratch = vec![F::Elem::default(); conversion_scratch_elements(plan.size())];
    let mut conversion_byte_scratch = vec![0u8; conversion_scratch_elements(plan.size()) * row_len];

    // Warm the process-wide backend detection and any lazy state before
    // arming, so what is measured is the transform itself.
    plan.forward_bytes(&mut rows, row_len).unwrap();
    plan.inverse_bytes(&mut rows, row_len).unwrap();

    let allocations = count_allocations(|| {
        plan.forward_bytes(&mut rows, row_len).unwrap();
        plan.inverse_bytes(&mut rows, row_len).unwrap();
        plan.derivative_into_bytes(&mut derivative, row_len, &rows)
            .unwrap();
        plan.derivative_plus_identity_bytes(&mut derivative, row_len)
            .unwrap();
        plan.forward(&mut values).unwrap();
        plan.inverse(&mut values).unwrap();
        plan.derivative_into(&mut element_derivative, &values)
            .unwrap();
        novel_to_monomial_scratch(&mut values, &plan, &mut conversion_scratch).unwrap();
        monomial_to_novel_scratch(&mut values, &plan, &mut conversion_scratch).unwrap();
        novel_to_monomial_bytes_scratch(&mut rows, row_len, &plan, &mut conversion_byte_scratch)
            .unwrap();
        monomial_to_novel_bytes_scratch(&mut rows, row_len, &plan, &mut conversion_byte_scratch)
            .unwrap();
    });

    assert_eq!(
        allocations,
        0,
        "{} log_size {log_size} allocated during execution",
        F::NAME
    );
}

fn check_ntt<F: fgf::kernel::FieldKernels>(size: usize, lanes: usize) {
    let plan = butterfly_fft::ntt::NttPlan::<F>::new(size).expect("valid plan");
    let row_len = lanes * F::BYTES;
    let mut rows = vec![0x5Au8; size * row_len];
    let mut scratch = plan.scratch(row_len).expect("scratch");
    // Warm backend resolution and the plan's lazy state.
    plan.forward_bytes_scratch(&mut rows, row_len, &mut scratch)
        .expect("warm forward");

    let allocations = count_allocations(|| {
        plan.forward_bytes_scratch(&mut rows, row_len, &mut scratch)
            .expect("forward");
        plan.inverse_bytes_scratch(&mut rows, row_len, &mut scratch)
            .expect("inverse");
    });

    assert_eq!(
        allocations,
        0,
        "{} NTT size {size} lanes {lanes} allocated during execution",
        F::NAME
    );
}

fn check_restricted<F: butterfly_fft::kernel::ButterflyKernels>(size: usize, row_len: usize) {
    let plan = TransformPlan::<F>::new(size).expect("valid plan");
    let active = size / 2 + 1;
    let mut coefficients = vec![0x5au8; size * row_len];
    coefficients[active * row_len..].fill(0);
    let mut evaluations = coefficients.clone();
    plan.forward_bytes(&mut evaluations, row_len).unwrap();
    let mut rows = coefficients.clone();
    let mut half_rows = coefficients[..size / 2 * row_len].to_vec();
    let mut inverse_scratch =
        vec![0u8; plan.inverse_bytes_truncated_scratch_rows(active).unwrap() * row_len];
    let mut conversion_scratch = vec![0u8; conversion_scratch_elements(size) * row_len];
    let selected = [0, size / 2, size - 1];

    let allocations = count_allocations(|| {
        plan.forward_bytes_selected(&mut rows, row_len, &selected)
            .unwrap();
        rows.copy_from_slice(&coefficients);
        plan.forward_bytes_range(&mut rows, row_len, 1..size - 1)
            .unwrap();
        rows.copy_from_slice(&coefficients);
        plan.forward_bytes_truncated_range(&mut rows, row_len, active, 1..size)
            .unwrap();
        plan.forward_bytes_high_coset_range(&mut half_rows, row_len, 0..size / 2)
            .unwrap();
        rows.copy_from_slice(&evaluations);
        plan.inverse_bytes_truncated_scratch(&mut rows, row_len, active, &mut inverse_scratch)
            .unwrap();
        rows.copy_from_slice(&evaluations);
        interpolate_bytes_scratch(&mut rows, row_len, &plan, &mut conversion_scratch).unwrap();
    });
    assert_eq!(allocations, 0, "{} restricted execution allocated", F::NAME);
}

/// One test per binary: the global arming flag is not thread-safe against a
/// second concurrently running test, so every check runs here in order.
#[test]
fn execution_allocates_nothing() {
    check_restricted::<Gf16>(8, 6);
    check_restricted::<Gf8B>(8, 33);
    // Under miri the magnitudes shrink — the contract is the allocation
    // count, not the transform size, and every row-geometry class
    // (multi-element rows, partial-element tails, huge single rows) stays
    // represented.
    if cfg!(miri) {
        check_field::<Gf16>(4, 64);
        check_field::<Gf16>(3, 1_024);
        check_field::<Gf16>(1, 2);
        check_field::<Gf8B>(4, 33);
        check_field::<Gf8B>(3, 4_096);
        check_field::<Gf8B>(0, 1);
        check_ntt::<fgf::Goldilocks>(8, 1);
        check_ntt::<fgf::Goldilocks>(8, 4);
        check_ntt::<fgf::QuadMersenne31>(8, 2);
    } else {
        check_field::<Gf16>(10, 64);
        check_field::<Gf16>(3, 1_024);
        check_field::<Gf16>(1, 2);
        check_field::<Gf8B>(8, 33);
        check_field::<Gf8B>(3, 65_536);
        check_field::<Gf8B>(0, 1);
        check_ntt::<fgf::Goldilocks>(64, 1);
        check_ntt::<fgf::Goldilocks>(64, 16);
        check_ntt::<fgf::QuadMersenne31>(64, 4);
    }
}
