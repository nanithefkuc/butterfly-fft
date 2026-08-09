#![cfg(feature = "std")]

//! Execution must allocate nothing.
//!
//! Plans own every table they need; forward, inverse and derivative walk them
//! in place. This is a hard contract for codec hot paths, so it is checked
//! with a counting global allocator rather than by inspection.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use cafft::basis::{
    conversion_scratch_elements, monomial_to_novel_bytes, monomial_to_novel_with_scratch,
    novel_to_monomial_bytes, novel_to_monomial_with_scratch,
};
use cafft::core::transform::TransformPlan;
use fgf::{Gf8, Gf16};

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

fn check_field<F: cafft::core::kernel::ButterflyKernels>(log_size: usize, row_len: usize) {
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
        plan.derivative_bytes(&rows, row_len, &mut derivative)
            .unwrap();
        plan.forward(&mut values).unwrap();
        plan.inverse(&mut values).unwrap();
        plan.derivative(&values, &mut element_derivative).unwrap();
        novel_to_monomial_with_scratch(&mut values, &plan, &mut conversion_scratch).unwrap();
        monomial_to_novel_with_scratch(&mut values, &plan, &mut conversion_scratch).unwrap();
        novel_to_monomial_bytes(&mut rows, row_len, &plan, &mut conversion_byte_scratch).unwrap();
        monomial_to_novel_bytes(&mut rows, row_len, &plan, &mut conversion_byte_scratch).unwrap();
    });

    assert_eq!(
        allocations,
        0,
        "{} log_size {log_size} allocated during execution",
        F::NAME
    );
}

/// One test per binary: the global arming flag is not thread-safe against a
/// second concurrently running test, so every check runs here in order.
#[test]
fn execution_allocates_nothing() {
    check_field::<Gf16>(10, 64);
    check_field::<Gf16>(1, 2);
    check_field::<Gf8>(8, 33);
    check_field::<Gf8>(0, 1);
    #[cfg(feature = "rs")]
    check_rs_helpers();
}

/// The `rs` helpers make the same promise once their scratch is sized: the
/// locator recompute, the Forney recovery and the strip encode are steady
/// state on a codec hot path.
#[cfg(feature = "rs")]
fn check_rs_helpers() {
    use cafft::rs::{ErasureLocator, LocatorScratch, RecoveryScratch, StripEncoder, recover_rows};

    let (data_points, repair_points, row_len) = (9usize, 7usize, 8usize);
    let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).expect("geometry");
    let plan = encoder.evaluation_plan();
    let size = plan.size();

    let data = vec![0x5Au8; data_points * row_len];
    let mut repairs = vec![0u8; repair_points * row_len];
    let mut encode_scratch = encoder.scratch();

    let known: Vec<bool> = (0..size).map(|point| point >= 3).collect();
    let missing: Vec<usize> = (0..3).collect();
    let mut locator = ErasureLocator::<Gf16>::for_domain(size);
    let mut locator_scratch = LocatorScratch::for_domain(size);
    let mut recovery = RecoveryScratch::for_geometry(size, row_len);
    let received = vec![0u8; size * row_len];
    let mut recovered = vec![0u8; missing.len() * row_len];

    // Warm lazy tables, plan caches and backend detection.
    encoder
        .encode(&data, &mut repairs, &mut encode_scratch)
        .unwrap();
    locator
        .recompute(plan, &known, &mut locator_scratch)
        .unwrap();

    let allocations = count_allocations(|| {
        encoder
            .encode(&data, &mut repairs, &mut encode_scratch)
            .unwrap();
        locator
            .recompute(plan, &known, &mut locator_scratch)
            .unwrap();
        recover_rows(
            plan,
            &locator,
            &received,
            row_len,
            &missing,
            &mut recovery,
            &mut recovered,
        )
        .unwrap();
    });
    assert_eq!(allocations, 0, "rs helpers allocated during execution");
}
