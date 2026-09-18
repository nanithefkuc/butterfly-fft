//! Tuning-only execution controls for benchmark crossovers.
//!
//! Each function bypasses production crossover selection to drive one exact
//! schedule directly. Execution arguments are destination, row geometry,
//! plan, then sources or scratch. Reachable only through the `internals`
//! facade; nothing here is a compatibility promise.

use crate::error::{NttError, TransformError};
use crate::kernel::ButterflyKernels;
use crate::ntt::{NttPlan, NttScratch};
use crate::transform::TransformPlan;
use fgf::kernel::FieldKernels;

/// The plan's twiddle and derivative tables.
///
/// This unstable inspection API carries no compatibility guarantee.
// Reachable only through the `internals` facade.
#[allow(dead_code)]
#[must_use]
pub fn plan_table<F: ButterflyKernels>(
    plan: &TransformPlan<F>,
) -> &crate::transform::factors::FactorTable<F> {
    plan.table()
}

/// Tuning-only exact derivative using the source-oriented lowbit sweep.
///
/// This bypasses production crossover selection for benchmark controls. It
/// has the same output and geometry contract as
/// [`TransformPlan::derivative_into_bytes`] and allocates nothing.
///
/// # Errors
/// As [`TransformPlan::derivative_into_bytes`].
///
/// # Panics
/// As [`TransformPlan::derivative_into_bytes`].
// Reachable only through the `internals` facade.
#[allow(dead_code)]
pub fn derivative_into_bytes_sweep<F: ButterflyKernels>(
    derivative: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    coefficients: &[u8],
) -> Result<(), TransformError> {
    plan.derivative_into_bytes_sweep(derivative, row_len, coefficients)
}

/// Tuning-only exact derivative using one multi-source gather per output row.
///
/// This exposes an alternative execution schedule for benchmark crossover
/// measurements. It has the same output and geometry contract as
/// [`TransformPlan::derivative_into_bytes`] and allocates nothing.
///
/// # Errors
/// As [`TransformPlan::derivative_into_bytes`].
///
/// # Panics
/// As [`TransformPlan::derivative_into_bytes`].
// Reachable only through the `internals` facade.
#[allow(dead_code)]
pub fn derivative_into_bytes_gather<F: ButterflyKernels>(
    derivative: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    coefficients: &[u8],
) -> Result<(), TransformError> {
    plan.derivative_into_bytes_gather(derivative, row_len, coefficients)
}

/// Tuning-only exact derivative initialized from first contributions.
///
/// This exposes an alternative execution schedule for benchmark crossover
/// measurements. It has the same output and geometry contract as
/// [`TransformPlan::derivative_into_bytes`] and allocates nothing.
///
/// # Errors
/// As [`TransformPlan::derivative_into_bytes`].
///
/// # Panics
/// As [`TransformPlan::derivative_into_bytes`].
// Reachable only through the `internals` facade.
#[allow(dead_code)]
pub fn derivative_into_bytes_overwrite<F: ButterflyKernels>(
    derivative: &mut [u8],
    row_len: usize,
    plan: &TransformPlan<F>,
    coefficients: &[u8],
) -> Result<(), TransformError> {
    plan.derivative_into_bytes_overwrite(derivative, row_len, coefficients)
}

/// Tuning-only forward NTT forced through the fused register butterflies.
///
/// This bypasses production schedule selection for benchmark controls. It
/// has the same output and geometry contract as
/// [`NttPlan::forward_bytes_scratch`] and allocates nothing.
///
/// # Errors
/// As [`NttPlan::forward_bytes_scratch`].
// Reachable only through the `internals` facade.
#[allow(dead_code)]
pub fn ntt_forward_fused<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    plan: &NttPlan<F>,
    scratch: &mut NttScratch,
) -> Result<(), NttError> {
    plan.forward_fused(rows, row_len, scratch)
}

/// Tuning-only forward NTT forced through the packed op-call butterflies.
///
/// This exposes the alternative execution schedule for benchmark crossover
/// measurements. It has the same output and geometry contract as
/// [`NttPlan::forward_bytes_scratch`] and allocates nothing.
///
/// # Errors
/// As [`NttPlan::forward_bytes_scratch`].
// Reachable only through the `internals` facade.
#[allow(dead_code)]
pub fn ntt_forward_packed<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    plan: &NttPlan<F>,
    scratch: &mut NttScratch,
) -> Result<(), NttError> {
    plan.forward_packed(rows, row_len, scratch)
}

/// Tuning-only forward NTT forced through the batched region-pass
/// butterflies.
///
/// This exposes the alternative execution schedule for benchmark crossover
/// measurements. It has the same output and geometry contract as
/// [`NttPlan::forward_bytes_scratch`] and allocates nothing, except that
/// the scratch must carry batch regions the plan's widest stage fills.
///
/// # Errors
/// As [`NttPlan::forward_bytes_scratch`].
// Reachable only through the `internals` facade.
#[allow(dead_code)]
pub fn ntt_forward_batched<F: FieldKernels>(
    rows: &mut [u8],
    row_len: usize,
    plan: &NttPlan<F>,
    scratch: &mut NttScratch,
) -> Result<(), NttError> {
    plan.forward_batched(rows, row_len, scratch)
}
