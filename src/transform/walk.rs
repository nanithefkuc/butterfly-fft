//! Recursive walkers behind the transform execution models.
//!
//! Each byte-row entry is generic over a backend tag `B` chosen once per
//! call by the dispatch macro, so a whole recursion monomorphizes onto one
//! backend instead of dispatching per butterfly.

use ::core::ops::Range;

use fgf::field::Elem;
use fgf::ops::{self, Coeff};

use crate::kernel::{
    ButterflyBackend, ButterflyKernels, fused_forward_backend, fused_inverse_backend,
};

pub(super) fn forward_node<E: Elem>(
    values: &mut [E],
    factors: &[E],
    node: usize,
    dimension: usize,
) {
    let half = values.len() / 2;
    let factor = factors[node];
    for position in 0..half {
        let low = values[position];
        let high = values[half + position];
        let left = low.add(factor.mul(high));
        values[position] = left;
        values[half + position] = left.add(high);
    }
    if dimension > 1 {
        let (left, right) = values.split_at_mut(half);
        forward_node(left, factors, node * 2, dimension - 1);
        forward_node(right, factors, node * 2 + 1, dimension - 1);
    }
}

pub(super) fn inverse_node<E: Elem>(
    values: &mut [E],
    factors: &[E],
    node: usize,
    dimension: usize,
) {
    let half = values.len() / 2;
    if dimension > 1 {
        let (left, right) = values.split_at_mut(half);
        inverse_node(left, factors, node * 2, dimension - 1);
        inverse_node(right, factors, node * 2 + 1, dimension - 1);
    }
    let factor = factors[node];
    for position in 0..half {
        let left = values[position];
        let right = values[half + position];
        let high = left.add(right);
        values[position] = left.add(factor.mul(high));
        values[half + position] = high;
    }
}

fn derivative_into_bytes_add_node<F: ButterflyKernels>(
    derivative: &mut [u8],
    coefficients: &[u8],
    factors: &[Coeff<F>],
    dimension: usize,
) {
    if dimension == 0 {
        return;
    }
    let half = coefficients.len() / 2;
    let (coefficient_low, coefficient_high) = coefficients.split_at(half);
    let (derivative_low, derivative_high) = derivative.split_at_mut(half);
    let factor = &factors[dimension - 1];
    ops::mul_add_with::<F>(derivative_low, factor, coefficient_high);
    derivative_into_bytes_add_node(derivative_low, coefficient_low, factors, dimension - 1);
    derivative_into_bytes_add_node(derivative_high, coefficient_high, factors, dimension - 1);
}

pub(super) fn derivative_into_bytes_overwrite_node<F: ButterflyKernels>(
    derivative: &mut [u8],
    coefficients: &[u8],
    factors: &[Coeff<F>],
    dimension: usize,
) {
    if dimension == 0 {
        derivative.fill(0);
        return;
    }
    let half = coefficients.len() / 2;
    let (coefficient_low, coefficient_high) = coefficients.split_at(half);
    let (derivative_low, derivative_high) = derivative.split_at_mut(half);
    let factor = &factors[dimension - 1];
    ops::mul_into_with::<F>(derivative_low, factor, coefficient_high);
    derivative_into_bytes_add_node(derivative_low, coefficient_low, factors, dimension - 1);
    derivative_into_bytes_overwrite_node(derivative_high, coefficient_high, factors, dimension - 1);
}

/// Deepest dimension handled by the explicit fused base cases instead of
/// recursion: a subtree spans at most `2^BASE_DIMENSION` rows, whose
/// butterflies all touch one cache-resident block.
const BASE_DIMENSION: usize = 3;

/// Fused base case, forward order: the node's butterfly first, then both
/// child subtrees depth-first. All butterflies stay within one small row
/// block, so per-call kernel setup amortizes over the subtree instead of
/// paying one call frame per level.
fn forward_bytes_base<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
) {
    match dimension {
        1 => {
            let (low_half, high_half) = rows.split_at_mut(rows.len() / 2);
            fused_forward_backend::<F, B>(low_half, high_half, factors[node]);
        }
        2 => {
            let half_bytes = rows.len() / 2;
            let quarter_bytes = half_bytes / 2;
            let (low_half, high_half) = rows.split_at_mut(half_bytes);
            fused_forward_backend::<F, B>(low_half, high_half, factors[node]);
            let (low_low, low_high) = low_half.split_at_mut(quarter_bytes);
            let (high_low, high_high) = high_half.split_at_mut(quarter_bytes);
            fused_forward_backend::<F, B>(low_low, low_high, factors[node * 2]);
            fused_forward_backend::<F, B>(high_low, high_high, factors[node * 2 + 1]);
        }
        3 => {
            let half_bytes = rows.len() / 2;
            let quarter_bytes = half_bytes / 2;
            let eighth_bytes = quarter_bytes / 2;
            let (low_half, high_half) = rows.split_at_mut(half_bytes);
            fused_forward_backend::<F, B>(low_half, high_half, factors[node]);
            let (low_low, low_high) = low_half.split_at_mut(quarter_bytes);
            let (high_low, high_high) = high_half.split_at_mut(quarter_bytes);
            fused_forward_backend::<F, B>(low_low, low_high, factors[node * 2]);
            fused_forward_backend::<F, B>(high_low, high_high, factors[node * 2 + 1]);
            // Leaf level: each remaining quarter is one row pair.
            let (r0, r1) = low_low.split_at_mut(eighth_bytes);
            let (r2, r3) = low_high.split_at_mut(eighth_bytes);
            let (r4, r5) = high_low.split_at_mut(eighth_bytes);
            let (r6, r7) = high_high.split_at_mut(eighth_bytes);
            fused_forward_backend::<F, B>(r0, r1, factors[node * 4]);
            fused_forward_backend::<F, B>(r2, r3, factors[node * 4 + 1]);
            fused_forward_backend::<F, B>(r4, r5, factors[node * 4 + 2]);
            fused_forward_backend::<F, B>(r6, r7, factors[node * 4 + 3]);
        }
        _ => {}
    }
}

/// Fused base case, inverse order: both child subtrees first, then the
/// node's butterfly.
fn inverse_bytes_base<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
) {
    match dimension {
        1 => {
            let (low_half, high_half) = rows.split_at_mut(rows.len() / 2);
            fused_inverse_backend::<F, B>(low_half, high_half, factors[node]);
        }
        2 => {
            let half_bytes = rows.len() / 2;
            let quarter_bytes = half_bytes / 2;
            let (low_half, high_half) = rows.split_at_mut(half_bytes);
            let (low_low, low_high) = low_half.split_at_mut(quarter_bytes);
            let (high_low, high_high) = high_half.split_at_mut(quarter_bytes);
            fused_inverse_backend::<F, B>(low_low, low_high, factors[node * 2]);
            fused_inverse_backend::<F, B>(high_low, high_high, factors[node * 2 + 1]);
            fused_inverse_backend::<F, B>(low_half, high_half, factors[node]);
        }
        3 => {
            let half_bytes = rows.len() / 2;
            let quarter_bytes = half_bytes / 2;
            let eighth_bytes = quarter_bytes / 2;
            let (low_half, high_half) = rows.split_at_mut(half_bytes);
            let (low_low, low_high) = low_half.split_at_mut(quarter_bytes);
            let (high_low, high_high) = high_half.split_at_mut(quarter_bytes);
            let (r0, r1) = low_low.split_at_mut(eighth_bytes);
            let (r2, r3) = low_high.split_at_mut(eighth_bytes);
            let (r4, r5) = high_low.split_at_mut(eighth_bytes);
            let (r6, r7) = high_high.split_at_mut(eighth_bytes);
            fused_inverse_backend::<F, B>(r0, r1, factors[node * 4]);
            fused_inverse_backend::<F, B>(r2, r3, factors[node * 4 + 1]);
            fused_inverse_backend::<F, B>(r4, r5, factors[node * 4 + 2]);
            fused_inverse_backend::<F, B>(r6, r7, factors[node * 4 + 3]);
            fused_inverse_backend::<F, B>(low_low, low_high, factors[node * 2]);
            fused_inverse_backend::<F, B>(high_low, high_high, factors[node * 2 + 1]);
            fused_inverse_backend::<F, B>(low_half, high_half, factors[node]);
        }
        _ => {}
    }
}

pub(super) fn forward_bytes_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
) {
    if dimension <= BASE_DIMENSION {
        forward_bytes_base::<F, B>(rows, factors, node, dimension);
        return;
    }
    let half_bytes = rows.len() / 2;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    fused_forward_backend::<F, B>(low_half, high_half, factor);
    forward_bytes_node::<F, B>(low_half, factors, node * 2, dimension - 1);
    forward_bytes_node::<F, B>(high_half, factors, node * 2 + 1, dimension - 1);
}

pub(super) fn inverse_bytes_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
) {
    if dimension <= BASE_DIMENSION {
        inverse_bytes_base::<F, B>(rows, factors, node, dimension);
        return;
    }
    let half_bytes = rows.len() / 2;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    inverse_bytes_node::<F, B>(low_half, factors, node * 2, dimension - 1);
    inverse_bytes_node::<F, B>(high_half, factors, node * 2 + 1, dimension - 1);
    let factor = factors[node];
    fused_inverse_backend::<F, B>(low_half, high_half, factor);
}

pub(super) fn forward_bytes_selected_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    row_offset: usize,
    selected: &[usize],
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / row_len;
    let middle = row_offset + half_rows;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    fused_forward_backend::<F, B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    let split = selected.partition_point(|&index| index < middle);
    if split != 0 {
        forward_bytes_selected_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            row_offset,
            &selected[..split],
        );
    }
    if split != selected.len() {
        forward_bytes_selected_node::<F, B>(
            high_half,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            middle,
            &selected[split..],
        );
    }
}

pub(super) fn forward_bytes_range_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    range_start: usize,
    range_end: usize,
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / row_len;
    let factor = factors[node];
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    fused_forward_backend::<F, B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    if range_start < half_rows {
        forward_bytes_range_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            range_start,
            range_end.min(half_rows),
        );
    }
    if range_end > half_rows {
        forward_bytes_range_node::<F, B>(
            high_half,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            range_start.saturating_sub(half_rows),
            range_end - half_rows,
        );
    }
}

pub(super) fn forward_bytes_truncated_range_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    active: usize,
    range: Range<usize>,
) {
    let half_bytes = rows.len() / 2;
    let half_rows = half_bytes / row_len;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);

    // The active prefix fits in the low half: the high half's coefficients
    // are all zero, so the butterfly degenerates to a copy and only the
    // low half needs recursion.
    if active <= half_rows {
        if dimension <= 1 {
            if range.end > half_rows {
                high_half[..active * row_len].copy_from_slice(&low_half[..active * row_len]);
            }
            return;
        }
        let need_low = range.start < half_rows;
        let need_high = range.end > half_rows;
        if need_high {
            high_half[..active * row_len].copy_from_slice(&low_half[..active * row_len]);
        }
        if need_low {
            forward_bytes_truncated_range_node::<F, B>(
                low_half,
                row_len,
                factors,
                node * 2,
                dimension - 1,
                active,
                range.start..range.end.min(half_rows),
            );
        }
        if need_high {
            forward_bytes_truncated_range_node::<F, B>(
                high_half,
                row_len,
                factors,
                node * 2 + 1,
                dimension - 1,
                active,
                range.start.saturating_sub(half_rows)..range.end - half_rows,
            );
        }
        return;
    }

    let factor = factors[node];
    fused_forward_backend::<F, B>(low_half, high_half, factor);
    if dimension <= 1 {
        return;
    }
    if range.start < half_rows {
        forward_bytes_truncated_range_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            half_rows,
            range.start..range.end.min(half_rows),
        );
    }
    if range.end > half_rows {
        forward_bytes_truncated_range_node::<F, B>(
            high_half,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            half_rows,
            range.start.saturating_sub(half_rows)..range.end - half_rows,
        );
    }
}

pub(super) fn inverse_bytes_truncated_node<F: ButterflyKernels, B: ButterflyBackend<F>>(
    rows: &mut [u8],
    row_len: usize,
    factors: &[F::Elem],
    node: usize,
    dimension: usize,
    active: usize,
    scratch: &mut [u8],
) {
    let row_count = rows.len() / row_len;
    if row_count == 1 {
        return;
    }
    if active == row_count {
        inverse_bytes_node::<F, B>(rows, factors, node, dimension);
        return;
    }
    let half_rows = row_count / 2;
    let half_bytes = half_rows * row_len;
    let (low_half, high_half) = rows.split_at_mut(half_bytes);
    if active <= half_rows {
        inverse_bytes_truncated_node::<F, B>(
            low_half,
            row_len,
            factors,
            node * 2,
            dimension - 1,
            active,
            scratch,
        );
        return;
    }

    let right_active = active - half_rows;
    inverse_bytes_node::<F, B>(low_half, factors, node * 2, dimension - 1);

    // The recovered low half's tail (coefficients right_active..half) is
    // part of the active prefix and generally nonzero: subtract its forward
    // evaluation over the high subtree from the high half's evaluations
    // before recursing, so the recursion sees evaluations of a polynomial
    // truncated to right_active coefficients.
    let tail_start = right_active * row_len;
    {
        let known_tail = &mut scratch[..half_bytes];
        known_tail.fill(0);
        known_tail[tail_start..].copy_from_slice(&low_half[tail_start..]);
        forward_bytes_range_node::<F, B>(
            known_tail,
            row_len,
            factors,
            node * 2 + 1,
            dimension - 1,
            0,
            right_active,
        );
        for (evaluation, contribution) in high_half[..tail_start]
            .iter_mut()
            .zip(&known_tail[..tail_start])
        {
            *evaluation ^= contribution;
        }
    }
    inverse_bytes_truncated_node::<F, B>(
        high_half,
        row_len,
        factors,
        node * 2 + 1,
        dimension - 1,
        right_active,
        scratch,
    );

    let factor = factors[node];
    let active_bytes = right_active * row_len;
    fused_inverse_backend::<F, B>(
        &mut low_half[..active_bytes],
        &mut high_half[..active_bytes],
        factor,
    );
}
