//! Portable scalar butterfly kernels over any binary field.
//!
//! These are the reference implementation: every SIMD backend is
//! differentially tested against them, and vector kernels hand them their
//! sub-lane tails. They work element by element through the field's stable
//! byte encoding, so they are correct for every [`fgf::field::Field`].

use fgf::field::{Elem, Field};

/// Fused forward butterfly: `low' = low ⊕ c·high`, `high' = low' ⊕ high`.
///
/// Reads each half once and writes each once, in place. `low` and `high`
/// must have equal length, a whole number of elements.
pub(crate) fn fused_forward<F: Field>(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % F::BYTES, 0);
    for (l, h) in low
        .chunks_exact_mut(F::BYTES)
        .zip(high.chunks_exact_mut(F::BYTES))
    {
        let lo = F::read(l);
        let hi = F::read(h);
        let new_low = lo.add(coefficient.mul(hi));
        let new_high = hi.add(new_low);
        F::write(l, new_low);
        F::write(h, new_high);
    }
}

/// Fused inverse butterfly: `high' = high ⊕ low`, `low' = low ⊕ c·high'`.
///
/// Undoes [`fused_forward`] with the same coefficient, in place.
pub(crate) fn fused_inverse<F: Field>(low: &mut [u8], high: &mut [u8], coefficient: F::Elem) {
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % F::BYTES, 0);
    for (l, h) in low
        .chunks_exact_mut(F::BYTES)
        .zip(high.chunks_exact_mut(F::BYTES))
    {
        let lo = F::read(l);
        let hi = F::read(h);
        let new_high = hi.add(lo);
        let new_low = lo.add(coefficient.mul(new_high));
        F::write(l, new_low);
        F::write(h, new_high);
    }
}
