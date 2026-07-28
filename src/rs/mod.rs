//! RS-facing erasure algebra.
//!
//! Erasure-locator evaluation over the transform domain, Forney-style
//! recovery through the novel-basis formal derivative, systematic-locator
//! caches for the small-erasure targeted path, targeted dense solver
//! helpers, and the strip-blocked encode skeleton. Deals in domains, point
//! sets, and byte rows only — receipt bookkeeping, wire layouts, and codec
//! error enums belong to consumers.
//!
//! # The two decode paths
//!
//! A systematic code with `k` data points loses some subset of them. Two
//! algorithms cover the range:
//!
//! - **Locator path** ([`ErasureLocator`] + [`recover_rows`]): cost is a
//!   fixed three domain-sized transforms regardless of how many points are
//!   erased. Right answer when many are.
//! - **Targeted path** ([`SystematicLocators`] + [`generator_row`] +
//!   [`invert_square_into`]): with `t` missing data points, solve the dense
//!   `t × t` system relating them to `t` received repairs. Cost scales with
//!   `t`, not with the domain, so it wins for small `t`. The crossover is a
//!   consumer policy decision; this module supplies both.
//!
//! Both take a plain [`TransformPlan`](crate::core::transform::TransformPlan)
//! and an evaluation-domain point set. Neither knows what a wire index is.
//!
//! # Field support
//!
//! Everything here works in discrete logarithms, so it is available for
//! fields with tabulated logs: extension degree at most 16 (see
//! [`RsField`]). [`StripEncoder`] needs no logarithms and works over every
//! field.

mod encode;
mod forney;
mod locator;
mod solve;
mod tables;

pub use encode::{EncodeScratch, StripEncoder, strip_width};
pub use forney::{RecoveryScratch, recover_rows};
pub use locator::{ErasureLocator, LocatorScratch, SystematicLocators, walsh_hadamard};
pub use solve::{generator_row, inverse_scratch_elements, invert_square_into};
pub use tables::{LogExpTables, RsField};

#[cfg(test)]
mod tests {
    use ::alloc::vec;
    use ::alloc::vec::Vec;

    use fff::field::{Elem, Field};
    use fff::{Gf8, Gf16};

    use super::*;

    fn bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state.to_le_bytes()[5]
            })
            .collect()
    }

    /// End-to-end: encode, drop symbols, recover through the locator path.
    fn locator_round_trip<F: RsField>(
        data_points: usize,
        repair_points: usize,
        row_len: usize,
        lost_data: &[usize],
        lost_repairs: &[usize],
    ) where
        F::Elem: Send + Sync,
    {
        let encoder = StripEncoder::<F>::new(data_points, repair_points, row_len).unwrap();
        let data = bytes(data_points * row_len, 0x51ed_270b_9f4d_1e73);
        let mut repairs = vec![0u8; repair_points * row_len];
        let mut scratch = encoder.scratch();
        encoder.encode(&data, &mut repairs, &mut scratch).unwrap();

        let plan = encoder.evaluation_plan();
        let size = plan.size();
        let transmitted = data_points + repair_points;

        // Assemble the received codeword over the evaluation domain. Points
        // beyond `transmitted` were never sent, so they count as erased.
        let mut received = vec![0u8; size * row_len];
        let mut known = vec![false; size];
        for point in 0..transmitted {
            if point < data_points {
                if lost_data.contains(&point) {
                    continue;
                }
                received[point * row_len..(point + 1) * row_len]
                    .copy_from_slice(&data[point * row_len..(point + 1) * row_len]);
            } else {
                let repair = point - data_points;
                if lost_repairs.contains(&repair) {
                    continue;
                }
                received[point * row_len..(point + 1) * row_len]
                    .copy_from_slice(&repairs[repair * row_len..(repair + 1) * row_len]);
            }
            known[point] = true;
        }

        let locator = ErasureLocator::new(plan, &known).unwrap();
        let missing: Vec<usize> = lost_data.to_vec();
        let mut recovery = RecoveryScratch::for_geometry(size, row_len);
        let mut recovered = vec![0u8; missing.len() * row_len];
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
        for (row, &point) in missing.iter().enumerate() {
            assert_eq!(
                &recovered[row * row_len..(row + 1) * row_len],
                &data[point * row_len..(point + 1) * row_len],
                "data point {point}"
            );
        }
    }

    #[test]
    fn locator_path_recovers_lost_data_symbols() {
        // Every erasure count the code tolerates, over both fields, fused
        // (k a power of two, m <= k) and padded geometries.
        locator_round_trip::<Gf16>(8, 8, 4, &[0, 3, 7], &[]);
        locator_round_trip::<Gf16>(8, 8, 4, &[1, 2], &[0, 5]);
        locator_round_trip::<Gf16>(5, 3, 2, &[0, 2, 4], &[]);
        locator_round_trip::<Gf16>(9, 7, 6, &[1, 4, 8], &[2, 3]);
        locator_round_trip::<Gf8>(4, 4, 1, &[0, 1, 2, 3], &[]);
        locator_round_trip::<Gf8>(5, 3, 3, &[2], &[1]);
        // Losing exactly `m` data symbols is the worst tolerable case.
        locator_round_trip::<Gf16>(9, 7, 2, &[0, 1, 2, 3, 4, 5, 6], &[]);
    }

    /// End-to-end targeted path: solve the dense system relating the missing
    /// data points to the received repairs.
    fn targeted_round_trip<F: RsField>(
        data_points: usize,
        repair_points: usize,
        row_len: usize,
        lost_data: &[usize],
    ) where
        F::Elem: Send + Sync,
    {
        assert!(lost_data.len() <= repair_points);
        let encoder = StripEncoder::<F>::new(data_points, repair_points, row_len).unwrap();
        let data = bytes(data_points * row_len, 0x1d87_2f4e_a3b0_c915);
        let mut repairs = vec![0u8; repair_points * row_len];
        let mut scratch = encoder.scratch();
        encoder.encode(&data, &mut repairs, &mut scratch).unwrap();

        let plan = encoder.evaluation_plan();
        let locators = SystematicLocators::<F>::new();
        let locator = locators.get(plan, data_points).unwrap();
        let missing = lost_data.len();

        // One generator row per repair symbol used, restricted to the
        // missing data columns.
        let mut generator = vec![F::Elem::ZERO; missing * data_points];
        for row in 0..missing {
            generator_row(
                plan,
                &locator,
                data_points + row,
                &mut generator[row * data_points..(row + 1) * data_points],
            );
        }
        let mut system = vec![F::Elem::ZERO; missing * missing];
        for row in 0..missing {
            for (column, &point) in lost_data.iter().enumerate() {
                system[row * missing + column] = generator[row * data_points + point];
            }
        }
        let mut augmented = vec![F::Elem::ZERO; inverse_scratch_elements(missing)];
        let mut inverse = vec![F::Elem::ZERO; missing * missing];
        assert!(invert_square_into(
            &system,
            missing,
            &mut augmented,
            &mut inverse
        ));

        // Residual: repair minus the contribution of the surviving data.
        let mut residuals = repairs[..missing * row_len].to_vec();
        let mut coefficients = vec![F::Elem::ZERO; missing];
        for point in 0..data_points {
            if lost_data.contains(&point) {
                continue;
            }
            for row in 0..missing {
                coefficients[row] = generator[row * data_points + point];
            }
            crate::core::kernel::xor_scaled_bytes_rows::<F>(
                &mut residuals,
                row_len,
                &coefficients,
                &data[point * row_len..(point + 1) * row_len],
            );
        }

        let mut recovered = vec![0u8; missing * row_len];
        for residual in 0..missing {
            for row in 0..missing {
                coefficients[row] = inverse[row * missing + residual];
            }
            crate::core::kernel::xor_scaled_bytes_rows::<F>(
                &mut recovered,
                row_len,
                &coefficients,
                &residuals[residual * row_len..(residual + 1) * row_len],
            );
        }
        for (row, &point) in lost_data.iter().enumerate() {
            assert_eq!(
                &recovered[row * row_len..(row + 1) * row_len],
                &data[point * row_len..(point + 1) * row_len],
                "data point {point}"
            );
        }
    }

    #[test]
    fn targeted_path_recovers_small_erasure_patterns() {
        targeted_round_trip::<Gf16>(8, 8, 4, &[0]);
        targeted_round_trip::<Gf16>(8, 8, 4, &[3, 5]);
        targeted_round_trip::<Gf16>(9, 7, 6, &[0, 4, 8]);
        targeted_round_trip::<Gf16>(17, 15, 2, &[1, 2, 3, 15, 16]);
        targeted_round_trip::<Gf8>(5, 3, 3, &[1, 4]);
    }

    #[test]
    fn both_paths_agree_on_the_same_erasures() {
        // The locator path and the targeted path are independent algorithms;
        // for a pattern both handle they must produce identical bytes.
        let (data_points, repair_points, row_len) = (9usize, 7usize, 4usize);
        let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
        let data = bytes(data_points * row_len, 0x7a3c_ffee_0011_2233);
        let mut repairs = vec![0u8; repair_points * row_len];
        let mut scratch = encoder.scratch();
        encoder.encode(&data, &mut repairs, &mut scratch).unwrap();

        let plan = encoder.evaluation_plan();
        let size = plan.size();
        let lost = [2usize, 6];
        let mut received = vec![0u8; size * row_len];
        let mut known = vec![false; size];
        for point in 0..data_points {
            if lost.contains(&point) {
                continue;
            }
            received[point * row_len..(point + 1) * row_len]
                .copy_from_slice(&data[point * row_len..(point + 1) * row_len]);
            known[point] = true;
        }
        for repair in 0..repair_points {
            let point = data_points + repair;
            received[point * row_len..(point + 1) * row_len]
                .copy_from_slice(&repairs[repair * row_len..(repair + 1) * row_len]);
            known[point] = true;
        }
        let locator = ErasureLocator::new(plan, &known).unwrap();
        let mut recovery = RecoveryScratch::new();
        let mut recovered = vec![0u8; lost.len() * row_len];
        recover_rows(
            plan,
            &locator,
            &received,
            row_len,
            &lost,
            &mut recovery,
            &mut recovered,
        )
        .unwrap();
        for (row, &point) in lost.iter().enumerate() {
            assert_eq!(
                &recovered[row * row_len..(row + 1) * row_len],
                &data[point * row_len..(point + 1) * row_len]
            );
        }
        targeted_round_trip::<Gf16>(data_points, repair_points, row_len, &lost);
    }

    #[test]
    fn recovery_is_exact_not_merely_nonzero() {
        // A codeword of all-zero data must recover to exact zeros, and a
        // single nonzero data element must recover to exactly that element.
        let (data_points, repair_points, row_len) = (4usize, 4usize, 2usize);
        let encoder = StripEncoder::<Gf16>::new(data_points, repair_points, row_len).unwrap();
        let plan = encoder.evaluation_plan();
        let mut data = vec![0u8; data_points * row_len];
        <Gf16 as Field>::write(&mut data[2 * row_len..3 * row_len], Gf16::GENERATOR);
        let mut repairs = vec![0u8; repair_points * row_len];
        let mut scratch = encoder.scratch();
        encoder.encode(&data, &mut repairs, &mut scratch).unwrap();

        let size = plan.size();
        let mut received = vec![0u8; size * row_len];
        let mut known = vec![false; size];
        received[data_points * row_len..(data_points + repair_points) * row_len]
            .copy_from_slice(&repairs);
        known[data_points..data_points + repair_points].fill(true);
        let locator = ErasureLocator::new(plan, &known).unwrap();
        let missing: Vec<usize> = (0..data_points).collect();
        let mut recovery = RecoveryScratch::new();
        let mut recovered = vec![0u8; data_points * row_len];
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
        assert_eq!(recovered, data);
        assert_eq!(
            <Gf16 as Field>::read(&recovered[2 * row_len..3 * row_len]),
            Gf16::GENERATOR
        );
        for point in [0usize, 1, 3] {
            assert_eq!(
                <Gf16 as Field>::read(&recovered[point * row_len..(point + 1) * row_len]),
                <Gf16 as Field>::Elem::ZERO
            );
        }
    }
}
