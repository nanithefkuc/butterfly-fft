//! The Cantor basis and its process-wide cache.
//!
//! Cantor's chain (Cantor 1989) is `v_0 = 1` with `v_i² + v_i = v_{i-1}`.
//! Over `GF(2^m)` with `m` a power of two — which every field this crate
//! supports is — the chain runs the full `m` steps and its elements form a
//! GF(2)-basis.
//!
//! Its defining property is that **every normalizer is one**:
//! `W_k(v_k) = 1`, so the subspace-polynomial recursion degenerates from
//! `W_{k+1} = W_k² + W_k(v_k)·W_k` to plain `W_{k+1} = W_k² + W_k` — a
//! Frobenius and an add, with no field multiplication anywhere in table
//! construction, and `W̄_k = W_k` needing no normalization. The coefficients
//! are then Pascal's triangle mod two: `W_k = Σ_j C(k,j)·x^(2^j)`, so by
//! Lucas' theorem `x^(2^j)` appears exactly when `j` is a submask of `k`.
//! (`W_k` is a *binomial* `x^(2^k) + x` only when `k` is itself a power of
//! two — `W_3 = x^8 + x^4 + x^2 + x`, not `x^8 + x`.)
//!
//! Each step solves a linear system (`x ↦ x² + x` is GF(2)-linear in
//! characteristic two), so the whole basis costs `O(m³)` bit operations
//! once per field — hence the cache.

use ::alloc::vec::Vec;

use fff::field::{Elem, Field};

use super::gf2;
use super::{OrderedBasis, independent};
use crate::error::PlanError;

/// The Cantor basis of a field: `v_0 = 1`, `v_i² + v_i = v_{i-1}`.
///
/// Obtain a shared instance from [`cantor_basis`] rather than rebuilding;
/// the chain is a property of the field, not of any one transform.
#[derive(Clone, Debug)]
pub struct CantorBasis<F: Field> {
    elements: Vec<F::Elem>,
}

impl<F: Field> CantorBasis<F> {
    /// Solve the Cantor chain for this field.
    ///
    /// # Errors
    /// [`PlanError::NoCantorBasis`] if the chain breaks: either a step's
    /// quadratic has no root (the field's degree is not a power of two), or
    /// the resulting elements are GF(2)-dependent.
    pub fn build() -> Result<Self, PlanError> {
        let bits = F::BITS as usize;
        let mut elements = Vec::with_capacity(bits);
        elements.push(F::Elem::ONE);
        for dimension in 1..bits {
            let previous = elements[dimension - 1];
            let next = gf2::solve_quadratic::<F>(previous)
                .ok_or(PlanError::NoCantorBasis { dimension })?;
            elements.push(next);
        }
        if !independent::<F>(&elements) {
            return Err(PlanError::NoCantorBasis { dimension: bits });
        }
        Ok(Self { elements })
    }

    /// The basis elements `v_0 … v_{BITS-1}`.
    #[must_use]
    pub fn elements(&self) -> &[F::Elem] {
        &self.elements
    }
}

impl<F: Field> OrderedBasis<F> for CantorBasis<F>
where
    F::Elem: Send + Sync,
{
    #[inline]
    fn element(&self, index: usize) -> F::Elem {
        self.elements[index]
    }

    #[inline]
    fn bits(&self) -> usize {
        self.elements.len()
    }
}

/// The shared Cantor basis for this field, solved on first use.
///
/// # Errors
/// As [`CantorBasis::build`].
#[cfg(feature = "std")]
pub fn cantor_basis<F>() -> Result<::alloc::sync::Arc<CantorBasis<F>>, PlanError>
where
    F: Field,
    F::Elem: Send + Sync,
{
    use ::alloc::sync::Arc;
    use ::core::any::{Any, TypeId};
    use ::std::collections::HashMap;
    use ::std::sync::{LazyLock, PoisonError, RwLock};

    type Cache = LazyLock<RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>;
    static CANTOR_BASES: Cache = LazyLock::new(|| RwLock::new(HashMap::new()));

    let key = TypeId::of::<F>();
    {
        let cache = CANTOR_BASES.read().unwrap_or_else(PoisonError::into_inner);
        if let Some(basis) = cache
            .get(&key)
            .and_then(|erased| erased.clone().downcast::<CantorBasis<F>>().ok())
        {
            return Ok(basis);
        }
    }
    let fresh = Arc::new(CantorBasis::<F>::build()?);
    let erased: Arc<dyn Any + Send + Sync> = fresh.clone();
    let mut cache = CANTOR_BASES.write().unwrap_or_else(PoisonError::into_inner);
    let stored = cache.entry(key).or_insert(erased).clone();
    // The key is F's TypeId, so the downcast cannot fail; falling back to
    // the freshly built basis keeps this path panic-free.
    Ok(stored.downcast::<CantorBasis<F>>().unwrap_or(fresh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factors;
    use fff::{FanPaar8, FanPaar16, Gf8, Gf16, Gf32, Gf64};

    /// Contract: `v_0 = 1` and `v_i² + v_i = v_{i-1}` for every step.
    fn chain_holds<F: Field>() {
        let basis = CantorBasis::<F>::build().expect("power-of-two degree");
        let elements = basis.elements();
        assert_eq!(elements.len(), F::BITS as usize);
        assert!(elements[0].is_one(), "{}: v_0 must be one", F::NAME);
        for index in 1..elements.len() {
            let value = elements[index];
            assert_eq!(
                value.square().add(value),
                elements[index - 1],
                "{}: chain broken at v_{index}",
                F::NAME
            );
        }
    }

    #[test]
    fn cantor_chain_holds_for_every_field() {
        chain_holds::<Gf8>();
        chain_holds::<Gf16>();
        chain_holds::<Gf32>();
        chain_holds::<Gf64>();
        chain_holds::<FanPaar8>();
        chain_holds::<FanPaar16>();
    }

    #[test]
    fn cantor_elements_are_a_basis() {
        fn check<F: Field>() {
            let basis = CantorBasis::<F>::build().unwrap();
            assert!(independent::<F>(basis.elements()), "{}", F::NAME);
        }
        check::<Gf8>();
        check::<Gf16>();
        check::<Gf32>();
        check::<Gf64>();
    }

    /// The reason the basis exists: over it every normalizer `W_k(v_k)` is
    /// one, so table construction needs no multiplication, and the
    /// coefficients are Pascal's triangle mod two — `x^(2^j)` is present
    /// exactly when `j` is a submask of `k` (Lucas).
    #[test]
    fn subspace_polynomials_follow_pascal_mod_two() {
        fn check<F: Field>() {
            let basis = CantorBasis::<F>::build().unwrap();
            let polynomials = factors::subspace_polynomials(basis.elements())
                .expect("cantor basis is independent");
            for (k, polynomial) in polynomials.iter().enumerate() {
                assert!(
                    polynomial.normalizer_inverse.is_one(),
                    "{} W_{k}(v_{k}) is not one",
                    F::NAME
                );
                assert_eq!(polynomial.coefficients.len(), k + 1);
                for (j, &coefficient) in polynomial.coefficients.iter().enumerate() {
                    let expected = if j & !k == 0 {
                        F::Elem::ONE
                    } else {
                        F::Elem::ZERO
                    };
                    assert_eq!(coefficient, expected, "{} W_{k} coefficient {j}", F::NAME);
                }
            }
        }
        check::<Gf8>();
        check::<Gf16>();
        check::<Gf32>();
        check::<Gf64>();
    }

    #[cfg(feature = "std")]
    #[test]
    fn shared_cantor_bases_are_cached() {
        let first = cantor_basis::<Gf16>().unwrap();
        let second = cantor_basis::<Gf16>().unwrap();
        assert!(::alloc::sync::Arc::ptr_eq(&first, &second));
        let other = cantor_basis::<Gf8>().unwrap();
        assert_eq!(other.elements().len(), 8);
    }
}
