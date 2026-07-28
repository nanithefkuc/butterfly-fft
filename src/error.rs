//! Error types for plan construction and transform execution.

/// Error returned when a transform input has the wrong length.
///
/// Element-domain methods report lengths in field elements; byte-row
/// methods report lengths in bytes (documented per method).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformLengthError {
    /// Length required by the plan.
    pub expected: usize,
    /// Length supplied by the caller.
    pub got: usize,
}

impl ::core::fmt::Display for TransformLengthError {
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        write!(
            formatter,
            "wrong transform length: expected {}, got {}",
            self.expected, self.got
        )
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for TransformLengthError {}

/// Error returned when a transform plan cannot be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// Transform size was zero or not a power of two.
    InvalidSize {
        /// The offending size.
        size: usize,
    },
    /// `log2(size)` exceeds the domain cap for the field: the smaller of
    /// the field's extension degree and the table-size cap
    /// (`MAX_LOG_SIZE`).
    DomainTooLarge {
        /// The requested base-two logarithm.
        log_size: usize,
        /// The cap that was exceeded.
        cap: usize,
    },
    /// Fewer basis elements than `log2(size)` were supplied.
    BasisTooShort {
        /// Elements required.
        needed: usize,
        /// Elements supplied.
        got: usize,
    },
    /// The basis prefix is linearly dependent over GF(2).
    DependentBasis,
    /// The Cantor chain `v_i² + v_i = v_{i-1}` does not run the full length
    /// of this field: the step at `dimension` has no root, or the resulting
    /// elements are not a basis. Only fields of power-of-two extension
    /// degree admit a full Cantor basis.
    NoCantorBasis {
        /// The step at which the chain broke.
        dimension: usize,
    },
}

impl ::core::fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            Self::InvalidSize { size } => {
                write!(
                    formatter,
                    "invalid transform size {size}: not a power of two"
                )
            }
            Self::DomainTooLarge { log_size, cap } => {
                write!(
                    formatter,
                    "transform log size {log_size} exceeds domain cap {cap}"
                )
            }
            Self::BasisTooShort { needed, got } => {
                write!(
                    formatter,
                    "basis too short: need {needed} elements, got {got}"
                )
            }
            Self::DependentBasis => {
                write!(
                    formatter,
                    "basis elements are linearly dependent over GF(2)"
                )
            }
            Self::NoCantorBasis { dimension } => {
                write!(
                    formatter,
                    "no Cantor basis for this field: chain broke at v_{dimension}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for PlanError {}
