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
/// Error returned by multiplicative-transform plan construction and
/// execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttError {
    /// Transform size was zero or not a power of two.
    InvalidSize {
        /// The offending size.
        size: usize,
    },
    /// The field has no primitive `size`-th root of unity, or `size`
    /// exceeds the table-size cap `MAX_LOG_SIZE`. A power-of-two `size`
    /// above one needs `size` to divide `field_order - 1`, which no binary
    /// field satisfies.
    UnsupportedSize {
        /// The requested size.
        size: usize,
        /// Number of elements in the field.
        field_order: u128,
    },
    /// A size, offset, or byte length did not fit in `usize`/`u64`.
    GeometryOverflow,
    /// A table or scratch allocation failed.
    AllocationFailed,
    /// The row buffer length does not match `size * row_len`.
    BufferLength {
        /// Length the plan requires, in bytes.
        expected: usize,
        /// Length the caller supplied, in bytes.
        actual: usize,
    },
    /// A row length is zero or not a whole number of field elements.
    InvalidRowLength {
        /// The offending row length, in bytes.
        row_len: usize,
        /// Width of one field element, in bytes.
        element_bytes: usize,
    },
    /// The supplied scratch row temporary was built for a field of a
    /// different element width.
    ScratchFieldMismatch {
        /// Element width the scratch was built for, in bytes.
        scratch_element_bytes: usize,
        /// Element width of this plan's field, in bytes.
        field_element_bytes: usize,
    },
    /// The supplied scratch row temporary is shorter than the requested
    /// row length.
    ScratchTooSmall {
        /// Row bytes required.
        required: usize,
        /// Row bytes the scratch was built for.
        available: usize,
    },
}

impl ::core::fmt::Display for NttError {
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            Self::InvalidSize { size } => {
                write!(
                    formatter,
                    "invalid transform size {size}: not a power of two"
                )
            }
            Self::UnsupportedSize { size, field_order } => {
                write!(
                    formatter,
                    "transform size {size} is unsupported: it exceeds the table-size cap, or the field of order {field_order} has no primitive root of unity of that order"
                )
            }
            Self::GeometryOverflow => {
                write!(formatter, "transform geometry overflowed")
            }
            Self::AllocationFailed => {
                write!(formatter, "transform table allocation failed")
            }
            Self::BufferLength { expected, actual } => {
                write!(
                    formatter,
                    "wrong row buffer length: expected {expected} bytes, got {actual}"
                )
            }
            Self::InvalidRowLength {
                row_len,
                element_bytes,
            } => {
                write!(
                    formatter,
                    "row length {row_len} is zero or not a whole number of {element_bytes}-byte elements"
                )
            }
            Self::ScratchFieldMismatch {
                scratch_element_bytes,
                field_element_bytes,
            } => {
                write!(
                    formatter,
                    "scratch was built for {scratch_element_bytes}-byte elements, but this plan transforms {field_element_bytes}-byte elements"
                )
            }
            Self::ScratchTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "scratch too small: need {required} row bytes, have {available}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for NttError {}
