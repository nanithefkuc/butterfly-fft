//! Error types for plan construction and transform execution.

/// Error returned when additive-transform execution inputs are invalid.
///
/// Errors are reported before any destination or scratch is mutated.
/// Element-domain methods report lengths in field elements; byte-row
/// methods report lengths in bytes (documented per method).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformError {
    /// A transform buffer has the wrong length.
    BufferLength {
        /// Length required by the operation.
        expected: usize,
        /// Length supplied by the caller.
        got: usize,
    },
    /// The caller-provided workspace is too short.
    ScratchTooSmall {
        /// Minimum workspace length.
        required: usize,
        /// Workspace length supplied by the caller.
        available: usize,
    },
    /// A row length is zero or not a whole number of field elements.
    InvalidRowLength {
        /// The offending row length, in bytes.
        row_len: usize,
        /// Width of one field element, in bytes.
        element_bytes: usize,
    },
    /// A derived buffer or workspace length cannot be represented by `usize`.
    GeometryOverflow,
    /// Selected indices are out of range, nonascending, or duplicated.
    InvalidSelection,
    /// An output range is reversed or extends beyond the domain.
    InvalidRange {
        /// Requested inclusive start.
        start: usize,
        /// Requested exclusive end.
        end: usize,
        /// Number of rows in the operation's domain.
        size: usize,
    },
    /// The active coefficient prefix is outside the operation's valid bounds.
    InvalidActivePrefix {
        /// Requested active prefix length.
        active: usize,
        /// Number of rows in the plan.
        size: usize,
    },
    /// The operation requires a larger transform plan.
    UnsupportedPlanSize {
        /// Number of rows in the supplied plan.
        size: usize,
        /// Smallest supported plan size.
        minimum: usize,
    },
}

impl ::core::fmt::Display for TransformError {
    fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            Self::BufferLength { expected, got } => {
                write!(
                    formatter,
                    "wrong transform length: expected {expected}, got {got}"
                )
            }
            Self::ScratchTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "scratch too small: need {required}, have {available}"
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
            Self::GeometryOverflow => write!(formatter, "transform geometry overflowed"),
            Self::InvalidSelection => write!(
                formatter,
                "selected rows must be in range, sorted, and unique"
            ),
            Self::InvalidRange { start, end, size } => {
                write!(formatter, "range {start}..{end} is invalid for {size} rows")
            }
            Self::InvalidActivePrefix { active, size } => {
                write!(
                    formatter,
                    "active prefix {active} is invalid for {size} rows"
                )
            }
            Self::UnsupportedPlanSize { size, minimum } => {
                write!(
                    formatter,
                    "plan size {size} is unsupported: need at least {minimum}"
                )
            }
        }
    }
}

impl ::core::error::Error for TransformError {}

/// Error returned when a transform plan cannot be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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

impl ::core::error::Error for PlanError {}
/// Error returned by multiplicative-transform plan construction and
/// execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
    ScratchRowTooSmall {
        /// Row bytes required.
        required: usize,
        /// Row bytes the scratch was built for.
        available: usize,
    },
    /// The supplied scratch batch is shorter than the required byte length.
    ScratchBatchTooSmall {
        /// Batch bytes required.
        required: usize,
        /// Batch bytes available.
        available: usize,
    },
    /// The supplied scratch transpose buffer is shorter than required.
    ScratchTransposeTooSmall {
        /// Transpose bytes required.
        required: usize,
        /// Transpose bytes available.
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
            Self::ScratchRowTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "scratch too small: need {required} row bytes, have {available}"
                )
            }
            Self::ScratchBatchTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "scratch batch too small: need {required} bytes, have {available}"
                )
            }
            Self::ScratchTransposeTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "scratch transpose too small: need {required} bytes, have {available}"
                )
            }
        }
    }
}

impl ::core::error::Error for NttError {}
