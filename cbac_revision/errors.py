"""Domain-specific failures that prevent silent benchmark fallbacks."""


class RevisionProtocolError(ValueError):
    """Base class for a predeclared-protocol violation."""


class MissingControlError(RevisionProtocolError):
    """Raised when none of the explicitly declared control labels are present."""


class AmbiguousAliasError(RevisionProtocolError):
    """Raised when one raw alias maps to incompatible canonical labels."""


class LeakageRiskError(RevisionProtocolError):
    """Raised when preprocessing cannot prove a control-only fit scope."""


class GraphSupportError(RevisionProtocolError):
    """Raised when a graph support violates the matched-control protocol."""


class IncompleteSeedSetError(RevisionProtocolError):
    """Raised when a condition is missing a predeclared seed."""


class UnpairedConditionError(RevisionProtocolError):
    """Raised when compared arms do not contain identical condition keys."""


class ArtifactValidationError(RevisionProtocolError):
    """Raised when a fold artifact is incomplete or internally inconsistent."""
