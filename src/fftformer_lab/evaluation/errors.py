"""Exceptions raised by the WS4 evaluation boundary (C-05)."""


class EvaluationError(RuntimeError):
    """Base error for evaluation operations."""


class EvaluationInputError(EvaluationError, ValueError):
    """Prediction, target, identity, or batch inputs violate C-02--C-05."""


class UnknownProtocolError(EvaluationError, ValueError):
    """The requested metric protocol is not registered."""


class ProtocolUnavailableError(EvaluationError):
    """A registered protocol cannot run in the current environment/scope."""


class MetricExecutionError(EvaluationError):
    """A metric adapter failed after validated inputs were supplied."""


class ConfigurationError(EvaluationError, ValueError):
    """An evaluation configuration violates the C-06 schema or filesystem policy."""
