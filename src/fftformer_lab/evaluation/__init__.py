"""WS4 evaluation public API for contract C-05."""

from .engine import CpuBasicSRAdapter, EvaluationSummary, SampleEvaluation, evaluate_batch, evaluate_sample
from .errors import EvaluationError, EvaluationInputError, MetricExecutionError, ProtocolUnavailableError, UnknownProtocolError
from .protocol import GROUP_BASICSR_CPU, REALBLUR_OFFICIAL, UPSTREAM_BASICSR, MetricProtocol, get_metric_protocol, registered_metric_protocols

__all__ = [
    "CpuBasicSRAdapter", "EvaluationError", "EvaluationInputError", "EvaluationSummary",
    "GROUP_BASICSR_CPU", "MetricExecutionError", "MetricProtocol", "ProtocolUnavailableError",
    "REALBLUR_OFFICIAL", "SampleEvaluation", "UPSTREAM_BASICSR", "UnknownProtocolError",
    "evaluate_batch", "evaluate_sample", "get_metric_protocol", "registered_metric_protocols",
]
