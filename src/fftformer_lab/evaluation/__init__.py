"""WS4 evaluation public API for contract C-05."""

from .engine import CpuBasicSRAdapter, EvaluationSummary, SampleEvaluation, evaluate_batch, evaluate_sample
from .errors import ConfigurationError, EvaluationError, EvaluationInputError, MetricExecutionError, ProtocolUnavailableError, UnknownProtocolError
from .config import CONTRACT_VERSION, SCHEMA_VERSION, ResolvedEvaluationConfig, load_evaluation_config
from .metadata import build_experiment_metadata, write_experiment_metadata
from .protocol import GROUP_BASICSR_CPU, REALBLUR_OFFICIAL, UPSTREAM_BASICSR, MetricProtocol, get_metric_protocol, registered_metric_protocols

__all__ = [
    "ConfigurationError", "CONTRACT_VERSION", "CpuBasicSRAdapter", "EvaluationError", "EvaluationInputError", "EvaluationSummary",
    "GROUP_BASICSR_CPU", "MetricExecutionError", "MetricProtocol", "ProtocolUnavailableError",
    "REALBLUR_OFFICIAL", "SampleEvaluation", "UPSTREAM_BASICSR", "UnknownProtocolError",
    "ResolvedEvaluationConfig", "SCHEMA_VERSION", "build_experiment_metadata", "evaluate_batch", "evaluate_sample",
    "get_metric_protocol", "load_evaluation_config", "registered_metric_protocols", "write_experiment_metadata",
]
