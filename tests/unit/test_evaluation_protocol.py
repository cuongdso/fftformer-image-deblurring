"""Unit tests for WS4-02 protocol registration and availability boundaries."""

from dataclasses import FrozenInstanceError
import unittest

from fftformer_lab.evaluation import (
    GROUP_BASICSR_CPU,
    REALBLUR_OFFICIAL,
    UPSTREAM_BASICSR,
    MetricProtocol,
    ProtocolUnavailableError,
    UnknownProtocolError,
    get_metric_protocol,
    registered_metric_protocols,
)


class EvaluationProtocolTests(unittest.TestCase):
    def test_registry_contains_three_distinct_protocols(self) -> None:
        protocols = registered_metric_protocols()
        self.assertEqual([item.protocol_id for item in protocols], [
            "upstream-basicsr", "group-basicsr-cpu", "realblur-official",
        ])
        self.assertIs(get_metric_protocol("group-basicsr-cpu"), GROUP_BASICSR_CPU)

    def test_unknown_protocol_is_not_a_fallback(self) -> None:
        with self.assertRaises(UnknownProtocolError):
            get_metric_protocol("basicsr")

    def test_registered_id_cannot_override_its_policy(self) -> None:
        altered = MetricProtocol(
            "group-basicsr-cpu", "ws4-02-v1", GROUP_BASICSR_CPU.implementation_source,
            "BGR", "HWC", (0, 255), GROUP_BASICSR_CPU.clamp_policy,
            GROUP_BASICSR_CPU.quantization_policy, 4, False, "ssim3d=False", "CPU", True,
        )
        with self.assertRaises(UnknownProtocolError):
            get_metric_protocol(altered)

    def test_protocol_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            GROUP_BASICSR_CPU.crop_border = 4  # type: ignore[misc]

    def test_cpu_variant_is_distinct_from_upstream_cuda_default(self) -> None:
        self.assertNotEqual(GROUP_BASICSR_CPU.protocol_id, UPSTREAM_BASICSR.protocol_id)
        self.assertEqual(GROUP_BASICSR_CPU.ssim_variant, "ssim3d=False")
        self.assertEqual(UPSTREAM_BASICSR.ssim_variant, "ssim3d=True")
        self.assertIn("CUDA", UPSTREAM_BASICSR.device_requirement)

    def test_realblur_is_registered_but_not_silently_substituted(self) -> None:
        self.assertFalse(REALBLUR_OFFICIAL.supported)
        self.assertIn("ECC", REALBLUR_OFFICIAL.unavailable_reason or "")

    def test_invalid_protocol_definition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricProtocol("", "v1", "x", "BGR", "HWC", (0, 255), "x", "x", 0, False, "x", "CPU", True)
