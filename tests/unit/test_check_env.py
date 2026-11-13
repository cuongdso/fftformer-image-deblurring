"""Tests for the environment reporting helper; no external dependency required."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_env  # noqa: E402


class PackageVersionTests(unittest.TestCase):
    def test_unknown_distribution_is_reported(self) -> None:
        self.assertEqual(check_env.package_version("package-that-does-not-exist-fftformer-lab"), "not installed")


if __name__ == "__main__":
    unittest.main()
