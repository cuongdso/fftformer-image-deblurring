"""Dependency-free regressions for WS1 evidence reporting, including failed subtests."""

import importlib.util
import io
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ws1_03_verify.py"
spec = importlib.util.spec_from_file_location("ws1_03_evidence_reporter", SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def run_cases(*cases: unittest.TestCase) -> reporter.RecordingResult:
    """Execute small in-memory test cases while capturing their expected failures."""
    return unittest.TextTestRunner(stream=io.StringIO(), resultclass=reporter.RecordingResult).run(
        unittest.TestSuite(cases)
    )


class EvidenceRecordingTests(unittest.TestCase):
    def test_successful_subtests_count_only_the_parent(self) -> None:
        class Case(unittest.TestCase):
            def runTest(self) -> None:
                for value in (1, 2, 3):
                    with self.subTest(value=value):
                        self.assertGreater(value, 0)

        result = run_cases(Case())
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.subtests, {})
        self.assertEqual(reporter.summarize_result(result)["passed"], 1)
        self.assertEqual(reporter.summarize_result(result)["status"], "PASS")

    def test_multiple_failed_subtests_and_error_are_recorded(self) -> None:
        class Case(unittest.TestCase):
            def runTest(self) -> None:
                for value in (1, 2):
                    with self.subTest(value=value):
                        self.fail(f"failed value {value}")
                with self.subTest(value="error"):
                    raise ValueError("subtest error")

        case = Case()
        result = run_cases(case)
        self.assertEqual(result.records[case.id()]["status"], "ERROR")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.subtests), 3)
        self.assertEqual({record["status"] for record in result.subtests.values()}, {"FAIL", "ERROR"})
        self.assertTrue(all(record["parent"] == case.id() for record in result.subtests.values()))
        self.assertTrue(all(record["message"] for record in result.subtests.values()))
        summary = reporter.summarize_result(result)
        self.assertEqual((summary["total"], summary["passed"], summary["failed"], summary["errors"]), (1, 0, 2, 1))
        self.assertEqual(summary["status"], "FAIL")

    def test_skipped_subtest_is_not_overwritten_by_parent_success(self) -> None:
        class Case(unittest.TestCase):
            def runTest(self) -> None:
                with self.subTest(value="skip"):
                    self.skipTest("subtest unavailable")
                with self.subTest(value="pass"):
                    self.assertTrue(True)

        case = Case()
        result = run_cases(case)
        self.assertEqual(result.records[case.id()]["status"], "SKIP")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.subtests), 1)
        self.assertEqual(reporter.summarize_result(result)["passed"], 0)
        self.assertEqual(reporter.summarize_result(result)["status"], "FAIL")

    def test_repeated_subtest_labels_preserve_every_failure_and_error(self) -> None:
        class Case(unittest.TestCase):
            def runTest(self) -> None:
                for number in range(3):
                    with self.subTest(state_type="dict"):
                        if number == 1:
                            raise ValueError("second occurrence error")
                        self.fail(f"failure occurrence {number}")

        case = Case()
        result = run_cases(case)
        events = list(result.subtests.values())
        self.assertEqual(len(events), 3)
        self.assertEqual([event["status"] for event in events], ["FAIL", "ERROR", "FAIL"])
        self.assertEqual([event["message"] for event in events], [
            "failure occurrence 0", "second occurrence error", "failure occurrence 2",
        ])
        self.assertEqual(len({event["id"] for event in events}), 1)
        self.assertTrue(all(event["parent"] == case.id() for event in events))
        self.assertEqual(result.records[case.id()]["status"], "ERROR")
        summary = reporter.summarize_result(result)
        self.assertEqual((summary["total"], summary["passed"], summary["failed"], summary["errors"]), (1, 0, 2, 1))
        self.assertEqual(summary["status"], "FAIL")

    def test_repeated_skipped_subtest_labels_preserve_every_reason(self) -> None:
        class Case(unittest.TestCase):
            def runTest(self) -> None:
                for number in range(3):
                    with self.subTest(state_type="dict"):
                        self.skipTest(f"unavailable occurrence {number}")

        case = Case()
        result = run_cases(case)
        events = list(result.subtests.values())
        self.assertEqual(len(events), 3)
        self.assertEqual([event["reason"] for event in events], [
            "unavailable occurrence 0", "unavailable occurrence 1", "unavailable occurrence 2",
        ])
        self.assertEqual(len({event["id"] for event in events}), 1)
        self.assertTrue(all(event["parent"] == case.id() and event["status"] == "SKIP" for event in events))
        self.assertEqual(result.records[case.id()]["status"], "SKIP")
        summary = reporter.summarize_result(result)
        self.assertEqual((summary["total"], summary["passed"], summary["skipped"]), (1, 0, 3))
        self.assertEqual(summary["status"], "FAIL")

    def test_regular_failure_and_error_keep_separate_parent_records(self) -> None:
        class Failure(unittest.TestCase):
            def runTest(self) -> None:
                self.fail("ordinary failure")

        class Error(unittest.TestCase):
            def runTest(self) -> None:
                raise RuntimeError("ordinary error")

        result = run_cases(Failure(), Error())
        self.assertEqual(len(result.records), 2)
        self.assertEqual({record["status"] for record in result.records.values()}, {"FAIL", "ERROR"})
        self.assertEqual(reporter.summarize_result(result)["status"], "FAIL")

    def test_whole_test_skip_is_recorded(self) -> None:
        class Case(unittest.TestCase):
            def runTest(self) -> None:
                self.skipTest("whole test unavailable")

        case = Case()
        result = run_cases(case)
        self.assertEqual(result.records[case.id()], {"status": "SKIP", "reason": "whole test unavailable"})
        self.assertEqual(reporter.summarize_result(result)["status"], "FAIL")

    def test_empty_suite_cannot_claim_pass(self) -> None:
        result = run_cases()
        summary = reporter.summarize_result(result)
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["status"], "FAIL")
