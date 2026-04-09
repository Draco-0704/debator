import unittest

from agents.judge_agent import _score_overall_verdict


class OverallVerdictTests(unittest.TestCase):
    def test_supported_claims_drive_real_verdict(self):
        verdicts = [
            {"verdict": "SUPPORTED", "confidence": 0.92},
            {"verdict": "SUPPORTED", "confidence": 0.81},
            {"verdict": "UNVERIFIABLE", "confidence": 0.45},
        ]

        scored = _score_overall_verdict(verdicts)

        self.assertEqual(scored["overall_verdict"], "REAL")
        self.assertGreater(
            scored["confidence_metrics"]["REAL"],
            scored["confidence_metrics"]["MISLEADING"],
        )

    def test_refuted_claims_drive_fake_verdict(self):
        verdicts = [
            {"verdict": "REFUTED", "confidence": 0.88},
            {"verdict": "REFUTED", "confidence": 0.79},
            {"verdict": "UNVERIFIABLE", "confidence": 0.30},
        ]

        scored = _score_overall_verdict(verdicts)

        self.assertEqual(scored["overall_verdict"], "FAKE")
        self.assertGreater(
            scored["confidence_metrics"]["FAKE"],
            scored["confidence_metrics"]["MISLEADING"],
        )

    def test_balanced_real_and_fake_becomes_misleading(self):
        verdicts = [
            {"verdict": "SUPPORTED", "confidence": 0.84},
            {"verdict": "REFUTED", "confidence": 0.83},
            {"verdict": "UNVERIFIABLE", "confidence": 0.50},
        ]

        scored = _score_overall_verdict(verdicts)

        self.assertEqual(scored["overall_verdict"], "MISLEADING")
        self.assertGreaterEqual(
            scored["confidence_metrics"]["MISLEADING"],
            scored["confidence_metrics"]["REAL"],
        )
        self.assertGreaterEqual(
            scored["confidence_metrics"]["MISLEADING"],
            scored["confidence_metrics"]["FAKE"],
        )


if __name__ == "__main__":
    unittest.main()
