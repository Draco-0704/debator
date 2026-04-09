import unittest

from agents.judge_agent import _format_evidence_block


class JudgeAgentTests(unittest.TestCase):
    def test_format_evidence_block_orders_by_confidence(self):
        evidence = [
            {
                "title": "Low",
                "confidence": 0.2,
                "stance": "SUPPORT",
                "url": "https://a.test",
                "snippet": "Low confidence.",
            },
            {
                "title": "High",
                "confidence": 0.9,
                "stance": "SUPPORT",
                "url": "https://b.test",
                "snippet": "High confidence.",
            },
        ]

        block = _format_evidence_block(evidence)

        self.assertIn("High", block.splitlines()[0])
        self.assertIn("https://b.test", block)

    def test_format_evidence_block_handles_missing_evidence(self):
        self.assertEqual(_format_evidence_block([]), "No direct evidence provided.")


if __name__ == "__main__":
    unittest.main()
