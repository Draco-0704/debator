import unittest

from tools.stance_detector import _parse_result


class StanceDetectorTests(unittest.TestCase):
    def test_parse_result_maps_groq_labels_correctly(self):
        result = {"stance": "supported", "confidence": 0.81}

        parsed = _parse_result(result)

        self.assertEqual(parsed["stance"], "SUPPORT")
        self.assertEqual(parsed["confidence"], 0.81)

    def test_parse_result_handles_hf_style_outputs_for_compatibility(self):
        result = [[
            {"label": "contradiction", "score": 0.7},
            {"label": "entailment", "score": 0.2},
            {"label": "neutral", "score": 0.1},
        ]]

        parsed = _parse_result(result)

        self.assertEqual(parsed["stance"], "CONTRADICT")
        self.assertEqual(parsed["confidence"], 0.7)

    def test_parse_result_defaults_unknown_labels_to_neutral(self):
        result = {"stance": "maybe", "confidence": 0.4}

        parsed = _parse_result(result)

        self.assertEqual(parsed["stance"], "NEUTRAL")
        self.assertEqual(parsed["confidence"], 0.4)


if __name__ == "__main__":
    unittest.main()
