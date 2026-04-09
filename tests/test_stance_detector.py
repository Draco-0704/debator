import unittest

from tools.stance_detector import _parse_result


class StanceDetectorTests(unittest.TestCase):
    def test_parse_result_maps_model_labels_correctly(self):
        result = [
            {"label": "LABEL_0", "score": 0.1},
            {"label": "LABEL_1", "score": 0.8},
            {"label": "LABEL_2", "score": 0.1},
        ]

        parsed = _parse_result(result)

        self.assertEqual(parsed["stance"], "SUPPORT")
        self.assertEqual(parsed["confidence"], 0.8)

    def test_parse_result_handles_nested_outputs(self):
        result = [[
            {"label": "contradiction", "score": 0.7},
            {"label": "entailment", "score": 0.2},
            {"label": "neutral", "score": 0.1},
        ]]

        parsed = _parse_result(result)

        self.assertEqual(parsed["stance"], "CONTRADICT")
        self.assertEqual(parsed["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
