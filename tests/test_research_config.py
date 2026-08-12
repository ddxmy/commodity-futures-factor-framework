import unittest

from config.research_config import ResearchConfig


class ResearchConfigTest(unittest.TestCase):
    def test_defaults_match_current_research_conventions(self):
        config = ResearchConfig()

        self.assertEqual(config.rebalance_freq, 1)
        self.assertEqual(config.cost_rate, 0.0005)
        self.assertEqual(config.min_assets, 10)
        self.assertEqual(config.group_count, 5)
        self.assertEqual(config.rolling_ic_window, 20)

    def test_rejects_invalid_values(self):
        invalid_cases = [
            {"rebalance_freq": 0},
            {"cost_rate": -0.0001},
            {"min_assets": 1},
            {"group_count": 1},
            {"rolling_ic_window": 0},
            {"nw_lags": -1},
        ]

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    ResearchConfig(**parameters)


if __name__ == "__main__":
    unittest.main()
