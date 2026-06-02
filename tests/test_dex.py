from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pokezero.dex import load_showdown_dex_cached, showdown_dex_from_payload


class DexTest(unittest.TestCase):
    def test_showdown_dex_payload_normalizes_perfect_accuracy(self) -> None:
        dex = showdown_dex_from_payload(
            {
                "moves": {
                    "swift": {
                        "id": "swift",
                        "name": "Swift",
                        "type": "Normal",
                        "category": "Special",
                        "basePower": 60,
                        "accuracy": True,
                        "priority": 0,
                    }
                },
                "species": {},
                "typeChart": {},
            }
        )

        assert dex.move_info("swift") is not None
        self.assertEqual(dex.move_info("swift").accuracy, 100.0)

    def test_showdown_dex_cached_loads_once_per_root(self) -> None:
        first = showdown_dex_from_payload({"moves": {}, "species": {}, "typeChart": {}})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("pokezero.dex.load_showdown_dex", return_value=first) as load:
                first_result = load_showdown_dex_cached(root)
                second_result = load_showdown_dex_cached(root)

        self.assertIs(first_result, first)
        self.assertIs(second_result, first)
        self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
