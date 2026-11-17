"""The bounded id memos (dex.normalize_id, showdown._normalize_identifier) return exactly the unmemoized expression on every
identifier the live constructor normalizes (a committed sample of screen-r5a's prepare corpus) and on non-str inputs."""
import json
from pathlib import Path
import re
import unittest

import numpy as np

from pokezero import dex, showdown

CORPUS = json.loads((Path(__file__).parent / 'fixtures' / 'identifier_corpus_r5a.json').read_text())['strings']
EXTRA = ['', ' ', 'Mr. Mime', "Farfetch’d", 'Nidoran-F', 'Porygon2', 'Ho-Oh', 'Hidden Power [Fire]', 'p2a: Snorlax', 'É', 'ß', 'ǅ', 'İstanbul',
         '[from] ability: Flash Fire', 'move: Encore', '100/100', 'L81', '-' * 40]


def expression(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


class NormalizeMemo(unittest.TestCase):
    def test_memoized_equals_the_expression_on_the_corpus_and_edge_strings(self):
        self.assertGreater(len(CORPUS), 1000)
        for value in CORPUS + EXTRA:
            expected = expression(value)
            self.assertEqual(dex.normalize_id(value), expected, value)
            self.assertEqual(showdown._normalize_identifier(value), expected, value)
            self.assertEqual(dex.normalize_id(value), expected, value)   # a cache hit too

    def test_non_str_arguments_take_the_unmemoized_expression(self):
        for value in (None, 0, 17, 1.5, np.str_('Abc-D'), b'xy', ('a',)):
            self.assertEqual(dex.normalize_id(value), expression(value), repr(value))
            self.assertEqual(showdown._normalize_identifier(value), expression(value), repr(value))

    def test_the_memos_are_bounded(self):
        self.assertEqual(dex._normalize_str.cache_info().maxsize, 65536)
        self.assertEqual(showdown._normalize_identifier_str.cache_info().maxsize, 65536)


if __name__ == '__main__':
    unittest.main()
