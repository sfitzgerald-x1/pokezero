"""Exact debug payloads from sealed Gen 3 owner streams."""

import unittest

from gpu_engine.public_annotations import classify


class OwnerDebugLiteralTests(unittest.TestCase):
    def test_witnessed_diagnostics_and_close_variants(self):
        for root, line, variant in (
            (7, '|debug|Wonder Guard immunity: hiddenpower',
             '|debug|Wonder Guard immunity: hiddenpowerice'),
            (175, '|debug|Sand Veil - decreasing accuracy',
             '|debug|Sand Veil - increasing accuracy'),
            (178, '|debug|Negating (sp)atk boost/penalty.',
             '|debug|Negating (sp)atk boost/penalty'),
        ):
            with self.subTest(root=root, line=line):
                self.assertEqual(classify(line), 'text')
            for refused in (variant, line + '|extra', line + ' ',
                            line + '|[from] ability: Pressure', line + '\n'):
                with self.subTest(root=root, line=refused):
                    self.assertIsNone(classify(refused))
