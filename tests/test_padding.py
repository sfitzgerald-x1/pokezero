import unittest

from pokezero.padding import zeros_like


class PaddingTest(unittest.TestCase):
    def test_zeros_like_reuses_regular_shape_padding(self) -> None:
        first = ((1, 2, 3), (4, 5, 6))
        second = ((9, 8, 7), (6, 5, 4))

        padding = zeros_like(first)
        self.assertEqual(padding, ((0, 0, 0), (0, 0, 0)))
        self.assertIs(padding, zeros_like(second))
        self.assertIs(padding[0], padding[1])

    def test_zeros_like_preserves_scalar_zero_types(self) -> None:
        bool_padding = zeros_like((True, True))
        int_padding = zeros_like((1, 2))
        float_padding = zeros_like((1.5, 2.5))

        self.assertEqual(bool_padding, (False, False))
        self.assertEqual(int_padding, (0, 0))
        self.assertEqual(float_padding, (0.0, 0.0))
        self.assertIs(type(bool_padding[0]), bool)
        self.assertIs(type(int_padding[0]), int)
        self.assertIs(type(float_padding[0]), float)

    def test_zeros_like_handles_empty_tuple(self) -> None:
        self.assertEqual(zeros_like(()), ())
        self.assertIs(zeros_like(()), zeros_like(()))


if __name__ == "__main__":
    unittest.main()
