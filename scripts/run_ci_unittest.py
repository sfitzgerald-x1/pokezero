"""Run an explicit unittest selection and require real, unskipped execution."""
import sys
import unittest


def run_selected(names):
    if not names:
        raise ValueError("At least one test selector is required")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in names:
        selected = loader.loadTestsFromName(name)
        if not selected.countTestCases():
            raise ValueError(f"No tests found for {name}")
        suite.addTest(selected)

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return int(not result.wasSuccessful() or bool(result.skipped) or result.testsRun == 0)


if __name__ == "__main__":
    raise SystemExit(run_selected(sys.argv[1:]))
