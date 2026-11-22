"""Pure-stdlib tests of the production GC guard, without runtime imports."""
import gc
import importlib.util
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

# Import this file directly: package initialization also loads training/runtime
# interfaces that are unnecessary for these concurrency and restoration checks.
spec = importlib.util.spec_from_file_location('joint_gc_guard_test_target',
    Path(__file__).resolve().parents[1]/'src/pokezero/joint_budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
defer_joint_gc = budget.defer_joint_gc


class GCDeferral(unittest.TestCase):
    def setUp(self): self.enabled = gc.isenabled()
    def tearDown(self): (gc.enable if self.enabled else gc.disable)()

    def test_nested_exception_restores_both_initial_states_without_collecting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(gc, 'collect') as collect:
                (gc.enable if enabled else gc.disable)()
                with self.assertRaisesRegex(RuntimeError, 'failure'):
                    with defer_joint_gc():
                        self.assertFalse(gc.isenabled())
                        with defer_joint_gc(): self.assertFalse(gc.isenabled())
                        raise RuntimeError('failure')
                self.assertEqual(gc.isenabled(), enabled)
                collect.assert_not_called()

    def test_overlapping_users_restore_only_after_last_exit_in_either_order(self):
        for first_exit in (0, 1):
            with self.subTest(first_exit=first_exit):
                gc.enable()
                entered = [threading.Event(), threading.Event()]
                leave = [threading.Event(), threading.Event()]
                exited = [threading.Event(), threading.Event()]
                errors = []
                def worker(i):
                    try:
                        with defer_joint_gc():
                            entered[i].set()
                            if not leave[i].wait(5): raise RuntimeError('test synchronization timeout')
                    except Exception as error: errors.append(error)
                    finally: exited[i].set()
                threads = [threading.Thread(target=worker,args=(i,)) for i in range(2)]
                try:
                    for i,t in enumerate(threads): t.start(); self.assertTrue(entered[i].wait(5))
                    leave[first_exit].set(); self.assertTrue(exited[first_exit].wait(5))
                    self.assertFalse(gc.isenabled())
                    leave[1-first_exit].set(); self.assertTrue(exited[1-first_exit].wait(5))
                finally:
                    for event in leave: event.set()
                    for t in threads:
                        if t.ident is not None: t.join(5)
                self.assertFalse(errors)
                self.assertTrue(gc.isenabled())
