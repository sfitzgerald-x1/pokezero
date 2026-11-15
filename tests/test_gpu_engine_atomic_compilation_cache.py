"""gpu_engine.atomic_compilation_cache: this process's persistent-cache entries appear whole (a temporary file renamed into
place), with JAX's own key, bytes, get and no-overwrite rule; a size-limited cache keeps JAX's put. CPU only."""
import os
import tempfile
import unittest
from unittest import mock

from gpu_engine import atomic_compilation_cache


class AtomicCacheTests(unittest.TestCase):
    def setUp(self):
        from jax._src import lru_cache
        self.lru = lru_cache;self.assertTrue(atomic_compilation_cache.install());self.assertTrue(atomic_compilation_cache.install())   # idempotent
        self.assertTrue(lru_cache.LRUCache.put.atomic)
        self.dir = tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)

    def test_an_entry_is_renamed_into_place_whole_and_never_overwritten(self):
        cache = self.lru.LRUCache(self.dir.name, max_size=-1);value = os.urandom(1 << 20)
        final = os.path.join(self.dir.name, 'k' + self.lru._CACHE_SUFFIX);seen = []
        replace = os.replace
        def checked(src, dst):   # at the rename: the entry is not there yet, the temporary holds every byte
            seen.append((os.path.exists(final), open(src, 'rb').read() == value, os.path.dirname(os.fspath(src)) == self.dir.name))
            return replace(src, dst)
        with mock.patch.object(os, 'replace', checked):
            cache.put('k', value)
        self.assertEqual(seen, [(False, True, True)]);self.assertEqual(cache.get('k'), value)
        self.assertEqual(sorted(os.listdir(self.dir.name)), ['k' + self.lru._CACHE_SUFFIX])   # no temporary left
        cache.put('k', b'other');self.assertEqual(cache.get('k'), value)   # JAX's rule: an existing entry stays
        with self.assertRaises(ValueError):
            cache.put('', value)

    def test_a_size_limited_cache_keeps_jaxs_own_put(self):
        if self.lru.filelock is None:
            self.skipTest('filelock is not installed')
        cache = self.lru.LRUCache(self.dir.name, max_size=1 << 30)
        with mock.patch.object(os, 'replace', side_effect=AssertionError('not JAX\'s put')):
            cache.put('k', b'v')
        self.assertEqual(cache.get('k'), b'v')


if __name__ == '__main__':
    unittest.main()
