"""gpu_engine.atomic_compilation_cache: this process's persistent-cache entries appear whole (a unique temporary file, fsynced,
renamed into place; a failed write leaves nothing), with JAX's own key, bytes, get and no-overwrite rule; a size-limited cache
and a remote one keep JAX's put; an unknown JAX is left alone and said so. CPU only."""
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

    def test_a_failed_write_leaves_no_entry_and_no_temporary(self):
        cache = self.lru.LRUCache(self.dir.name, max_size=-1)
        with mock.patch.object(os, 'replace', side_effect=OSError('disk full')), self.assertRaises(OSError):
            cache.put('k', b'v')
        self.assertEqual(os.listdir(self.dir.name), []);self.assertIsNone(cache.get('k'))

    def test_temporaries_are_unique_per_write(self):   # mkstemp: no pid / thread name to collide across pods on a shared volume
        cache = self.lru.LRUCache(self.dir.name, max_size=-1);names = []
        replace = os.replace
        with mock.patch.object(os, 'replace', lambda src, dst: (names.append(os.path.basename(src)), replace(src, dst))):
            for key in ('a', 'b', 'c'):
                cache.put(key, key.encode())
        self.assertEqual(len(set(names)), 3);self.assertTrue(all(n.startswith('.') and n.endswith('.tmp') for n in names))

    def test_a_cache_off_the_local_filesystem_keeps_jaxs_own_put(self):
        cache = self.lru.LRUCache(self.dir.name, max_size=-1);cache._is_local = False
        with mock.patch.object(os, 'replace', side_effect=AssertionError('not JAX\'s put')):
            cache.put('k', b'v')
        self.assertEqual(cache.get('k'), b'v')

    def test_an_unknown_jax_is_left_alone_and_said_so(self):
        with mock.patch.object(self.lru, '_CACHE_SUFFIX', None):
            self.assertFalse(atomic_compilation_cache.install())
            self.assertEqual(atomic_compilation_cache.STATUS['installed'], False);self.assertIn('patched shape', atomic_compilation_cache.STATUS['reason'])
        self.assertTrue(atomic_compilation_cache.install());self.assertTrue(atomic_compilation_cache.STATUS['installed'])

    def test_an_entry_gets_the_mode_jaxs_own_write_gives(self):   # mkstemp's 0600 would hide entries from another user's reader
        cache = self.lru.LRUCache(self.dir.name, max_size=-1);before = dict(atomic_compilation_cache.STATUS)
        cache.put('k', b'v');reference = os.path.join(self.dir.name, 'reference')
        with open(reference, 'wb') as f:   # Path.write_bytes' open: 0666 under the process umask
            f.write(b'v')
        mode = lambda path: os.stat(path).st_mode & 0o777
        self.assertEqual(mode(os.path.join(self.dir.name, 'k' + self.lru._CACHE_SUFFIX)), mode(reference))
        self.assertEqual(atomic_compilation_cache.STATUS['atomic_writes'], before['atomic_writes'] + 1)
        self.assertEqual(atomic_compilation_cache.STATUS['jax_writes'], before['jax_writes'])

    def test_a_cache_this_patch_does_not_know_keeps_jaxs_own_put(self):   # neither attribute's value is assumed
        for attribute in ('_is_local', 'eviction_enabled'):
            with self.subTest(attribute=attribute):
                cache = self.lru.LRUCache(self.dir.name, max_size=-1);setattr(cache, attribute, None);before = dict(atomic_compilation_cache.STATUS)
                with mock.patch.object(os, 'replace', side_effect=AssertionError('not JAX\'s put')):
                    cache.put(attribute, b'v')
                self.assertEqual(cache.get(attribute), b'v')
                self.assertEqual(atomic_compilation_cache.STATUS['jax_writes'], before['jax_writes'] + 1)
                self.assertEqual(atomic_compilation_cache.STATUS['atomic_writes'], before['atomic_writes'])

    def test_a_size_limited_cache_keeps_jaxs_own_put(self):
        if self.lru.filelock is None:
            self.skipTest('filelock is not installed')
        cache = self.lru.LRUCache(self.dir.name, max_size=1 << 30)
        with mock.patch.object(os, 'replace', side_effect=AssertionError('not JAX\'s put')):
            cache.put('k', b'v')
        self.assertEqual(cache.get('k'), b'v')


if __name__ == '__main__':
    unittest.main()
