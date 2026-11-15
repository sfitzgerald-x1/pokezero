"""JAX's persistent compilation cache, written atomically by this process.

JAX's file cache (jax._src.lru_cache.LRUCache; without a size limit, max_size -1, it takes no lock) writes an entry in place
(Path.write_bytes). A reader in another process that finds the file before that write ends reads a truncated entry: JAX warns
"Error reading persistent compilation cache entry" and compiles again. A writer killed mid-write leaves the truncated entry for
good, since put never overwrites an existing one. Several processes share one cache directory here (a server and its prepare
workers, two servers on one GPU), so install() makes this process's put write a temporary file in the cache directory and
rename it into place (os.replace: atomic within one directory): a reader sees a whole entry or none. Nothing else changes: the
key, the bytes, get(), put's no-overwrite rule; a size-limited cache (eviction on, JAX's own file lock) keeps JAX's put.
Idempotent. A JAX whose cache does not have the shape patched here is left alone (install returns False)."""
import os
import threading


def install():
    """Patch LRUCache.put in this process (True), or leave an unknown JAX alone (False)."""
    try:
        from jax._src import lru_cache
    except ImportError:
        return False
    cls, suffix = getattr(lru_cache, 'LRUCache', None), getattr(lru_cache, '_CACHE_SUFFIX', None)
    if cls is None or type(suffix) is not str or not callable(getattr(cls, 'put', None)):
        return False
    if getattr(cls.put, 'atomic', False):
        return True
    original = cls.put

    def put(self, key, value):
        if getattr(self, 'eviction_enabled', True):   # the size-limited cache: JAX's own put, under JAX's own lock
            return original(self, key, value)
        if not key:
            raise ValueError('key cannot be empty')
        path = self.path / f'{key}{suffix}'
        if path.exists():
            return
        temporary = self.path / f'.{key}{suffix}.{os.getpid()}.{threading.get_ident()}.tmp'
        temporary.write_bytes(value)
        os.replace(temporary, path)

    put.atomic, put.original = True, original
    cls.put = put
    return True
