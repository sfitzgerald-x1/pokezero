"""JAX's persistent compilation cache, written atomically by this process.

JAX's file cache (jax._src.lru_cache.LRUCache; without a size limit, max_size -1, it takes no lock) writes an entry in place
(Path.write_bytes). A reader in another process that finds the file before that write ends reads a truncated entry: JAX warns
"Error reading persistent compilation cache entry" and compiles again. A writer killed mid-write leaves the truncated entry for
good, since put never overwrites an existing one. Several processes share one cache directory here (a server and its prepare
workers, two servers on one GPU, pods on one shared volume), so install() makes this process's put write a temporary file in
the cache directory (tempfile.mkstemp: a unique name on any host), fsync it and rename it into place (os.replace: atomic within
one directory): a reader sees a whole entry or none, and a failed write leaves nothing behind (its temporary is unlinked).
Nothing else changes: the key, the bytes, get(), put's no-overwrite rule. JAX's own put stays for a size-limited cache (eviction
on, under JAX's own file lock), for a cache that is not a local filesystem path (gs://...) and for a cache whose two flags are
not the booleans this patch reads (a JAX that changed them); STATUS counts the writes each way. Entries get JAX's own mode (0666
under the umask), not mkstemp's 0600. Idempotent. A JAX whose cache
does not have the shape patched here is left alone: install() returns False (STATUS says why), and callers that rely on whole
entries (depthn_live_server with --compilation-cache) refuse to start."""
import os
import tempfile

STATUS = dict(installed=False, reason='not installed', jax=None, atomic_writes=0, jax_writes=0)


def install():
    """Patch LRUCache.put in this process (True), or leave an unknown JAX alone (False). STATUS records the outcome."""
    try:
        import jax
        from jax._src import lru_cache
    except ImportError as error:
        STATUS.update(installed=False, reason=f'jax not importable: {error}');return False
    STATUS['jax'] = jax.__version__
    cls, suffix = getattr(lru_cache, 'LRUCache', None), getattr(lru_cache, '_CACHE_SUFFIX', None)
    if cls is None or type(suffix) is not str or not callable(getattr(cls, 'put', None)):
        STATUS.update(installed=False, reason='jax._src.lru_cache has no LRUCache.put or _CACHE_SUFFIX of the patched shape');return False
    if getattr(cls.put, 'atomic', False):
        STATUS.update(installed=True, reason='installed');return True
    original = cls.put;umask = os.umask(0);os.umask(umask);mode = 0o666 & ~umask   # the mode JAX's own write_bytes gives a file

    def put(self, key, value):
        local, eviction = getattr(self, '_is_local', None), getattr(self, 'eviction_enabled', None)
        if eviction is not False or local is not True:   # JAX's own: its lock, a remote path, or a cache this patch does not know
            STATUS['jax_writes'] += 1
            return original(self, key, value)
        if not key:
            raise ValueError('key cannot be empty')
        path = self.path / f'{key}{suffix}'
        if path.exists():
            return
        handle, temporary = tempfile.mkstemp(dir=os.fspath(self.path), prefix=f'.{key}{suffix}.', suffix='.tmp')
        try:
            os.fchmod(handle, mode)   # mkstemp makes 0600: readable by other users as JAX's own entries are
            with os.fdopen(handle, 'wb') as f:
                f.write(value);f.flush();os.fsync(f.fileno())
            os.replace(temporary, path);STATUS['atomic_writes'] += 1
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    put.atomic, put.original = True, original
    cls.put = put
    STATUS.update(installed=True, reason='installed')
    return True
