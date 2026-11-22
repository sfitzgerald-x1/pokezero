"""One admission clock and short process-wide automatic-GC deferral."""
import gc
from threading import Lock

_lock = Lock()
_acquire, _release = _lock.acquire, _lock.release
_users = 0
_was_enabled = False


class _GCLease:
    __slots__ = ('closed',)

    def __init__(self): self.closed = False
    def __enter__(self): return self
    def __exit__(self, *exception):
        global _users
        _acquire()
        try:
            if self.closed: return False
            self.closed = True
            _users -= 1
            if not _users:
                (gc.enable if _was_enabled else gc.disable)()
        finally:
            _release()
        return False


def defer_joint_gc():
    """Restore prior automatic-GC state after all overlapping users exit.

    Disable before allocating the lease itself, and avoid a generator's
    StopIteration allocation after restoration. No collection is forced.
    Explicit external collections remain observable timing failures.
    """
    global _users, _was_enabled
    _acquire()
    try:
        if not _users:
            _was_enabled = gc.isenabled()
            if _was_enabled: gc.disable()
        _users += 1
    finally:
        _release()
    try:
        return _GCLease()
    except BaseException:
        _acquire()
        try:
            _users -= 1
            if not _users: (gc.enable if _was_enabled else gc.disable)()
        finally:
            _release()
        raise


def new_joint_deadline(seconds=.25):
    from pokezero_search import JointDeadline
    return JointDeadline(seconds)


def check_deadline(deadline, cutoff=None):
    if deadline is not None and (deadline.remaining_seconds() <= 0 or
            (cutoff is not None and deadline.elapsed_seconds() >= cutoff)):
        raise TimeoutError('joint deadline: Python phase exhausted')
