"""Registry of the hand-written CUDA tracker hooks (M1 fills it) and the signature the hook: harness calls.

CUDA_HOOKS maps a hook name to a callable with the signature

    fn(carry, *args, tables, **kwargs) -> carry

where everything after `carry` is exactly what the oracle hook received after its first argument, at the tapped call
site, teacher-forced (gpu_engine.v2e_hook_taps records it from the oracle's own trace):
- `carry` is the oracle's first argument with the Context peeled off: for the engine_device_public_hooks functions that
  take `c` (observe, hp_line, feature, _refresh, _narrow, reveal_*, move, ...), the Trackers NamedTuple `c.trackers`
  before the call; for carry-first functions (_write, trackers._observe, port.condition, visible_belief.refresh,
  negative_items.hp_line) the Trackers as passed; for damage_history.* the History sub-carry (or the pair array for
  damage_history._add). The hook returns the carry after the call, in the same pytree structure.
- `*args` / `**kwargs` are the remaining oracle arguments as received: pytrees of per-lane arrays (`state`, masks,
  values; masks are per-lane bool, or a Python bool / scalar the oracle broadcasts, so broadcast them to [B] before
  the FFI call) and Python statics (`side` int, feature `name` str, `source_class` int, `add`/`pp_use`/`start` flags,
  the EntryTable / MoveEffects table pytrees). A CUDA implementation turns the statics into FFI attributes.
- `tables` is HookTables(engine, belief, port): the engine tables (transition.MoveTable), observation_tables.belief
  (the Context's belief_tables) and engine_tables.public_port (its port_tables). Pack them once with
  gpu_engine.v2e_layout.pack_tables for the EngineTables buffer of gpu_engine/cuda/v2e_layout.cuh.
Hook names are the function names of engine_device_public_hooks (`observe`, `_write`, ...) or `<alias>.<function>`
for the callees the hooks reach through a module global (`trackers._observe`, `port.condition`,
`visible_belief.refresh`, `negative_items.hp_line`, `damage_history._add`, ...); see v2e_hook_taps.TAP_MODULES.

This module holds no CUDA. Two self-test registrations exist for the harness itself, never active by default; select one
with the environment variable V2E_HOOKS_SELFTEST or selftest(mode):
- 'oracle-as-cuda': every hook is the JAX oracle hook itself; hook:<name>@all must then show 0 differing lane-steps;
- 'negative': the oracle hook with one bit of its first output leaf flipped at one lane; it must be detected.
"""
from collections import namedtuple
import os

HookTables = namedtuple('HookTables', 'engine belief port')

# The M1 hook library (design note section 6, M1 brief): every name must be bitwise at every device call site.
M1_HOOKS = ('observe', 'hp_line', 'feature', '_write', '_refresh', '_narrow', 'reveal_ability', 'reveal_item', 'move', 'item_changes',
    'party_cure', 'damage', 'negative_hp_line', 'color_change', 'public_move_state', 'weather', 'wish', 'before_move', 'volatile',
    'port_count', 'port_protect')
# The switch family (M3) and the turn boundary (M2), tappable the same way.
M3_HOOKS = ('switch_decision', 'install', 'baton_copy', 'withdraw', 'depart', 'entry', 'drag')
M2_HOOKS = ('turn_end', 'public_turn')
# Callees the hooks reach through a module global (M1's list), as '<alias>.<function>' (aliases: v2e_hook_taps.TAP_MODULES).
CALLEES = ('trackers._observe', 'trackers._percent', 'port.condition', 'visible_belief.refresh', 'visible_belief.exclude',
    'negative_items.hp_line', 'damage_history.observe', 'damage_history._add', 'damage_history.settle')
ALL_HOOKS = M1_HOOKS + M3_HOOKS + M2_HOOKS + ('opportunity', 'matchup') + CALLEES

SELFTEST_ENV = 'V2E_HOOKS_SELFTEST'
SELFTESTS = ('oracle-as-cuda', 'negative')
NEGATIVE_LANE = 3   # the lane whose first output leaf gets one bit flipped (lane 0 when the batch is smaller)

CUDA_HOOKS = {}
_selftest = {'mode': None}


def selftest(mode):
    """Select a harness self-test ('oracle-as-cuda' or 'negative'); None disables. Returns the previous mode."""
    if mode is not None and mode not in SELFTESTS:raise ValueError(f'unknown self-test {mode!r}; one of {SELFTESTS}')
    previous = _selftest['mode'];_selftest['mode'] = mode
    return previous


def selftest_mode():
    """The active self-test: selftest(mode) if set, else the V2E_HOOKS_SELFTEST environment variable, else None."""
    mode = _selftest['mode'] or os.environ.get(SELFTEST_ENV) or None
    if mode is not None and mode not in SELFTESTS:raise ValueError(f'{SELFTEST_ENV}={mode!r}: one of {SELFTESTS}')
    return mode


def resolve(name):
    """The callable for a hook name: CUDA_HOOKS[name], or the active self-test's stand-in; KeyError when neither exists."""
    if name in CUDA_HOOKS:return CUDA_HOOKS[name]
    mode = selftest_mode()
    if mode is None:raise KeyError(f'no CUDA hook registered for {name!r} (CUDA_HOOKS has {sorted(CUDA_HOOKS)}); set {SELFTEST_ENV}=oracle-as-cuda for the harness self-test')
    return selftest_hook(name, mode)


def _flip_one_bit(tree):
    """The pytree with bit 0 of the first leaf's first element at NEGATIVE_LANE flipped (any dtype), the rest untouched."""
    import jax
    import jax.numpy as j
    from gpu_engine.v2e_diff import unsigned
    leaves, treedef = jax.tree.flatten(tree)
    x = leaves[0];b = x.shape[0];lane = NEGATIVE_LANE if b > NEGATIVE_LANE else 0
    flat = x.reshape(b, -1)
    if x.dtype == j.bool_:flipped = flat.at[lane, 0].set(~flat[lane, 0])
    else:
        u = unsigned(flat);flipped = u.at[lane, 0].set(u[lane, 0] ^ 1)
        flipped = flipped if flipped.dtype == x.dtype else jax.lax.bitcast_convert_type(flipped, x.dtype)
    leaves[0] = flipped.reshape(x.shape)
    return jax.tree.unflatten(treedef, leaves)


def selftest_hook(name, mode):
    """The self-test stand-in for `name`: the oracle hook (with a Context rebuilt around the carry), optionally flipped."""
    from gpu_engine import v2e_hook_taps as taps
    original = taps.original(name);context_first = taps.context_first(name)
    def fn(carry, *args, tables, **kwargs):
        if context_first:
            from gpu_engine.engine_device_public_hooks import Context
            out = original(Context(carry, tables.belief, tables.port), *args, **kwargs).trackers
        else:out = original(carry, *args, **kwargs)
        return _flip_one_bit(out) if mode == 'negative' else out
    fn.__name__ = f'{mode}:{name}'
    return fn
