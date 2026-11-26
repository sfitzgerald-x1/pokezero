"""Registry of the hand-written CUDA tracker hooks (M1 fills it) and the signature the hook: harness calls.

CUDA_HOOKS maps a hook name to a callable with the signature

    fn(carry, *args, tables, **kwargs) -> carry            or    -> (carry, err)

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
  the EntryTable / MoveEffects table pytrees). A CUDA implementation turns the statics into FFI attributes. The
  harness groups call sites by their static-argument signature (v2e_hook_taps signatures): M1's definition of done
  needs every signature exercised (a lane where the oracle hook changed a leaf) on real or synthetic lanes.
- `tables` is HookTables(engine, belief, port): the engine tables (transition.MoveTable), observation_tables.belief
  (the Context's belief_tables) and engine_tables.public_port (its port_tables). Pack them once with
  gpu_engine.v2e_layout.pack_tables for the EngineTables buffer of gpu_engine/cuda/v2e_layout.cuh. The handler's
  `layout_digest` attribute is gpu_engine.v2e_layout.digest_attr(live_digests(state, actions, trackers, tables.engine,
  tables.belief)['kLayoutDigest'][0]), computed from the pytrees in hand, and the handler compares it with the header's
  kLayoutDigest, returning kStaleLayout in `err` when they differ.
- `err`, optional, is the handler's error word: uint32 of shape [1] (one word for the call) or [B] (per lane). The
  harness emits it as `<name>@<site>/err` and compares it with zeros, so a stale-layout refusal (v2e_layout.cuh
  kStaleLayout) or a debug clamp/drop count shows as a difference at its site. A hook that returns the carry alone
  reports zeros.
Hook names are the function names of engine_device_public_hooks (`observe`, `_write`, ...) or `<alias>.<function>`
for the callees the hooks reach through a module global (`trackers._observe`, `port.condition`,
`visible_belief.refresh`, `negative_items.hp_line`, `damage_history._add`, ...); see v2e_hook_taps.TAP_MODULES.

This module holds no CUDA. In normal mode a hook missing from CUDA_HOOKS is a hard error (KeyError), never a fallback.
Self-tests of the harness itself are selected only by the v2e_diff flag `--hooks-selftest {oracle,negative,
negative-untouched}` (or selftest(mode) in code); in self-test mode every hook resolves to the stand-in, registered or
not, and diff.json records per hook the implementation that ran, the mode and where it was selected (`provider`):
- 'oracle': every hook is the JAX oracle hook itself; hook:<name>@all must then show 0 differing lane-steps;
- 'negative': the oracle hook with one bit of its first written leaf flipped at one lane; it must be detected;
- 'negative-untouched': one bit of a leaf the hook does not write flipped at one lane; the untouched_equal check must
  detect it.
"""
from collections import namedtuple

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

SELFTESTS = ('oracle', 'negative', 'negative-untouched')
ALIASES = {'oracle-as-cuda': 'oracle'}
NEGATIVE_LANE = 3   # the lane whose leaf gets one bit flipped (lane 0 when the batch is smaller)

CUDA_HOOKS = {}
_selftest = {'mode': None, 'source': None}


def selftest(mode, source='selftest()'):
    """Select a harness self-test (SELFTESTS; None disables) and record where it was selected. Returns the previous mode.
    In self-test mode every hook resolves to the stand-in, registered or not (the result records which ran)."""
    mode = ALIASES.get(mode, mode)
    if mode is not None and mode not in SELFTESTS:raise ValueError(f'unknown self-test {mode!r}; one of {SELFTESTS}')
    previous = _selftest['mode'];_selftest.update(mode=mode, source=source if mode is not None else None)
    return previous


def selftest_mode():
    """The active self-test mode (None in normal mode)."""
    return _selftest['mode']


def resolve(name):
    """The callable for a hook name: CUDA_HOOKS[name] in normal mode (KeyError when missing: never a fallback); in
    self-test mode the stand-in, whether or not the hook is registered."""
    mode = selftest_mode()
    if mode is None:
        if name not in CUDA_HOOKS:raise KeyError(f'no CUDA hook registered for {name!r} (CUDA_HOOKS has {sorted(CUDA_HOOKS)})')
        return CUDA_HOOKS[name]
    return selftest_hook(name, mode)


def implementation(name):
    """What resolve(name) returns, as a record for diff.json: the implementation's module and qualified name, the
    self-test mode and where it was selected."""
    fn = resolve(name)
    return dict(hook=name, implementation=f'{getattr(fn, "__module__", "?")}.{getattr(fn, "__qualname__", getattr(fn, "__name__", "?"))}',
        selftest=_selftest['mode'], selftest_source=_selftest['source'], registered=name in CUDA_HOOKS)


def _flip_one_bit(tree, index):
    """The pytree with bit 0 of leaf `index`'s first element at NEGATIVE_LANE flipped (any dtype), the rest untouched."""
    import jax
    import jax.numpy as j
    from gpu_engine.v2e_diff import unsigned
    leaves, treedef = jax.tree.flatten(tree)
    x = leaves[index];b = x.shape[0];lane = NEGATIVE_LANE if b > NEGATIVE_LANE else 0
    flat = x.reshape(b, -1)
    if x.dtype == j.bool_:flipped = flat.at[lane, 0].set(~flat[lane, 0])
    else:
        u = unsigned(flat);flipped = u.at[lane, 0].set(u[lane, 0] ^ 1)
        flipped = flipped if flipped.dtype == x.dtype else jax.lax.bitcast_convert_type(flipped, x.dtype)
    leaves[index] = flipped.reshape(x.shape)
    return jax.tree.unflatten(treedef, leaves)


def selftest_hook(name, mode):
    """The self-test stand-in for `name`: the oracle hook (with a Context rebuilt around the carry), optionally with one
    bit flipped in its first written leaf ('negative') or its first untouched leaf ('negative-untouched')."""
    import jax
    from gpu_engine import v2e_hook_taps as taps
    original = taps.original(name);context_first = taps.context_first(name)
    def fn(carry, *args, tables, **kwargs):
        if context_first:
            from gpu_engine.engine_device_public_hooks import Context
            out = original(Context(carry, tables.belief, tables.port), *args, **kwargs).trackers
        else:out = original(carry, *args, **kwargs)
        if mode == 'oracle':return out
        before = jax.tree.leaves(carry) if jax.tree.structure(carry) == jax.tree.structure(out) else []
        after = jax.tree.leaves(out)
        written = [k for k, (x, y) in enumerate(zip(before, after)) if x is not y]
        untouched = [k for k, (x, y) in enumerate(zip(before, after)) if x is y]
        wanted = written if mode == 'negative' else untouched
        return _flip_one_bit(out, wanted[0] if wanted else 0)
    fn.__name__ = fn.__qualname__ = f'selftest_{mode.replace("-", "_")}:{name}'
    return fn
