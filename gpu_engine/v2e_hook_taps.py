"""Hook-level taps of the oracle step and the `hook:` harness provider (design note section 2.2, "Hook-level references").

transition.step_with_chance calls the public-tracker hooks through the module attribute `device_public.<name>(...)`
(engine_device_public_hooks), the hooks call each other and their callees through module globals, and the mechanics
(secondary.py) call hooks the same way. At trace time this module replaces those module attributes with recording
wrappers (restored in a finally): each call records its call site (the caller frame's module and line, `<module>:<line>`,
with an occurrence number when the site is reached more than once per step), the carry before the call, the hook's other
arguments and the carry after the call. The recorded values are tracers of the oracle's own trace, so returning them
as extra outputs makes every call site a teacher-forced unit test on real data; nothing in the oracle changes.

Provider for gpu_engine.v2e_diff: `hook:<name>@<site>` where <name> is a hook name (v2e_hooks: `observe`, `_write`,
`port.condition`, ...; several names joined by commas share one trace) and <site> is `all`, a transition.py line, or
`<module>:<line>`. jax_fn returns the oracle's carry after each selected call, keyed `<name>@<site>/<leaf>`; cuda_fn
feeds the recorded (carry before, args) to v2e_hooks.CUDA_HOOKS[name] (or the active self-test) and keys its result the
same way. Only the leaves the oracle hook wrote (a new array object after the call) are returned per site; the other
leaves are summarized as one per-lane bool `<name>@<site>/untouched_equal`, bitwise equality of the CUDA result with the
input (the oracle side is all true), so a hook that writes where it should not is still caught.

`python -m gpu_engine.v2e_hook_taps sites ...` traces one step at the template and writes hook_sites.json/.md: per hook,
its call sites on the device path and the call count per step.
"""
import argparse
from contextlib import contextmanager
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

# Modules whose functions are tappable, by the alias the hooks module uses for them; '' is the hooks module itself.
TAP_MODULES = {'': 'gpu_engine.engine_device_public_hooks', 'trackers': 'gpu_engine.engine_device_trackers',
    'port': 'gpu_engine.engine_device_public_port', 'visible_belief': 'gpu_engine.engine_device_visible_belief',
    'negative_items': 'gpu_engine.engine_device_negative_items', 'damage_history': 'gpu_engine.engine_device_damage_history'}
# Modules whose globals may bind a tapped function under any name (the wrapper replaces every binding of the same object).
SCAN_MODULES = tuple(TAP_MODULES.values()) + ('gpu_engine.transition', 'gpu_engine.secondary')
_lock = threading.RLock()
_installed = {}   # (module name, attribute) -> original, while taps are installed


class Call:
    __slots__ = ('name', 'site', 'occurrence', 'before', 'args', 'kwargs', 'after', 'context_first', 'order')
    def __init__(self, **values):
        for k, v in values.items():setattr(self, k, v)
    @property
    def key(self):
        return f'{self.name}@{self.site}' + (f'#{self.occurrence}' if self.occurrence is not None else '')


def split(name):
    """('alias', 'function') of a hook name: 'observe' -> ('', 'observe'); 'port.condition' -> ('port', 'condition')."""
    alias, _, function = name.rpartition('.')
    if alias not in TAP_MODULES:raise KeyError(f'unknown hook module alias {alias!r} in {name!r}; one of {sorted(TAP_MODULES)}')
    return alias, function


def module_of(name):
    return importlib.import_module(TAP_MODULES[split(name)[0]])


def original(name):
    """The unwrapped oracle function of a hook name (the installed wrapper's original while taps are active)."""
    alias, function = split(name);module = module_of(name)
    key = (TAP_MODULES[alias], function)
    if key in _installed:return _installed[key]
    fn = getattr(module, function, None)
    if not callable(fn) or inspect.isclass(fn):raise KeyError(f'{name!r} is not a function of {TAP_MODULES[alias]}')
    return fn


def context_first(name):
    """Whether the hook's first parameter is the Context `c` (its carry is c.trackers); False for carry-first functions."""
    params = list(inspect.signature(original(name)).parameters)
    return bool(params) and params[0] == 'c'


def tappable():
    """Every tappable hook name: the hooks module's functions, then the callee modules' functions as '<alias>.<name>'."""
    names = []
    for alias, module_name in TAP_MODULES.items():
        module = importlib.import_module(module_name)
        for attr, value in sorted(vars(module).items()):
            if inspect.isfunction(value) and value.__module__ == module.__name__:names.append(f'{alias}.{attr}' if alias else attr)
    return names


def _site(frame):
    return f'{Path(frame.f_code.co_filename).stem}:{frame.f_lineno}'


def _carry(value, is_context):
    return value.trackers if is_context else value


@contextmanager
def tap(names):
    """Install recording wrappers for the hook names (every module-global binding of each function); yields the list the
    calls are appended to, in call order; restores the originals on exit."""
    from gpu_engine.engine_device_public_hooks import Context
    calls = [];counts = {}
    with _lock:
        if _installed:raise RuntimeError('hook taps are already installed')
        bindings = []
        try:
            for name in names:
                fn = original(name);alias, function = split(name)
                key = (TAP_MODULES[alias], function)
                def wrapper(*args, __name=name, __fn=fn, **kwargs):
                    frame = sys._getframe(1);site = _site(frame)
                    is_context = bool(args) and isinstance(args[0], Context)
                    before = _carry(args[0], is_context) if args else None
                    result = __fn(*args, **kwargs)
                    after = _carry(result, is_context)
                    k = (__name, site);occurrence = counts.get(k, 0);counts[k] = occurrence + 1
                    calls.append(Call(name=__name, site=site, occurrence=occurrence, before=before, args=args[1:], kwargs=kwargs, after=after,
                        context_first=is_context, order=len(calls)))
                    return result
                functools.update_wrapper(wrapper, fn)
                _installed[key] = fn
                for module_name in SCAN_MODULES:
                    module = sys.modules.get(module_name) or importlib.import_module(module_name)
                    for attr, value in list(vars(module).items()):
                        if value is fn:setattr(module, attr, wrapper);bindings.append((module, attr, fn))
            yield calls
        finally:
            for module, attr, fn in bindings:setattr(module, attr, fn)
            _installed.clear()
    # Sites hit once per step carry no occurrence suffix.
    for call in calls:
        if counts[(call.name, call.site)] == 1:call.occurrence = None


def parse_argument(argument):
    """'observe@all' -> (['observe'], 'all'); 'observe,hp_line@756' -> (['observe', 'hp_line'], 'transition:756')."""
    names, sep, site = argument.partition('@')
    if not names:raise ValueError(f'hook: needs <name>@<site>, got {argument!r}')
    site = site or 'all'
    if site != 'all' and ':' not in site:site = f'transition:{site}'
    return [n.strip() for n in names.split(',') if n.strip()], site


def selected(call, site):
    return site == 'all' or call.site == site


def written_paths(before, after):
    """Leaf key paths of `after` whose array object differs from `before`'s (the leaves the hook wrote); None when the two
    pytrees differ in structure (then the whole result is compared)."""
    import jax
    if before is None:return None
    try:
        fb, tb = jax.tree_util.tree_flatten_with_path(before);fa, ta = jax.tree_util.tree_flatten_with_path(after)
    except Exception:return None
    if tb != ta:return None
    return [jax.tree_util.keystr(path) for (path, x), (_, y) in zip(fb, fa) if x is not y]


def site_outputs(prefix, before, after, written, *, oracle):
    """{'<prefix>/<leaf>': array} for the written leaves, plus '<prefix>/untouched_equal' (per-lane bool) when some
    leaves were not written: all true for the oracle, bitwise equality of `after` with `before` otherwise."""
    import jax
    import jax.numpy as j
    from gpu_engine.v2e_diff import unsigned
    out = {}
    if written is None:
        for path, x in jax.tree_util.tree_flatten_with_path(after)[0]:out[f'{prefix}/{jax.tree_util.keystr(path) or "value"}'] = x
        return out
    leaves_before = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(before)[0]}
    leaves_after = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(after)[0]}
    for path in written:out[f'{prefix}/{path}'] = leaves_after[path]
    untouched = [p for p in leaves_after if p not in written]
    if untouched:
        b = next(iter(leaves_after.values())).shape[0]
        if oracle:equal = j.ones((b,), bool)
        else:
            equal = j.ones((b,), bool)
            for path in untouched:
                x, y = leaves_before[path], leaves_after[path]
                if x.shape != y.shape or x.dtype != y.dtype:raise ValueError(f'{prefix}: CUDA hook changed the shape or dtype of {path}')
                d = unsigned(x) != unsigned(y)
                equal &= ~j.any(d.reshape(b, -1), axis=1)
        out[f'{prefix}/untouched_equal'] = equal
    return out


def diff_hook(argument, ctx):
    """v2e_diff provider: (jax_fn, cuda_fn) over (state, actions, trackers, context) for 'hook:<name>[,<name>]@<site>'."""
    from gpu_engine import v2e_hooks, v2e_modules
    names, site = parse_argument(argument)
    for name in names:original(name)   # fail early on unknown names
    hooks = {name: v2e_hooks.resolve(name) for name in names}
    step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    tables = v2e_hooks.HookTables(ctx['engine_tables'], getattr(ctx['observation_tables'], 'belief', None), getattr(ctx['engine_tables'], 'public_port', None))
    def tapped(state, actions, trackers):
        with tap(names) as calls:step(state, actions, trackers)
        chosen = [c for c in calls if selected(c, site)]
        if not chosen:
            sites = sorted({f'{c.name}@{c.site}' for c in calls})
            raise ValueError(f'hook:{argument}: no call of {names} at {site!r} on the device path; sites reached: {sites}')
        return chosen
    def jax_fn(state, actions, trackers, context):
        out = {}
        for call in tapped(state, actions, trackers):
            out.update(site_outputs(call.key, call.before, call.after, written_paths(call.before, call.after), oracle=True))
        return out
    def cuda_fn(state, actions, trackers, context):
        out = {}
        for call in tapped(state, actions, trackers):
            got = hooks[call.name](call.before, *call.args, tables=tables, **call.kwargs)
            out.update(site_outputs(call.key, call.before, got, written_paths(call.before, call.after), oracle=False))
        return out
    return jax_fn, cuda_fn


# ---------------------------------------------------------------------------
# Sites report.

def trace_sites(ctx, names):
    """Trace one step at the template under taps for `names`; returns ({hook: {site: calls per step}}, seconds)."""
    import jax
    from gpu_engine import v2e_modules
    step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables']);state, actions, trackers = ctx['template']
    begin = time.perf_counter();record = {}
    def traced(s, a, t):
        with tap(names) as calls:out = step(s, a, t)
        record['calls'] = [(c.name, c.site, c.occurrence, c.context_first, written_paths(c.before, c.after)) for c in calls]
        return out
    jax.eval_shape(traced, state, actions, trackers)
    sites = {}
    for name, site, _, context_first, written in record['calls']:
        row = sites.setdefault(name, {})
        entry = row.setdefault(site, dict(calls=0, context_first=context_first, written=set()))
        entry['calls'] += 1
        if written is not None:entry['written'].update(written)
    for row in sites.values():
        for entry in row.values():entry['written'] = sorted(entry['written'])
    return sites, time.perf_counter() - begin


def sites_markdown(sites, m1):
    out = ['| Hook | M1 | Call sites | Calls per step | Sites |', '| --- | --- | ---: | ---: | --- |']
    for name in sorted(sites, key=lambda n: (-sum(e['calls'] for e in sites[n].values()), n)):
        row = sites[name];calls = sum(e['calls'] for e in row.values())
        where = ', '.join(f"{site}{'' if e['calls'] == 1 else ' x' + str(e['calls'])}" for site, e in sorted(row.items(), key=lambda kv: (kv[0].split(':')[0], int(kv[0].split(':')[1]))))
        out.append(f"| {name} | {'yes' if name in m1 else ''} | {len(row)} | {calls} | {where} |")
    return '\n'.join(out) + '\n'


def cmd_sites(args):
    from gpu_engine import v2e_census as cz, v2e_hooks
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    ctx = cz.load_context(args)
    names = tappable() if args.hooks == ['all'] else (args.hooks or list(v2e_hooks.ALL_HOOKS))
    sites, seconds = trace_sites(ctx, names)
    missing = [n for n in v2e_hooks.M1_HOOKS if n not in sites]
    report = dict(schema='pokezero.v2e-hook-sites.v1', source_commit=os.environ.get('V2_SOURCE_COMMIT'), seconds=seconds, hooks_tapped=names,
        hooks_called=len(sites), calls_per_step=sum(e['calls'] for row in sites.values() for e in row.values()),
        m1_hooks=list(v2e_hooks.M1_HOOKS), m1_hooks_not_called=missing, sites=sites)
    (out / 'hook_sites.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'hook_sites.md').write_text(sites_markdown(sites, set(v2e_hooks.M1_HOOKS)))
    print(f"sites: {len(sites)} hooks called, {report['calls_per_step']} calls per step, M1 hooks not called: {missing} -> {out / 'hook_sites.json'}")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]);sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('sites', help='call sites and calls per step of every tappable hook (one traced step at the template)')
    q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True);q.add_argument('--arm', default='B', choices=('A', 'B'))
    q.add_argument('--out', required=True);q.add_argument('--hooks', nargs='*', help='hook names (default: v2e_hooks.ALL_HOOKS; "all": every tappable function)')
    q.set_defaults(fn=cmd_sites)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':raise SystemExit(main())
