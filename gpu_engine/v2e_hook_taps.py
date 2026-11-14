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
feeds the recorded (carry before, args) to v2e_hooks.CUDA_HOOKS[name] (or the self-test selected with
`--hooks-selftest`) and keys its result the same way. Only the leaves the oracle hook wrote (a new array object after the
call) are returned per site; the other leaves are summarized as one per-lane bool `<name>@<site>/untouched_equal`,
bitwise equality of the CUDA result with the input (the oracle side is all true), so a hook that writes where it should
not is still caught; `<name>@<site>/err` carries the hook's error word (zeros on the oracle side). A site proves
something only where the oracle hook changed a leaf: diff.json records the exercised lane-steps per site
(`exercised_by_site`, `sites_never_exercised`) and each site's static-argument signature; `signatures` groups the sites
by (hook, signature) and lists the signatures exercised nowhere (M1's synthetic-lane list).

`python -m gpu_engine.v2e_hook_taps sites ...` traces one step at the template and writes hook_sites.json/.md: per hook,
its call sites on the device path, the call count per step and the static signatures.
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
    """One recorded hook call. `order` is the entry order (the slot is reserved when the hook is entered, so an enclosing
    call precedes its nested calls); `step_line` is the step_with_chance line active at the call, or None."""
    __slots__ = ('name', 'site', 'occurrence', 'before', 'args', 'kwargs', 'after', 'context_first', 'order', 'step_line', 'result')
    def __init__(self, **values):
        for k, v in values.items():setattr(self, k, v)
    @property
    def key(self):
        return f'{self.name}@{self.site}' + (f'#{self.occurrence}' if self.occurrence is not None else '')


def split(name, modules=TAP_MODULES):
    """('alias', 'function') of a hook name: 'observe' -> ('', 'observe'); 'port.condition' -> ('port', 'condition')."""
    alias, _, function = name.rpartition('.')
    if alias not in modules:raise KeyError(f'unknown hook module alias {alias!r} in {name!r}; one of {sorted(modules)}')
    return alias, function


def module_of(name, modules=TAP_MODULES):
    return importlib.import_module(modules[split(name, modules)[0]])


def original(name, modules=TAP_MODULES):
    """The unwrapped oracle function of a hook name (the installed wrapper's original while taps are active)."""
    alias, function = split(name, modules);module = module_of(name, modules)
    key = (modules[alias], function)
    if key in _installed:return _installed[key]
    fn = getattr(module, function, None)
    if not callable(fn) or inspect.isclass(fn):raise KeyError(f'{name!r} is not a function of {modules[alias]}')
    return fn


def context_first(name, modules=TAP_MODULES):
    """Whether the hook's first parameter is the Context `c` (its carry is c.trackers); False for carry-first functions."""
    params = list(inspect.signature(original(name, modules)).parameters)
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


def _step_line(frame):
    """The line of the innermost transition.step_with_chance frame on the stack, or None."""
    while frame is not None:
        if frame.f_code.co_name == 'step_with_chance' and frame.f_code.co_filename.endswith('transition.py'):return frame.f_lineno
        frame = frame.f_back
    return None


def _carry(value, is_context):
    return value.trackers if is_context else value


@contextmanager
def tap(names, modules=TAP_MODULES, scan=SCAN_MODULES):
    """Install recording wrappers for the hook names (every binding of each function in the `scan` modules); yields the
    list the calls are appended to, in entry order (a call's slot is reserved when it is entered and its result filled
    when it returns); restores the originals in reverse order on exit. Two names resolving to one function, or a
    function that is already wrapped, are refused. `modules` maps the name aliases (v2e_coverage taps the mechanics
    with its own map)."""
    from gpu_engine.engine_device_public_hooks import Context
    calls = [];counts = {}
    with _lock:
        if _installed:raise RuntimeError('hook taps are already installed')
        bindings = [];wrapped = {}
        try:
            for name in names:
                fn = original(name, modules);alias, function = split(name, modules)
                key = (modules[alias], function)
                if getattr(fn, '__v2e_tap__', False):raise ValueError(f'{name!r} resolves to a function that is already wrapped')
                if id(fn) in wrapped:raise ValueError(f'{name!r} and {wrapped[id(fn)]!r} name the same function; tap it under one name')
                wrapped[id(fn)] = name
                def wrapper(*args, __name=name, __fn=fn, **kwargs):
                    frame = sys._getframe(1);site = _site(frame);step_line = _step_line(frame)
                    is_context = bool(args) and isinstance(args[0], Context)
                    before = _carry(args[0], is_context) if args else None
                    k = (__name, site);occurrence = counts.get(k, 0);counts[k] = occurrence + 1
                    call = Call(name=__name, site=site, occurrence=occurrence, before=before, args=args[1:], kwargs=kwargs, after=None,
                        context_first=is_context, order=len(calls), step_line=step_line, result=None)
                    calls.append(call)   # the slot is reserved at entry
                    result = __fn(*args, **kwargs)
                    call.after = _carry(result, is_context);call.result = result
                    return result
                functools.update_wrapper(wrapper, fn);wrapper.__v2e_tap__ = True
                _installed[key] = fn
                for module_name in scan:
                    module = sys.modules.get(module_name) or importlib.import_module(module_name)
                    for attr, value in list(vars(module).items()):
                        if value is fn:setattr(module, attr, wrapper);bindings.append((module, attr, fn))
            yield calls
        finally:
            for module, attr, fn in reversed(bindings):setattr(module, attr, fn)
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


def site_outputs(prefix, before, after, written, *, oracle, err=None):
    """{'<prefix>/<leaf>': array} for the written leaves, '<prefix>/untouched_equal' (per-lane bool) when some leaves
    were not written (all true for the oracle, bitwise equality of `after` with `before` otherwise) and '<prefix>/err'
    (uint32 [B]: the hook's error word broadcast per lane; zeros for the oracle and for hooks that return none)."""
    import jax
    import jax.numpy as j
    from gpu_engine.v2e_diff import unsigned
    out = {}
    if written is None:
        for path, x in jax.tree_util.tree_flatten_with_path(after)[0]:out[f'{prefix}/{jax.tree_util.keystr(path) or "value"}'] = x
        b = jax.tree.leaves(after)[0].shape[0]
    else:
        leaves_before = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(before)[0]}
        leaves_after = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(after)[0]}
        for path in written:out[f'{prefix}/{path}'] = leaves_after[path]
        untouched = [p for p in leaves_after if p not in written]
        b = next(iter(leaves_after.values())).shape[0]
        if untouched:
            equal = j.ones((b,), bool)
            if not oracle:
                for path in untouched:
                    x, y = leaves_before[path], leaves_after[path]
                    if x.shape != y.shape or x.dtype != y.dtype:raise ValueError(f'{prefix}: CUDA hook changed the shape or dtype of {path}')
                    d = unsigned(x) != unsigned(y)
                    equal &= ~j.any(d.reshape(b, -1), axis=1)
            out[f'{prefix}/untouched_equal'] = equal
    out[f'{prefix}/err'] = error_word(err, b)
    return out


def error_word(err, b):
    """A hook's error word as uint32 [B]: None -> zeros; shape [1] broadcast to every lane; shape [B] as is."""
    import jax.numpy as j
    if err is None:return j.zeros((b,), j.uint32)
    err = j.asarray(err)
    if err.dtype != j.uint32 or err.ndim != 1 or err.shape[0] not in (1, b):raise ValueError(f'hook error word must be uint32 [1] or [B], got {err.shape}{err.dtype}')
    return j.broadcast_to(err, (b,))


def split_result(result):
    """(carry, err) of a CUDA hook's return value: a (carry, err) tuple, or the carry alone."""
    if isinstance(result, tuple) and len(result) == 2 and not hasattr(result, '_fields') and hasattr(result[1], 'dtype'):return result
    return result, None


def exercised(before, after, written):
    """Per-lane bool: the oracle hook changed at least one written leaf at that lane (bitwise)."""
    import jax
    import jax.numpy as j
    from gpu_engine.v2e_diff import unsigned
    if written is None:return j.ones((jax.tree.leaves(after)[0].shape[0],), bool)
    leaves_before = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(before)[0]}
    leaves_after = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(after)[0]}
    b = next(iter(leaves_after.values())).shape[0];hit = j.zeros((b,), bool)
    for path in written:
        d = unsigned(leaves_before[path]) != unsigned(leaves_after[path]);hit |= j.any(d.reshape(b, -1), axis=1)
    return hit


STATIC_EXCLUDED = ('side', 'physical')   # indices the hooks are symmetric in (the seat; party_cure's Python slot) are not part of a signature


def _is_static(value):
    if value is None or isinstance(value, (bool, int, float, str)):return True
    return False


def statics(call, modules=TAP_MODULES):
    """{parameter: value} of the call's Python-static arguments (ints, bools, strs, None; not arrays or array pytrees),
    bound to the oracle signature; the carry/Context parameter excluded."""
    fn = original(call.name, modules);sig = inspect.signature(fn)
    try:bound = sig.bind(None, *call.args, **call.kwargs)
    except TypeError:return {}
    bound.apply_defaults();first = next(iter(sig.parameters))
    out = {}
    for name, value in bound.arguments.items():
        if name == first:continue
        param = sig.parameters[name]
        if param.kind == param.VAR_KEYWORD:
            for k, v in value.items():
                if _is_static(v):out[k] = v
        elif _is_static(value):out[name] = value
    return out


def signature(call, modules=TAP_MODULES):
    """'<hook>(k=v, ...)' over the call's statics, STATIC_EXCLUDED parameters left out; the grouping key of the
    signatures report."""
    items = sorted((k, v) for k, v in statics(call, modules).items() if k not in STATIC_EXCLUDED)
    return f"{call.name}({', '.join(f'{k}={v!r}' for k, v in items)})"


def diff_hook(argument, ctx):
    """v2e_diff provider: (jax_fn, cuda_fn) over (state, actions, trackers, context) for 'hook:<name>[,<name>]@<site>'.
    Besides the per-site leaves, both sides return `__exercised__`, int32 [sites]: per selected site, the lanes where the
    oracle hook changed a written leaf (the same oracle-computed vector on both sides, so it never differs); jax_fn's
    accumulate()/summary() give v2e_diff the per-site totals (`exercised_by_site`), and both functions carry `info`
    (the resolved implementation per hook, the self-test mode) for the report."""
    from gpu_engine import v2e_hooks, v2e_modules
    names, site = parse_argument(argument)
    for name in names:original(name)   # fail early on unknown names
    hooks = {name: v2e_hooks.resolve(name) for name in names}
    info = dict(argument=argument, site=site, hooks=[v2e_hooks.implementation(name) for name in names], selftest=v2e_hooks.selftest_mode())
    step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    tables = v2e_hooks.HookTables(ctx['engine_tables'], getattr(ctx['observation_tables'], 'belief', None), getattr(ctx['engine_tables'], 'public_port', None))
    sites = {}
    def tapped(state, actions, trackers):
        import jax.numpy as j
        with tap(names) as calls:step(state, actions, trackers)
        chosen = [c for c in calls if selected(c, site)]
        if not chosen:
            reached = sorted({f'{c.name}@{c.site}' for c in calls})
            raise ValueError(f'hook:{argument}: no call of {names} at {site!r} on the device path; sites reached: {reached}')
        keys = [c.key for c in chosen]
        if sites and sites['keys'] != keys:raise RuntimeError('hook call sites differ between the two traces')
        sites.update(keys=keys, signatures={c.key: signature(c) for c in chosen}, step_lines={c.key: c.step_line for c in chosen}, sides={c.key: statics(c).get('side') for c in chosen})
        info.update(sites=keys)
        hits = j.stack([j.sum(exercised(c.before, c.after, written_paths(c.before, c.after)).astype(j.int32)) for c in chosen])
        return chosen, hits
    def jax_fn(state, actions, trackers, context):
        chosen, hits = tapped(state, actions, trackers);out = {'__exercised__': hits}
        for call in chosen:
            out.update(site_outputs(call.key, call.before, call.after, written_paths(call.before, call.after), oracle=True))
        return out
    def cuda_fn(state, actions, trackers, context):
        chosen, hits = tapped(state, actions, trackers);out = {'__exercised__': hits}
        for call in chosen:
            got, err = split_result(hooks[call.name](call.before, *call.args, tables=tables, **call.kwargs))
            out.update(site_outputs(call.key, call.before, got, written_paths(call.before, call.after), oracle=False, err=err))
        return out
    totals = {'lane_steps': 0, 'by_site': {}}
    def accumulate(want, lanes):
        hits = np.asarray(want['__exercised__']);totals['lane_steps'] += int(lanes)
        for key, n in zip(sites['keys'], hits.tolist()):totals['by_site'][key] = totals['by_site'].get(key, 0) + int(n)
    def summary():
        by_site = totals['by_site']
        return dict(exercised_lane_steps_total=totals['lane_steps'], exercised_by_site=by_site,
            sites_never_exercised=sorted(k for k, n in by_site.items() if n == 0), signatures=sites.get('signatures'), step_lines=sites.get('step_lines'), sides=sites.get('sides'))
    jax_fn.info = cuda_fn.info = info;jax_fn.accumulate = accumulate;jax_fn.summary = summary
    return jax_fn, cuda_fn


# ---------------------------------------------------------------------------
# Sites report.

def trace_sites(ctx, names):
    """Trace one step at the template under taps for `names`; returns ({hook: {site: calls per step, ...}}, seconds)."""
    import jax
    from gpu_engine import v2e_modules
    step = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables']);state, actions, trackers = ctx['template']
    begin = time.perf_counter();record = {}
    def traced(s, a, t):
        with tap(names) as calls:out = step(s, a, t)
        record['calls'] = [(c.name, c.site, c.key, c.context_first, written_paths(c.before, c.after), signature(c), c.step_line, statics(c)) for c in calls]
        return out
    jax.eval_shape(traced, state, actions, trackers)
    sites = {}
    for name, site, key, context_first, written, sig, step_line, static in record['calls']:
        row = sites.setdefault(name, {})
        entry = row.setdefault(site, dict(calls=0, context_first=context_first, written=set(), signatures=set(), keys=[], step_lines=set(), statics=[], key_statics={}))
        entry['calls'] += 1;entry['signatures'].add(sig);entry['keys'].append(key);entry['step_lines'].add(step_line);entry['key_statics'][key] = static
        if static not in entry['statics']:entry['statics'].append(static)
        if written is not None:entry['written'].update(written)
    for row in sites.values():
        for entry in row.values():
            entry['written'] = sorted(entry['written']);entry['signatures'] = sorted(entry['signatures']);entry['step_lines'] = sorted(x for x in entry['step_lines'] if x is not None)
    return sites, time.perf_counter() - begin


def sites_markdown(sites, m1):
    out = ['| Hook | M1 | Call sites | Calls per step | Signatures | Sites |', '| --- | --- | ---: | ---: | ---: | --- |']
    for name in sorted(sites, key=lambda n: (-sum(e['calls'] for e in sites[n].values()), n)):
        row = sites[name];calls = sum(e['calls'] for e in row.values());signatures = {s for e in row.values() for s in e['signatures']}
        where = ', '.join(f"{site}{'' if e['calls'] == 1 else ' x' + str(e['calls'])}" for site, e in sorted(row.items(), key=lambda kv: (kv[0].split(':')[0], int(kv[0].split(':')[1]))))
        out.append(f"| {name} | {'yes' if name in m1 else ''} | {len(row)} | {calls} | {len(signatures)} | {where} |")
    return '\n'.join(out) + '\n'


def cmd_sites(args):
    from gpu_engine import v2e_census as cz, v2e_hooks
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    ctx = cz.load_context(args)
    names = tappable() if args.hooks == ['all'] else (args.hooks or list(v2e_hooks.ALL_HOOKS))
    sites, seconds = trace_sites(ctx, names)
    missing = [n for n in v2e_hooks.M1_HOOKS if n not in sites]
    report = dict(schema='pokezero.v2e-hook-sites.v2', source_commit=os.environ.get('V2_SOURCE_COMMIT'), seconds=seconds, hooks_tapped=names,
        hooks_called=len(sites), calls_per_step=sum(e['calls'] for row in sites.values() for e in row.values()),
        m1_hooks=list(v2e_hooks.M1_HOOKS), m1_hooks_not_called=missing, sites=sites)
    (out / 'hook_sites.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'hook_sites.md').write_text(sites_markdown(sites, set(v2e_hooks.M1_HOOKS)))
    print(f"sites: {len(sites)} hooks called, {report['calls_per_step']} calls per step, M1 hooks not called: {missing} -> {out / 'hook_sites.json'}")
    return 0


# ---------------------------------------------------------------------------
# Signatures report: sites grouped by (hook, static-argument signature) with the exercised lane-steps of the hook: gates.

def normalize_signature(sig):
    """A recorded signature string with the STATIC_EXCLUDED parameters removed (reports over older diff.json files)."""
    import re
    for name in STATIC_EXCLUDED:sig = re.sub(rf"(?:, )?\b{name}=[^,()]*", '', sig)
    return sig.replace('(, ', '(')


def signatures_report(diffs, sites=None):
    """Group every hook: result's sites by (hook, signature) and by (signature, side) with exercised lane-step totals.
    `diffs` are loaded diff.json reports (their results carry `signatures`, `exercised_by_site`, `sides`); `sites` is an
    optional hook_sites.json whose per-key statics supply the side for older diff.json files without `sides`."""
    side_of_key = {}
    for row in (sites or {}).get('sites', {}).values():
        for entry in row.values():
            for key, static in entry.get('key_statics', {}).items():side_of_key[key] = static.get('side')
    groups = {};per_side = {}
    for report in diffs:
        for r in report.get('results', []):
            sig_of = r.get('signatures') or {};by_site = r.get('exercised_by_site') or {};sides = r.get('sides') or {}
            for key, n in by_site.items():
                sig = normalize_signature(sig_of.get(key, key.split('@')[0] + '(?)'));hook = key.split('@')[0]
                side = sides.get(key, side_of_key.get(key));side = 'array' if side is None and key in sides else side
                g = groups.setdefault(sig, dict(hook=hook, signature=sig, sites=[], exercised=0, lane_steps=r.get('exercised_lane_steps_total', r.get('lane_steps'))))
                g['sites'].append(dict(site=key, exercised=int(n), side=side));g['exercised'] += int(n)
                ps = per_side.setdefault((sig, str(side)), dict(hook=hook, signature=sig, side=side, sites=0, exercised=0))
                ps['sites'] += 1;ps['exercised'] += int(n)
    for g in groups.values():g['sites'].sort(key=lambda s: (-s['exercised'], s['site']));g['sites_never_exercised'] = sum(1 for s in g['sites'] if s['exercised'] == 0)
    rows = sorted(groups.values(), key=lambda g: (g['hook'], -g['exercised'], g['signature']))
    side_rows = sorted(per_side.values(), key=lambda g: (g['hook'], g['signature'], str(g['side'])))
    never = [g['signature'] for g in rows if g['exercised'] == 0]
    never_side = [dict(signature=g['signature'], side=g['side'], sites=g['sites']) for g in side_rows if g['exercised'] == 0]
    sites_total = sum(len(g['sites']) for g in rows);sites_never = sum(g['sites_never_exercised'] for g in rows)
    return dict(schema='pokezero.v2e-hook-signatures.v2', hooks=sorted({g['hook'] for g in rows}), signatures=len(rows), signatures_never_exercised=never,
        signature_sides=len(side_rows), signature_sides_never_exercised=never_side, side_source='diff.json' if any(r.get('sides') for d in diffs for r in d.get('results', [])) else ('hook_sites.json' if side_of_key else 'none'),
        sites=sites_total, sites_never_exercised=sites_never, sites_under_100=sum(1 for g in rows for s in g['sites'] if 0 < s['exercised'] < 100), groups=rows, per_side=side_rows)


def signatures_markdown(report):
    out = [f"# Hook call-site signatures: {report['signatures']} (hook, static-argument) signatures over {len(report['hooks'])} hooks; "
        f"{report['sites']} sites, {report['sites_never_exercised']} never exercised, {report['sites_under_100']} under 100 lane-steps; "
        f"{report['signature_sides']} (signature, side) pairs (side from {report['side_source']})", '',
        f"Signatures exercised nowhere ({len(report['signatures_never_exercised'])}):", '']
    out += [f'- `{s}`' for s in report['signatures_never_exercised']] or ['- none']
    out += ['', f"(signature, side) pairs exercised nowhere ({len(report['signature_sides_never_exercised'])}; M1's synthetic-lane list, per seat; side `None` or `array`: "
        "the seat is not a Python static at that site, i.e. a per-lane array such as the residual loop's `event % 2`, or the hook has no side parameter):", '']
    out += [f"- `{g['signature']}` side {g['side']} ({g['sites']} sites)" for g in report['signature_sides_never_exercised']] or ['- none']
    out += ['', '| Hook | Signature | Sites | Sites never exercised | Exercised lane-steps | Best site |', '| --- | --- | ---: | ---: | ---: | --- |']
    for g in report['groups']:
        best = g['sites'][0]
        out.append(f"| {g['hook']} | `{g['signature']}` | {len(g['sites'])} | {g['sites_never_exercised']} | {g['exercised']:,} | `{best['site']}` ({best['exercised']:,}) |")
    out += ['', '| Hook | Signature | Side | Sites | Exercised lane-steps |', '| --- | --- | --- | ---: | ---: |']
    out += [f"| {g['hook']} | `{g['signature']}` | {g['side']} | {g['sites']} | {g['exercised']:,} |" for g in report['per_side']]
    return '\n'.join(out) + '\n'


def cmd_signatures(args):
    diffs = [json.loads(Path(p).read_text()) for p in args.diff]
    report = signatures_report(diffs, json.loads(Path(args.sites).read_text()) if args.sites else None)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    (out / 'hook_signatures.json').write_text(json.dumps(report, indent=1) + '\n');(out / 'hook_signatures.md').write_text(signatures_markdown(report))
    print(f"signatures: {report['signatures']} signatures, {len(report['signatures_never_exercised'])} never exercised; {report['sites']} sites, "
        f"{report['sites_never_exercised']} never exercised, {report['sites_under_100']} under 100 -> {out / 'hook_signatures.json'}")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]);sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('sites', help='call sites and calls per step of every tappable hook (one traced step at the template)')
    q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True);q.add_argument('--arm', default='B', choices=('A', 'B'))
    q.add_argument('--out', required=True);q.add_argument('--hooks', nargs='*', help='hook names (default: v2e_hooks.ALL_HOOKS; "all": every tappable function)')
    q.set_defaults(fn=cmd_sites)
    q = sub.add_parser('signatures', help='group the hook: gates\' sites by (hook, static-argument signature) with exercised lane-steps')
    q.add_argument('--diff', nargs='+', required=True, help='diff.json files of hook: runs');q.add_argument('--out', required=True)
    q.add_argument('--sites', help='hook_sites.json: supplies each site\'s side for diff.json files recorded without `sides`')
    q.set_defaults(fn=cmd_signatures)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':raise SystemExit(main())
