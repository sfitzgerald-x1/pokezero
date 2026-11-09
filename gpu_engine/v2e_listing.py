"""Device-path listing: the oracle's source pruned to the lines the collector's device path can execute (design note
section 2.2), as a reading copy for the hand-writers of M1..M4.

An AST pruner walks transition.step_with_chance and every function it reaches (through the module aliases of each
module's imports, recursively: the hooks, their callees, the mechanics) and evaluates `if` tests statically under the
device path's facts:
- `_device_public` is set (a device_public.Context);
- `_lifecycle_trace` is False, `_public_trace` is None, `_hypothetical_profile` is None and every `_*_context` is None;
- every `table.*` sub-table is present, as are the Context's belief and port tables and every tracker sub-carry;
- `hasattr(c.belief_tables, X)` / `getattr(table, X, None)` follow the real table pytrees (tests/fixtures/v2e_layout_description.json).
A statically true `if` keeps its header (so the reader sees the guard) and its body; a false one keeps only its else
branch; an unknown test keeps both. After a statically taken `if ...: return` (or a return) the rest of the block is
dead and dropped; a conditional expression whose test is static is annotated with its dead branch. Comments and blank
lines are not statements and do not survive. Every surviving line is printed with its original line number, one text
file per function (`<module>.<function>.txt`; with --trace-lines, `NNNN:~` marks a line without equations in the traced
jaxpr), plus listing.json (the surviving line sets, the static headers, the notes, counts by phase for step_with_chance)
and listing.md.

Cross-check: `trace-lines` (GPU host) traces the oracle step and records every gpu_engine (module, line) that appears in
the jaxpr's source info; `write --trace-lines <that json>` then reports, per function, listing lines without equations
(control flow, aliasing, Python-only statements) and jaxpr lines the listing dropped (a pruning error, or a function the
reachability walk missed), in crosscheck.json/.md. Paths are arguments.
"""
import argparse
import ast
import collections
import json
import os
from pathlib import Path
import re
import time

PACKAGE = Path(__file__).resolve().parent
DESCRIPTION = PACKAGE.parent / 'tests' / 'fixtures' / 'v2e_layout_description.json'
ROOT_FUNCTION = ('transition', 'step_with_public')   # the collector's entry; step_with_chance is reached from it
STEP_FUNCTION = ('transition', 'step_with_chance')
# Names the device path fixes (design note section 1.1).
NAME_FACTS = {'_device_public': 'present', '_lifecycle_trace': False, '_public_trace': None, '_hypothetical_profile': None, '_hypothetical': False,
    'table': 'present', 'tables': 'present', 'belief_tables': 'present', 'port_tables': 'present', 'leftovers': 'present', 'entry_table': 'present',
    'item_table': 'present', 'effects': 'present'}
CONTEXT_NAME = re.compile(r'^_\w*context\w*$')
# Roots whose attribute chains are present (not None) on the device path: tables, the Context, the trackers and their sub-carries.
PRESENT_ROOTS = {'table', 'tables', 'c', 't', 'trackers', 'h', 'carry', 'td', 'f', 'fields', 'facts', 'belief_tables', 'port_tables',
    'history', 'entry_table', 'item_table', 'effects'}
# Modules never followed (library code, trace-only modules).
SKIP_MODULES = {'jax', 'jnp', 'np', 'numpy', 'math', 'json', 'functools', 'itertools', 'lax', 'dtypes', 'lax_internal'}
UNKNOWN = object()


# ---------------------------------------------------------------------------
# Module loading and import maps.

class Module:
    def __init__(self, name, path):
        self.name = name;self.path = Path(path);self.source = self.path.read_text();self.lines = self.source.splitlines()
        self.tree = ast.parse(self.source);self.functions = {}   # name -> FunctionDef (top level)
        self.aliases = {}   # alias -> module name (import ... as alias / from . import x as alias)
        self.names = {}     # imported function name -> (module name, original name)
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):self.functions[node.name] = node
            elif isinstance(node, ast.Import):
                for a in node.names:self.aliases[a.asname or a.name] = a.name
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.module is None:   # from . import x as y
                    for a in node.names:self.aliases[a.asname or a.name] = a.name
                elif node.level:   # from .effects import a as b
                    for a in node.names:self.names[a.asname or a.name] = (node.module, a.name)


def load_modules(package=PACKAGE):
    modules = {}
    for path in sorted(Path(package).glob('*.py')):
        if path.name.startswith(('v2e_', 'v2_', 'test')) or path.name == '__init__.py':continue
        try:modules[path.stem] = Module(path.stem, path)
        except SyntaxError:continue
    return modules


def present_attributes(description_path=DESCRIPTION):
    """{'table': {...}, 'belief': {...}}: first-level attribute names of the engine and belief table pytrees."""
    out = {'table': set(), 'belief': set()}
    if Path(description_path).exists():
        for e in json.loads(Path(description_path).read_text())['tables']:
            path = e['path']
            if path.startswith('belief.'):out['belief'].add(path.split('.')[1])
            elif path.startswith('.'):out['table'].add(path.split('.')[1])
    return out


# ---------------------------------------------------------------------------
# Static evaluation of `if` tests.

class Facts:
    def __init__(self, attributes, local_aliases=None):
        self.attributes = attributes;self.locals = local_aliases or {}

    def name(self, n):
        if n in self.locals:return self.locals[n]
        if n in NAME_FACTS:return NAME_FACTS[n]
        if CONTEXT_NAME.match(n):return None
        return UNKNOWN

    def value(self, node):
        """'present' (an object that is not None), None, True/False, a constant, or UNKNOWN."""
        if isinstance(node, ast.Constant):return node.value
        if isinstance(node, ast.Name):return self.name(node.id)
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):root = root.value
            if isinstance(root, ast.Name) and (root.id in PRESENT_ROOTS or self.name(root.id) == 'present'):return 'present'
            return UNKNOWN
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == 'hasattr' and len(node.args) == 2 and isinstance(node.args[1], ast.Constant):
                target = node.args[0];attr = node.args[1].value
                if isinstance(target, ast.Attribute) and target.attr in ('belief_tables',):return attr in self.attributes['belief']
                if isinstance(target, ast.Name) and target.id in ('table', 'tables'):return attr in self.attributes['table']
                return UNKNOWN
            if node.func.id == 'getattr' and len(node.args) == 3 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[0], ast.Name):
                if node.args[0].id == 'table':return 'present' if node.args[1].value in self.attributes['table'] else self.value(node.args[2])
                return UNKNOWN
            if node.func.id == 'type' and len(node.args) == 1:return ('type', self.value(node.args[0]))
            return UNKNOWN
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            t = self.truth(node.operand);return UNKNOWN if t is UNKNOWN else not t
        if isinstance(node, ast.BoolOp):
            values = [self.truth(v) for v in node.values]
            if isinstance(node.op, ast.And):
                if any(v is False for v in values):return False
                return True if all(v is True for v in values) else UNKNOWN
            if any(v is True for v in values):return True
            return False if all(v is False for v in values) else UNKNOWN
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            left = self.value(node.left);right = self.value(node.comparators[0]);negate = isinstance(node.ops[0], ast.IsNot)
            if isinstance(left, tuple) and left[0] == 'type':   # type(_device_public) is device_public.Context
                known = left[1] == 'present' and isinstance(node.comparators[0], ast.Attribute) and node.comparators[0].attr == 'Context'
                return (not negate) if known else UNKNOWN
            if left is UNKNOWN or right is UNKNOWN:return UNKNOWN
            if right is None:same = left is None
            elif left is None:same = right is None
            else:return UNKNOWN
            return (not same) if negate else same
        if isinstance(node, ast.IfExp):
            t = self.truth(node.test)
            if t is True:return self.value(node.body)
            if t is False:return self.value(node.orelse)
            return UNKNOWN
        return UNKNOWN

    def truth(self, node):
        v = self.value(node)
        if v is UNKNOWN or isinstance(v, tuple):return UNKNOWN
        if v == 'present':return True
        return bool(v)


def local_aliases(fn, facts):
    """Names assigned exactly once at any depth of the function to a statically known value (carry = trackers.negative_items)."""
    counts = collections.Counter();values = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            counts[node.targets[0].id] += 1;values[node.targets[0].id] = node.value
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and isinstance(node.target, ast.Name):counts[node.target.id] += 2
        elif isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name):counts[node.target.id] += 2
    out = {}
    for name, value in values.items():
        if counts[name] == 1:
            v = facts.value(value)
            if v is not UNKNOWN and not isinstance(v, tuple):out[name] = v
    return out


# ---------------------------------------------------------------------------
# Pruning.

class Pruned:
    def __init__(self, module, fn):
        self.module = module;self.function = fn.name;self.first = fn.lineno;self.last = fn.end_lineno
        self.lines = set();self.static_headers = set();self.unknown_headers = set();self.dropped = set()
        self.notes = {}   # line -> annotation (a conditional expression whose test is static: which branch is dead)

    def keep(self, first, last):
        self.lines.update(range(first, last + 1))

    def drop(self, node):
        self.dropped.update(range(node.lineno, node.end_lineno + 1))


def header_span(node):
    """Lines of a compound statement's header (through its test/iter/items expression)."""
    expr = getattr(node, 'test', None) or getattr(node, 'iter', None)
    if expr is None and isinstance(node, ast.With):expr = node.items[-1].context_expr
    if expr is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        end = min(stmt.lineno for stmt in node.body) - 1 if node.body else node.lineno
        return node.lineno, max(node.lineno, end)
    return node.lineno, (expr.end_lineno if expr is not None else node.lineno)


def else_line(module, body_end, orelse_start):
    for n in range(body_end + 1, orelse_start):
        if module.lines[n - 1].strip().startswith('else'):return n
    return None


def terminates(statements):
    """Whether a statement list always leaves the function (ends in return/raise, or an if whose both branches do)."""
    if not statements:return False
    last = statements[-1]
    if isinstance(last, (ast.Return, ast.Raise)):return True
    if isinstance(last, ast.If):return terminates(last.body) and terminates(last.orelse)
    return False


def annotate(out, facts, node):
    """Note conditional expressions with a statically known test inside a kept statement (which branch is dead)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.IfExp):
            t = facts.truth(sub.test)
            if t is not UNKNOWN:
                dead = 'else' if t else 'if'
                text = f'{ast.unparse(sub.test)} is {"always" if t else "never"} true: the {dead}-branch of this conditional expression is dead on the device path'
                out.notes[sub.lineno] = (out.notes.get(sub.lineno, '') + '; ' + text).strip('; ')


def prune_function(module, fn, facts):
    facts = Facts(facts.attributes, dict(facts.locals, **local_aliases(fn, facts)))
    out = Pruned(module, fn);first, last = header_span(fn);out.keep(first, last)
    def walk(statements):
        """Returns True when the walked block always leaves the function on the device path (the caller drops the rest)."""
        for n, node in enumerate(statements):
            if isinstance(node, ast.If):
                t = facts.truth(node.test);h0, h1 = header_span(node);annotate(out, facts, node.test)
                if t is True:
                    out.keep(h0, h1);out.static_headers.update(range(h0, h1 + 1));done = walk(node.body)
                    for o in node.orelse:out.drop(o)
                elif t is False:
                    out.dropped.update(range(h0, h1 + 1))
                    for b in node.body:out.drop(b)
                    done = walk(node.orelse)
                else:
                    out.keep(h0, h1);out.unknown_headers.update(range(h0, h1 + 1));done_body = walk(node.body);done = False
                    if node.orelse:
                        if not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) and node.orelse[0].col_offset == node.col_offset):
                            e = else_line(module, node.body[-1].end_lineno, node.orelse[0].lineno)
                            if e:out.keep(e, e)
                        done_else = walk(node.orelse);done = done_body and done_else
                if done:   # a statically taken `if ...: return` (or both branches returning): the rest of the block is dead
                    for rest in statements[n + 1:]:out.drop(rest)
                    return True
            elif isinstance(node, (ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                h0, h1 = header_span(node);out.keep(h0, h1)
                for field in ('body', 'orelse', 'finalbody'):walk(getattr(node, field, []) or [])
                for handler in getattr(node, 'handlers', []) or []:out.keep(handler.lineno, handler.lineno);walk(handler.body)
            else:
                out.keep(node.lineno, node.end_lineno);annotate(out, facts, node)
                if isinstance(node, (ast.Return, ast.Raise)):
                    for rest in statements[n + 1:]:out.drop(rest)
                    return True
        return False
    walk(fn.body)
    return out


REFERENCE = re.compile(r'(?<![\w.])([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?')


def reached_calls(module, pruned):
    """(callee module name, function name) pairs the surviving lines reference (called or passed as a callback):
    `alias.name` through the module's import aliases, bare `name` for the module's own or from-imported functions."""
    text = '\n'.join(module.lines[n - 1] for n in sorted(pruned.lines))
    text = re.sub(r"'[^'\n]*'|\"[^\"\n]*\"|#[^\n]*", '', text)   # strings and comments
    out = set()
    for m in REFERENCE.finditer(text):
        head, attr = m.group(1), m.group(2)
        if attr is not None:
            if head in module.aliases and module.aliases[head] not in SKIP_MODULES:out.add((module.aliases[head], attr))
        elif head in module.functions and head != pruned.function:out.add((module.name, head))
        elif head in module.names:out.add(module.names[head])
    return out


def call_site_facts(sites_report):
    """{(module, function): {parameter: value}} from a hook_sites.json (v2e_hook_taps sites): a static argument that has
    the same value at every traced call site of a hook is a fact inside that hook (e.g. reveal_item's `original` is
    always True), so `x if original else y` can be annotated and `if original:` pruned."""
    from gpu_engine import v2e_hook_taps as taps
    out = {}
    for name, row in sites_report.get('sites', {}).items():
        alias, function = taps.split(name);module = taps.TAP_MODULES[alias].rsplit('.', 1)[-1]
        values = {}
        for entry in row.values():
            for statics in entry.get('statics', []):
                for k, v in statics.items():values.setdefault(k, set()).add(json.dumps(v))
        facts = {k: json.loads(next(iter(v))) for k, v in values.items() if len(v) == 1}
        if facts:out[(module, function)] = facts
    return out


def listing(modules, attributes, root=ROOT_FUNCTION, skip_modules=(), site_facts=None):
    """{(module, function): Pruned} of every function reached from `root` on the device path. `site_facts`
    (call_site_facts) adds per-function constants observed at every call site."""
    facts = Facts(attributes);done = {};queue = [root]
    while queue:
        key = queue.pop(0)
        if key in done:continue
        mod, fn = key
        if mod not in modules or fn not in modules[mod].functions or mod in skip_modules:continue
        local = Facts(attributes, dict((site_facts or {}).get(key, {})))
        pruned = prune_function(modules[mod], modules[mod].functions[fn], local);done[key] = pruned
        for callee in sorted(reached_calls(modules[mod], pruned)):
            if callee not in done:queue.append(callee)
    return done


def render(module, pruned, trace_lines=None):
    """The reading copy: `NNNN: source`. With trace lines, a line that produced no equation in the traced jaxpr (control
    flow, aliasing, Python-only work) is marked `NNNN:~ source`; a conditional expression whose test is static gets a
    trailing `## note`."""
    have = set(trace_lines.get(module.name, ())) if trace_lines is not None else None
    out = []
    for n in sorted(pruned.lines):
        mark = '~' if have is not None and n not in have else ' '
        note = f'   ## {pruned.notes[n]}' if n in pruned.notes else ''
        out.append(f'{n:5d}:{mark}{module.lines[n - 1]}{note}\n')
    return ''.join(out)


def phase_counts(pruned):
    from gpu_engine import v2e_census as cz
    counts = collections.Counter()
    for n in sorted(pruned.lines):counts[cz.phase_of(n) or 'outside'] += 1
    return dict(counts)


def write_listing(out, modules, listed, *, trace_lines=None):
    out = Path(out);out.mkdir(parents=True, exist_ok=True);report = dict(schema='pokezero.v2e-listing.v1', root=list(ROOT_FUNCTION), functions={})
    lines_by_module = {m: {int(k) for k in c} for m, c in trace_lines['lines'].items()} if trace_lines else None
    for (mod, fn), pruned in sorted(listed.items(), key=lambda kv: (kv[0][0] != 'transition', kv[0] != STEP_FUNCTION, kv[0])):
        (out / f'{mod}.{fn}.txt').write_text(render(modules[mod], pruned, lines_by_module))
        row = dict(module=mod, function=fn, span=[pruned.first, pruned.last], source_lines=pruned.last - pruned.first + 1,
            surviving=len(pruned.lines), static_true_headers=len(pruned.static_headers), unknown_headers=len(pruned.unknown_headers),
            dropped=len(pruned.dropped), lines=sorted(pruned.lines), static_true_header_lines=sorted(pruned.static_headers), notes={str(k): v for k, v in sorted(pruned.notes.items())})
        if (mod, fn) == STEP_FUNCTION:row['by_phase'] = phase_counts(pruned)
        report['functions'][f'{mod}.{fn}'] = row
    report['totals'] = dict(functions=len(listed), surviving=sum(r['surviving'] for r in report['functions'].values()),
        by_module={m: sum(r['surviving'] for r in report['functions'].values() if r['module'] == m) for m in sorted({k[0] for k in listed})})
    if trace_lines is not None:report['crosscheck'] = crosscheck(modules, listed, trace_lines)
    (out / 'listing.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'listing.md').write_text(listing_markdown(report))
    if trace_lines is not None:
        (out / 'crosscheck.json').write_text(json.dumps(report['crosscheck'], indent=1) + '\n')
        (out / 'crosscheck.md').write_text(crosscheck_markdown(report['crosscheck']))
    return report


def listing_markdown(report):
    out = ['| Function | Source lines | Surviving | Static-true `if` headers | Unknown `if` headers |', '| --- | ---: | ---: | ---: | ---: |']
    for name, r in report['functions'].items():
        out.append(f"| {name} | {r['source_lines']} | {r['surviving']} | {r['static_true_headers']} | {r['unknown_headers']} |")
    root = report['functions'].get('.'.join(STEP_FUNCTION))
    if root:
        out += ['', f"step_with_chance: {root['surviving']} surviving lines ({root['surviving'] - root['static_true_headers']} without the statically true `if` headers) by phase:", '']
        out += ['| Phase | Surviving lines |', '| --- | ---: |'] + [f'| {k} | {v} |' for k, v in root['by_phase'].items()]
    t = report['totals']
    out += ['', f"Total: {t['functions']} functions, {t['surviving']} surviving lines; by module: " + ', '.join(f'{m} {n}' for m, n in t['by_module'].items()) + '.']
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# Cross-check against the traced jaxpr's source info.

def jaxpr_lines(closed):
    """{module: {line: equations}} of every gpu_engine frame (oracle modules only) in the source info of every equation
    (sub-jaxprs included), plus {module: {function: lines}}."""
    from jax.extend import core as jcore
    lines = collections.defaultdict(collections.Counter);functions = collections.defaultdict(lambda: collections.defaultdict(set))
    def walk(jaxpr):
        for e in jaxpr.eqns:
            tb = getattr(e.source_info, 'traceback', None)
            for f in (tb.frames if tb is not None else ()):
                if '/gpu_engine/' not in f.file_name:continue
                module = f.file_name.rsplit('/', 1)[-1][:-3]
                if module.startswith(('v2e_', 'v2_')):continue
                lines[module][f.line_num] += 1;functions[module][f.function_name].add(f.line_num)
            for sub in jcore.jaxprs_in_params(e.params):walk(sub)
    walk(closed.jaxpr)
    return ({m: {str(k): v for k, v in sorted(c.items())} for m, c in lines.items()},
        {m: {fn: sorted(ls) for fn, ls in fs.items()} for m, fs in functions.items()})


def crosscheck(modules, listed, trace):
    """Per listed function: listing lines without equations and jaxpr lines the listing lacks; plus jaxpr functions not listed."""
    lines = {m: {int(k): v for k, v in c.items()} for m, c in trace['lines'].items()}
    out = dict(functions={}, jaxpr_functions_not_listed=[], totals=dict(listing_lines=0, listing_lines_without_equations=0, jaxpr_lines_missing_from_listing=0))
    for (mod, fn), pruned in sorted(listed.items()):
        have = lines.get(mod, {});span = range(pruned.first, pruned.last + 1)
        in_jaxpr = {n for n in have if n in span};missing = sorted(in_jaxpr - pruned.lines);silent = sorted(pruned.lines - in_jaxpr)
        out['functions'][f'{mod}.{fn}'] = dict(listing_lines=len(pruned.lines), jaxpr_lines=len(in_jaxpr), listing_lines_without_equations=silent,
            jaxpr_lines_missing_from_listing=missing, equations_on_missing_lines=sum(have[n] for n in missing))
        out['totals']['listing_lines'] += len(pruned.lines);out['totals']['listing_lines_without_equations'] += len(silent)
        out['totals']['jaxpr_lines_missing_from_listing'] += len(missing)
    listed_names = {f'{m}.{f}' for m, f in listed}
    for mod, fns in trace['functions'].items():
        for fn, ls in fns.items():
            if f'{mod}.{fn}' not in listed_names and mod in modules and fn in modules[mod].functions:
                out['jaxpr_functions_not_listed'].append(dict(function=f'{mod}.{fn}', lines=len(ls)))
    return out


def crosscheck_markdown(cc):
    out = ['| Function | Listing lines | Jaxpr lines | Listing lines without equations | Jaxpr lines missing from the listing |', '| --- | ---: | ---: | ---: | --- |']
    for name, r in cc['functions'].items():
        miss = ', '.join(map(str, r['jaxpr_lines_missing_from_listing'][:12])) + (' ...' if len(r['jaxpr_lines_missing_from_listing']) > 12 else '')
        out.append(f"| {name} | {r['listing_lines']} | {r['jaxpr_lines']} | {len(r['listing_lines_without_equations'])} | {len(r['jaxpr_lines_missing_from_listing'])}{': ' + miss if miss else ''} |")
    t = cc['totals']
    out += ['', f"Totals: {t['listing_lines']} listing lines, {t['listing_lines_without_equations']} without equations, {t['jaxpr_lines_missing_from_listing']} jaxpr lines missing from the listing.",
        '', 'Jaxpr functions not listed (reached by the trace but not by the static walk): ' + (', '.join(f"{r['function']} ({r['lines']} lines)" for r in cc['jaxpr_functions_not_listed']) or 'none') + '.']
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# CLI.

def cmd_write(args):
    modules = load_modules();attributes = present_attributes(args.description)
    site_facts = call_site_facts(json.loads(Path(args.sites).read_text())) if args.sites else None
    listed = listing(modules, attributes, site_facts=site_facts)
    trace = json.loads(Path(args.trace_lines).read_text()) if args.trace_lines else None
    report = write_listing(args.out, modules, listed, trace_lines=trace)
    root = report['functions']['.'.join(STEP_FUNCTION)]
    print(f"listing: {report['totals']['functions']} functions, {report['totals']['surviving']} lines; step_with_chance {root['surviving']} "
        f"({root['surviving'] - root['static_true_headers']} without static headers) -> {args.out}")
    if trace is not None:
        t = report['crosscheck']['totals']
        print(f"crosscheck: {t['jaxpr_lines_missing_from_listing']} jaxpr lines missing from the listing, {t['listing_lines_without_equations']} listing lines without equations, "
            f"{len(report['crosscheck']['jaxpr_functions_not_listed'])} jaxpr functions not listed")
    return 0


def cmd_trace_lines(args):
    import jax
    from gpu_engine import v2e_census as cz, v2e_modules
    ctx = cz.load_context(args);state, actions, trackers = ctx['template']
    fn = v2e_modules.step(ctx['engine_tables'], ctx['observation_tables'])
    begin = time.perf_counter();closed = jax.make_jaxpr(fn)(state, actions, trackers);traced = time.perf_counter() - begin
    lines, functions = jaxpr_lines(closed)
    report = dict(schema='pokezero.v2e-trace-lines.v1', source_commit=os.environ.get('V2_SOURCE_COMMIT'), jax=jax.__version__, seconds=traced,
        equations=sum(1 for _ in _walk(closed.jaxpr)), lines=lines, functions=functions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True);Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(f"trace-lines: {report['equations']} equations, {sum(len(v) for v in lines.values())} (module, line) pairs in {len(lines)} modules -> {args.out}")
    return 0


def _walk(jaxpr):
    from jax.extend import core as jcore
    for e in jaxpr.eqns:
        yield e
        for sub in jcore.jaxprs_in_params(e.params):yield from _walk(sub)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0]);sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('write', help='write the pruned listing (CPU; optional cross-check against trace-lines output)')
    q.add_argument('--out', required=True);q.add_argument('--description', default=str(DESCRIPTION));q.add_argument('--trace-lines')
    q.add_argument('--sites', help='hook_sites.json (v2e_hook_taps sites): static arguments constant over every call site become facts inside the hook')
    q.set_defaults(fn=cmd_write)
    q = sub.add_parser('trace-lines', help='GPU host: (module, line) pairs in the traced oracle step\'s source info')
    q.add_argument('--config', required=True);q.add_argument('--reference-run', required=True);q.add_argument('--arm', default='B', choices=('A', 'B'))
    q.add_argument('--out', required=True);q.set_defaults(fn=cmd_trace_lines)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':raise SystemExit(main())
