"""Rewrite per-lane point indexing in engine modules to gpu_engine.lane_index.

    x[r, i, ...]            ->  _li.take(x, r, i, ...)
    x.at[r, i, ...].set(v)  ->  _li.put(x, r, (i, ...), v)

``r`` must be a lane iota: a local name whose every binding in its function
(or an enclosing one) is ``j.arange(n)``/``jnp.arange(n)``/``_r(state)``, or such
a call written inline. ``_r`` counts only where the module defines it as a
one-argument arange. Index elements must be plain expressions (no slices,
None, Ellipsis or starred). Nothing else is touched; the rest of each file is
kept byte for byte. ``lane_index`` keeps the original indexing for large
indexed axes and checks the iota's static shape.

``--trust-rows NAME`` additionally treats NAME as the lane iota wherever it is
bound; use it only after checking every binding by hand (for example rows
returned from a helper that builds them with arange).

Usage: lane_index_codemod.py [--check] [--trust-rows NAME...] FILE...
"""
import argparse
import ast
import sys

IMPORT = b'from . import lane_index as _li\n'
ARANGE_MODULES = ('j', 'jnp')


def _is_arange(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'arange'
        and isinstance(node.func.value, ast.Name) and node.func.value.id in ARANGE_MODULES
        and len(node.args) == 1 and not node.keywords)


def _module_row_helpers(tree):
    """Module-level one-liners ``def _r(x): return j.arange(...)``."""
    helpers = set()
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and len(node.body) == 1 and isinstance(node.body[0], ast.Return)
                and _is_arange(node.body[0].value) and len(node.args.args) == 1):
            helpers.add(node.name)
    return helpers


def _bindings(fn):
    """Name -> list of bound value nodes (None for non-assignment bindings), this scope only."""
    out = {}
    def bind(name, value):out.setdefault(name, []).append(value)
    args = fn.args
    for a in args.posonlyargs + args.args + args.kwonlyargs + ([args.vararg] if args.vararg else []) + ([args.kwarg] if args.kwarg else []):
        bind(a.arg, None)
    body = fn.body if isinstance(fn.body, list) else [fn.body]
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            if not isinstance(node, ast.Lambda):bind(node.name, None)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):bind(target.id, node.value)
                else:
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name):bind(n.id, None)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.comprehension)):
            target = node.target
            for n in ast.walk(target):
                if isinstance(n, ast.Name):bind(n.id, None)
        elif isinstance(node, ast.NamedExpr):bind(node.target.id, None)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):bind(n.id, None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:bind((a.asname or a.name).split('.')[0], None)
        elif isinstance(node, ast.ExceptHandler) and node.name:bind(node.name, None)
        stack.extend(ast.iter_child_nodes(node))
    return out


class Rewriter:
    def __init__(self, source, trusted=()):
        self.src = source.encode();self.tree = ast.parse(source);self.trusted = set(trusted)
        self.helpers = _module_row_helpers(self.tree)
        starts = [0]
        for line in self.src.splitlines(keepends=True):starts.append(starts[-1] + len(line))
        self.starts = starts;self.edits = [];self.reads = 0;self.writes = 0

    def span(self, node):
        return self.starts[node.lineno - 1] + node.col_offset, self.starts[node.end_lineno - 1] + node.end_col_offset

    def text(self, node):
        a, b = self.span(node);return self.src[a:b]

    def row(self, node, rows):
        if isinstance(node, ast.Name):return node.id in rows
        return _is_arange(node) or (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in self.helpers and len(node.args) == 1 and not node.keywords)

    def point(self, sub, rows):
        if not isinstance(sub.slice, ast.Tuple) or len(sub.slice.elts) < 2:return None
        elts = sub.slice.elts
        if not self.row(elts[0], rows):return None
        if any(isinstance(e, (ast.Slice, ast.Starred)) or (isinstance(e, ast.Constant) and e.value in (None, Ellipsis))
                for e in elts[1:]):return None
        return elts[0], elts[1:]

    def match(self, node, rows):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'set'
                and len(node.args) == 1 and not node.keywords and isinstance(node.func.value, ast.Subscript)
                and isinstance(node.func.value.value, ast.Attribute) and node.func.value.value.attr == 'at'):
            hit = self.point(node.func.value, rows)
            if hit:return 'put', node.func.value.value.value, hit[0], hit[1], node.args[0]
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'at':return None
            hit = self.point(node, rows)
            if hit:return 'take', node.value, hit[0], hit[1], None
        return None

    def render(self, node, rows):
        m = self.match(node, rows)
        if m:
            kind, x, r, idx, v = m
            parts = [self.render(e, rows) for e in idx]
            if kind == 'take':
                self.reads += 1
                return b'_li.take(' + self.render(x, rows) + b',' + self.render(r, rows) + b',' + b','.join(parts) + b')'
            self.writes += 1
            index = b'(' + b','.join(parts) + (b',)' if len(parts) == 1 else b')')
            return b'_li.put(' + self.render(x, rows) + b',' + self.render(r, rows) + b',' + index + b',' + self.render(v, rows) + b')'
        a, b = self.span(node);out = self.src[a:b];pieces = []
        for child in self.children(node, rows):
            ca, cb = self.span(child);pieces.append((ca, cb, self.render(child, rows)))
        for ca, cb, text in sorted(pieces, reverse=True):out = out[:ca - a] + text + out[cb - a:]
        return out

    def children(self, node, rows):
        """Direct expression children that contain a match (positions valid)."""
        out = []
        for child in ast.iter_child_nodes(node):
            if not hasattr(child, 'lineno') or isinstance(child, (ast.Lambda,)) and False:continue
            if self.contains(child, rows):out.append(child)
        return out

    def contains(self, node, rows):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):return False
        if self.match(node, rows):return True
        return any(self.contains(c, rows) for c in ast.iter_child_nodes(node))

    def scope_rows(self, fn, inherited):
        rows = set(inherited)
        for name, values in _bindings(fn).items():
            if name in self.trusted or values and all(v is not None and self.row(v, set()) for v in values):rows.add(name)
            else:rows.discard(name)
        return rows

    def visit(self, node, rows):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            rows = self.scope_rows(node, rows)
            for d in getattr(node, 'decorator_list', []):self.visit(d, set())
            body = node.body if isinstance(node.body, list) else [node.body]
            for stmt in body:self.visit(stmt, rows)
            return
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:self.visit(stmt, set())
            return
        if hasattr(node, 'lineno') and not isinstance(node, ast.stmt) and self.match(node, rows):
            a, b = self.span(node);self.edits.append((a, b, self.render(node, rows)));return
        for child in ast.iter_child_nodes(node):self.visit(child, rows)

    def run(self):
        for stmt in self.tree.body:self.visit(stmt, set())
        out = self.src
        for a, b, text in sorted(self.edits, reverse=True):out = out[:a] + text + out[b:]
        if self.edits:
            if b'lane_index as _li' in self.src or any(isinstance(n, ast.Name) and n.id == '_li' for n in ast.walk(self.tree)):
                raise ValueError('module already binds _li')
            last = max((n for n in self.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), key=lambda n: n.end_lineno, default=None)
            at = self.starts[last.end_lineno] if last is not None else 0
            out = out[:at] + IMPORT + out[at:]
        return out.decode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--check', action='store_true', help='report counts only')
    parser.add_argument('--trust-rows', nargs='*', default=(), help='names asserted (by review) to always be the lane iota')
    args = parser.parse_args(argv)
    total = [0, 0]
    for path in args.files:
        source = open(path).read();rewriter = Rewriter(source, args.trust_rows);out = rewriter.run()
        ast.parse(out)
        total[0] += rewriter.reads;total[1] += rewriter.writes
        print(f'{path}: {rewriter.reads} reads, {rewriter.writes} writes')
        if not args.check and out != source:open(path, 'w').write(out)
    print(f'total: {total[0]} reads, {total[1]} writes')
    return 0


if __name__ == '__main__':sys.exit(main())
