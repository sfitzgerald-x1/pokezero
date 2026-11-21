"""Identity-plus-digest citations for regenerated research records.

The resolver selects executable Python statements (not physical line lengths).
Other sources use a unique normalized text span within a named function/hunk.
Line numbers are inputs to generation and optional display output, never identity.
"""
import ast
from functools import lru_cache
import io
import json
from pathlib import Path
import re
import tokenize

from gpu_engine import source_anchor as sa

PATTERN = re.compile(r'(?P<path>[\w/.-]+)::(?P<scope>[^\s\[\]]+)\[(?P<needle>"(?:[^"\\]|\\.)*")\]@(?P<digest>[0-9a-f]{12})')


def _rust_scopes(source):
    # Preserve positions while hiding comments and quoted braces. Lifetimes
    # such as &'static are not character literals.
    quoted = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|r(?P<hashes>\#+)"[\s\S]*?"(?P=hashes)|"(?:[^"\\]|\\[\s\S])*"|\x27(?:\\.|[^\x27\\\n])\x27')
    code = quoted.sub(lambda m: re.sub(r'[^\n]', ' ', m[0]), source)
    stack, ends = [], {}
    for i, char in enumerate(code):
        if char == '{':stack.append(i)
        elif char == '}' and stack:ends[stack.pop()] = i
    owners = []
    for match in re.finditer(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:mod\s+(\w+)|impl\s+([^\n{]+))\s*\{', code, re.M):
        opening = code.index('{', match.start(), match.end())
        name = match[1] or re.sub(r'\W+', '_', match[2].strip()).strip('_')
        if opening in ends:owners.append((name, opening, ends[opening]))
    scopes = []
    for match in re.finditer(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:async|unsafe|const)\s+)*fn\s+(\w+)\s*[(<]', code, re.M):
        opening = code.find('{', match.end())
        if opening not in ends:continue
        first = code.count('\n', 0, match.start()+len(match[0])-len(match[0].lstrip()))+1
        last = code.count('\n', 0, ends[opening])+1
        prefix = [name for name, lo, hi in sorted(owners, key=lambda row: row[1]) if lo < match.start() < hi]
        scopes.append(('.'.join(prefix+[match[1]]), first, last))
    return scopes


@lru_cache(maxsize=8)
def _parsed(path, source):
    lines = source.splitlines()
    if path.endswith('.py'):
        tree = ast.parse(source)
        scopes = []
        def visit(node, prefix=''):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = prefix + child.name
                    scopes.append((name, child.lineno, child.end_lineno))
                    visit(child, name + '.')
                else:
                    visit(child, prefix)
        visit(tree)
        code = list(lines)
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                code[token.start[0]-1] = code[token.start[0]-1][:token.start[1]]
        return lines, tree, scopes, code
    if path.endswith('.rs'):
        return lines, None, _rust_scopes(source), lines
    starts = []
    for number, line in enumerate(lines, 1):
        match = re.search(r'\bfn\s+(\w+)\s*[(<]', line)
        if match and (re.match(r'^\s*[+ -]?\s*(?:pub(?:\([^)]*\))?\s+)?fn\s', line) or line.startswith('@@')):
            starts.append((match.group(1), number))
    scopes = [(name, first, starts[i+1][1]-1 if i+1 < len(starts) else len(lines))
              for i, (name, first) in enumerate(starts)]
    return lines, None, scopes, lines


def _scope(scopes, line, size):
    candidates = [(name, lo, hi) for name, lo, hi in scopes if lo <= line <= hi]
    return min(candidates, key=lambda row: row[2]-row[1]) if candidates else ('module', 1, size)


def _statement(tree, line):
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno <= line <= n.end_lineno
             and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    return min(nodes, key=lambda n: (n.end_lineno-n.lineno, n.end_col_offset-n.col_offset)) if nodes else None


def _hits(lines, needle, lo, hi):
    parts = needle.split('\n')
    normalized = [' '.join(line.split()) for line in lines]
    return [n for n in range(lo, hi-len(parts)+2) if normalized[n-1:n-1+len(parts)] == parts]


def _call_end(lines, first, limit):
    """Include continued Rust/patch call arguments in the cited text span."""
    balance = 0
    for number in range(first, limit+1):
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', lines[number-1]).split('//', 1)[0]
        balance += sum(code.count(c) for c in '([') - sum(code.count(c) for c in ')]')
        if balance <= 0:
            return number
    raise sa.AnchorError(f'unterminated call at source line {first}')


@lru_cache(maxsize=128)
def _make(path, line, source):
    lines, tree, scopes, code = _parsed(path, source)
    scope, lo, hi = _scope(scopes, line, len(lines))
    if tree is not None:
        candidates = sorted((n for n in ast.walk(tree) if isinstance(n, ast.stmt)
                             and n.lineno <= line <= n.end_lineno
                             and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))),
                            key=lambda n: (n.end_lineno-n.lineno, n.end_col_offset-n.col_offset))
        for node in candidates:
            needle = ' '.join(code[node.lineno-1].split())
            if needle and _hits(code, needle, lo, hi) == [node.lineno]:
                selected = _statement(tree, node.lineno)
                span = sa.Span(path, selected.lineno, selected.end_lineno)
                break
        else:
            raise sa.AnchorAmbiguous(f'{path}::{scope}: no unique statement covering source line {line}')
    else:
        selected = None
        for width in range(1, min(32, hi-lo+2)):
            for first in (line, max(lo, line-width+1)):
                last = max(min(hi, first+width-1), _call_end(code, first, hi))
                if not first <= line <= last:
                    continue
                needle = '\n'.join(' '.join(s.split()) for s in code[first-1:last])
                if '@@' in needle or not needle:
                    continue  # patch hunk offsets must never enter an identity
                if _hits(code, needle, lo, hi) == [first]:
                    selected = sa.Span(path, first, last)
                    break
            if selected is not None:
                break
        if selected is None:
            raise sa.AnchorAmbiguous(f'{path}::{scope}: no unique source span covering line {line}')
        span = selected
    # Escaped punctuation keeps identities safe inside a Markdown table/code span.
    quoted = json.dumps(needle, ensure_ascii=False).replace('|', '\\u007c').replace('`', '\\u0060')
    return f'{path}::{scope}[{quoted}]@{sa.digest(span, Path("."), text=source)}'


def citation_at(path, line, root):
    """Regenerate the identity and digest of the statement selected by a source resolver."""
    source = (Path(root)/path).read_text(encoding='utf-8')
    if not 1 <= line <= len(source.splitlines()):
        raise sa.AnchorMissing(f'{path}: no source line {line}')
    return _make(path, line, source)


def resolve(citation, root):
    """Resolve a recorded identity, independently of its recorded digest."""
    match = PATTERN.fullmatch(citation)
    if match is None:
        raise sa.AnchorError(f'invalid source citation: {citation!r}')
    path, scope = match['path'], match['scope']
    source = (Path(root)/path).read_text(encoding='utf-8')
    lines, tree, scopes, code = _parsed(path, source)
    candidates = [(lo, hi) for name, lo, hi in scopes if name == scope]
    if scope == 'module':candidates = [(1, len(lines))]
    needle = json.loads(match['needle'])
    hits = [(n, hi) for lo, hi in candidates for n in _hits(code, needle, lo, hi)]
    identity = citation.rsplit('@', 1)[0]
    if not hits:raise sa.AnchorMissing(f'MISSING {identity}')
    if len(hits) != 1:raise sa.AnchorAmbiguous(f'AMBIGUOUS {identity}: {len(hits)} spans')
    first = hits[0][0]
    node = _statement(tree, first) if tree is not None else None
    span = sa.Span(path, node.lineno, node.end_lineno) if node else sa.Span(path, first, first+len(needle.split('\n'))-1)
    return span, source, match['digest']


def verify_text(text, root):
    """Return named failures for each distinct identity embedded in a generated record."""
    errors = []
    for citation in dict.fromkeys(m.group(0) for m in PATTERN.finditer(text)):
        try:
            span, source, expected = resolve(citation, root)
            if sa.digest(span, root, text=source) != expected:
                errors.append('CHANGED '+citation.rsplit('@', 1)[0])
        except (sa.AnchorError, OSError, SyntaxError) as error:
            errors.append(citation.rsplit('@', 1)[0]+': '+str(error))
    return errors
