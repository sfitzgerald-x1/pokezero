"""Content-anchored source citations: cite code by what it says, never by the line it sits on (stdlib only).

A line number in committed prose goes stale on any insertion above the cited code, and a shared table of them
conflicts on almost every PR. So prose cites `<path>#<id>` (`, #id2` and ` and #id2` reuse the path), and the entry
for that id names a scope, a needle that must sit on exactly ONE line of it, and an extent, and pins a digest of the
span's code. The guarantee, held by the negative controls in tests/test_gpu_engine_source_anchor.py:

  * a moved citation passes -- nothing stored is positional;
  * a changed cited statement fails CHANGED -- the .py digest is over the span's tokens;
  * a deleted or duplicated needle fails MISSING or AMBIGUOUS -- a guess is a wrong citation waiting to happen
    (scripts/c153_wide_negative_census.py `_anchor` learned that one the hard way).

Line numbers are only ever RENDERED, from the tree at hand, when output is produced.

The .py digest ignores position, whitespace, comments and bare string statements (docstrings), and nothing else. Two
token rules keep it position-free and Python-version-free: INDENT/DEDENT before the span's first token or after its
last are dropped (they belong to the code AROUND the span, so inserting a def above it would otherwise read as a
change), and 3.12+'s split f-/t-string tokens are re-joined into the one STRING 3.11 emits. Any other file type
digests its span's whitespace-normalised non-empty lines, so a comment edit there DOES change the digest.

The store is one JSON file per cited source, tests/source_anchors/<path>.json, one entry per line, sorted by id, so
two PRs editing different entries touch different lines and merge cleanly. `verify` reports a store whose text is
not `dump_store` of itself as FORMAT, so a hand-reformatted or old-format table cannot slip back in.
scripts/refresh_source_anchors.py checks it, re-pins CHANGED digests for review, and shows a citation's span.
"""
import ast
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import tokenize

STORE_DIR = 'tests/source_anchors'
FIELDS = ('scope', 'needle', 'extent', 'digest', 'cited_by')
_SLUG = r'[a-z0-9][a-z0-9_-]*'
_HEAD = re.compile(r'(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|cjs|mjs|js|ts|rs|md))#(' + _SLUG + ')')
_MORE = re.compile(r'(?:,\s*|\s+and\s+)#(' + _SLUG + ')')
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# Statement-bearing nodes only: a def is always a statement, and walking expressions could recurse ~1000 deep.
_BLOCKS = tuple(getattr(ast, name) for name in ('stmt', 'excepthandler', 'match_case') if hasattr(ast, name))
_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
_LAYOUT = {tokenize.INDENT, tokenize.DEDENT}
_BARE = _LAYOUT | {tokenize.NEWLINE}
_OPEN = {getattr(tokenize, name) for name in ('FSTRING_START', 'TSTRING_START') if hasattr(tokenize, name)}
_CLOSE = {getattr(tokenize, name) for name in ('FSTRING_END', 'TSTRING_END') if hasattr(tokenize, name)}


class AnchorError(Exception):
    """The anchor cannot name one span: a scope on a non-.py file, a bad extent or needle, an unparseable file."""


class AnchorMissing(AnchorError):
    """The file, scope or needle is gone. Re-point the citation; never renumber it."""


class AnchorAmbiguous(AnchorError):
    """The scope or needle matches more than once, so the citation has no single meaning."""


@dataclass(frozen=True)
class Anchor:
    """`needle` on exactly one line of `scope` in `path`; the span is that line, n lines from it, or its statement.

    path: repo-relative POSIX path. scope: '' for the whole file, or (.py only) the dotted qualname of a def/class,
    nested by plain dots ('Cls.method', 'outer.inner'). extent: 'line' | 'stmt' (.py only: the statement starting on
    the needle's line) | 'block' (.py only: the innermost compound statement whose body holds the needle's line, for
    a guard whose own first line is not unique) | 'header' or 'header:N' (the header of the Nth enclosing compound
    statement, innermost first, excluding its body) | int n >= 1.
    Or {'through': endpoint}, where endpoint has scope, needle and extent and names the last included code line;
    through spans survive inserted comments/docstrings.
    """
    path: str
    scope: str
    needle: str
    extent: object = 'line'

    @classmethod
    def from_entry(cls, path, entry):
        return cls(path, entry['scope'], entry['needle'], entry['extent'])


@dataclass(frozen=True)
class Span:
    """Lines `first`..`last` of `path`, 1-based and inclusive."""
    path: str
    first: int
    last: int


@dataclass(frozen=True)
class Problem:
    """One `verify` finding. `fixable`: refresh --write settles it (CHANGED, or a store that is only not canonical)."""
    kind: str
    cite: str
    message: str
    fixable: bool = False

    def __str__(self):
        return self.message


def _head(node):
    """First line of a statement: a decorated def/class starts at its first decorator."""
    return min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', ())])


class _Source:
    """One file's text and the parse products resolve and digest need, each built once."""

    def __init__(self, path, text):
        self.path, self.text = path, text.replace('\r\n', '\n').replace('\r', '\n')
        self.lines = self.text.split('\n')
        if self.lines[-1] == '':
            self.lines.pop()
        self.python = path.endswith('.py')
        self._tree = self._names = self._stmts = self._units = self._docs = None

    @property
    def tree(self):
        if self._tree is None:
            try:
                self._tree = ast.parse(self.text, self.path)
            except (SyntaxError, ValueError) as error:
                raise AnchorError(f'{self.path} does not parse: {error}') from None
        return self._tree

    def scope(self, qualname):
        """(first, last) of the scope: '' is the whole file, else the ONE def/class with that dotted qualname."""
        if not qualname:
            return 1, len(self.lines)
        if not self.python:
            raise AnchorError(f'{self.path}: scope {qualname!r} on a non-.py file (only .py has scopes)')
        if self._names is None:
            self._names = {}

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, _DEFS):
                        self._names.setdefault(prefix + child.name, []).append(child)
                        walk(child, prefix + child.name + '.')
                    elif isinstance(child, _BLOCKS):
                        walk(child, prefix)
            walk(self.tree, '')
        nodes = self._names.get(qualname, [])
        if not nodes:
            raise AnchorMissing(f'{self.path} has no def or class {qualname!r}')
        if len(nodes) > 1:
            raise AnchorAmbiguous(f'{self.path} defines {qualname!r} {len(nodes)} times, at lines '
                                  + ', '.join(str(n.lineno) for n in nodes))
        return _head(nodes[0]), nodes[0].end_lineno

    def statement(self, line):
        """(first, last) of the longest statement starting on `line`, or None. A decorated def/class starts at its
        first decorator, and a needle on that decorator's line selects it too."""
        if self._stmts is None:
            self._stmts = {}
            for node in ast.walk(self.tree):
                if isinstance(node, ast.stmt):
                    span = (_head(node), node.end_lineno)
                    for start in {node.lineno, span[0]}:
                        held = self._stmts.get(start)
                        if held is None or (span[1], -span[0]) > (held[1], -held[0]):
                            self._stmts[start] = span
        return self._stmts.get(line)

    def block(self, line):
        """(first, last) of the innermost compound statement (one with a body) whose body holds `line`, i.e. that
        starts before it and ends at or after it; None if none does."""
        best = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.stmt) and hasattr(node, 'body') and _head(node) < line <= node.end_lineno:
                span = (_head(node), node.end_lineno)
                if best is None or span[1] - span[0] < best[1] - best[0]:
                    best = span
        return best

    def header(self, line, levels=1):
        """Header of the innermost compound statement holding a unique body needle.

        This pins a repeated guard or loop header without pinning its body.
        """
        nodes = sorted((n for n in ast.walk(self.tree) if isinstance(n, ast.stmt)
                        and hasattr(n, 'body') and _head(n) < line <= n.end_lineno),
                       key=lambda n: n.end_lineno - _head(n))
        if len(nodes) < levels:
            return None
        node = nodes[levels - 1]
        return _head(node), max(_head(node), node.body[0].lineno - 1)

    def slice(self, start, end):
        (row, col), (end_row, end_col) = start, end
        if row == end_row:
            return self.lines[row - 1][col:end_col]
        return '\n'.join([self.lines[row - 1][col:]] + self.lines[row:end_row - 1]
                         + [self.lines[end_row - 1][:end_col]])

    def units(self):
        """(type, string, start) per token of the WHOLE file (so a fragment never fails to tokenize), with each
        3.12+ f-/t-string re-joined into one STRING of its source text, as 3.11 tokenizes it."""
        if self._units is None:
            self._units, depth, opened = [], 0, None
            try:
                for tok in tokenize.generate_tokens(io.StringIO(self.text).readline):
                    if tok.type in _OPEN:
                        opened = tok.start if depth == 0 else opened
                        depth += 1
                    elif depth:
                        if tok.type in _CLOSE:
                            depth -= 1
                            if depth == 0:
                                self._units.append((tokenize.STRING, self.slice(opened, tok.end), opened))
                    else:
                        self._units.append((tok.type, tok.string, tok.start))
            except (tokenize.TokenError, SyntaxError) as error:
                raise AnchorError(f'{self.path} does not tokenize: {error}') from None
        return self._units

    def docstring(self, position):
        """Whether `position` lies in a bare string statement (a docstring or a no-op string)."""
        if self._docs is None:
            def col(row, offset):   # ast columns count UTF-8 bytes, tokenize's count characters
                return len(self.lines[row - 1].encode('utf-8')[:offset].decode('utf-8', 'ignore'))
            self._docs = [((n.lineno, col(n.lineno, n.col_offset)), (n.end_lineno, col(n.end_lineno, n.end_col_offset)))
                          for n in ast.walk(self.tree)
                          if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                          and isinstance(n.value.value, str)]
        return any(start <= position < end for start, end in self._docs)


def _source(root, path, text=None):
    if text is None:
        file = Path(root) / path
        if not file.is_file():
            raise AnchorMissing(f'{path} does not exist')
        text = file.read_text(encoding='utf-8')
    return _Source(path, text)


def _resolve(src, anchor):
    if not anchor.needle:
        raise AnchorError(f'{anchor.path}: empty needle')
    if '\n' in anchor.needle:
        raise AnchorError(f'{anchor.path}: needle {anchor.needle!r} spans lines; it is matched against one raw line')
    first, last = src.scope(anchor.scope)
    where = anchor.path + (f' {anchor.scope}' if anchor.scope else '')
    hits = [n for n in range(first, last + 1) if anchor.needle in src.lines[n - 1]]
    if not hits:
        raise AnchorMissing(f'{anchor.needle!r} is not in {where}')
    if len(hits) > 1:
        raise AnchorAmbiguous(f'{anchor.needle!r} is on {len(hits)} lines of {where}:'
                              + ''.join(f'\n    {anchor.path}:{n}: {src.lines[n - 1].strip()}' for n in hits))
    line, extent = hits[0], anchor.extent
    if extent == 'line':
        return Span(anchor.path, line, line)
    if extent == 'stmt':
        if not src.python:
            raise AnchorError(f'{anchor.path}: extent stmt on a non-.py file')
        span = src.statement(line)
        if span is None:
            raise AnchorError(f'no statement starts on {anchor.path}:{line}: {src.lines[line - 1].strip()}')
        return Span(anchor.path, *span)
    header = re.fullmatch(r'header(?::([1-9][0-9]*))?', extent) if isinstance(extent, str) else None
    if extent == 'block' or header:
        if not src.python:
            raise AnchorError(f'{anchor.path}: extent {extent} on a non-.py file')
        span = src.header(line, int(header.group(1) or 1)) if header else src.block(line)
        if span is None:
            raise AnchorError(f'no compound statement holds {anchor.path}:{line}: {src.lines[line - 1].strip()}')
        return Span(anchor.path, *span)
    if isinstance(extent, int) and not isinstance(extent, bool) and extent >= 1:
        if line + extent - 1 > last:
            raise AnchorError(f'extent {extent} from {anchor.path}:{line} ends at line {line + extent - 1}, '
                              f'past the end of {where} at line {last}')
        return Span(anchor.path, line, line + extent - 1)
    if isinstance(extent, dict) and set(extent) == {'through'}:
        end = extent['through']
        if (not isinstance(end, dict) or set(end) != {'scope', 'needle', 'extent'}
                or not isinstance(end['scope'], str) or not isinstance(end['needle'], str)):
            raise AnchorError(f'{anchor.path}: through must name an endpoint scope, needle and extent')
        finish = _resolve(src, Anchor(anchor.path, end['scope'], end['needle'], end['extent'])).last
        if not line <= finish <= last:
            raise AnchorError(f'{anchor.path}: endpoint {finish} is outside the span starting at {line} in {where}')
        return Span(anchor.path, line, finish)
    raise AnchorError(f'{anchor.path}: extent {extent!r} is not line, stmt, block, header, through or an int >= 1')


def _code_units(src):
    """Drop whole bare-string statements, including their empty logical lines.

    Parentheses and the separator left by an inline docstring are also part of
    that no-op statement. Indentation still describes the executable blocks.
    """
    group, removed = [], False
    for kind, string, start in src.units():
        if kind in _SKIP:
            continue
        if src.docstring(start):
            removed = True
            continue
        if kind != tokenize.NEWLINE:
            group.append((kind, string, start))
            continue
        semantic = [i for i, (k, s, _) in enumerate(group) if k not in _LAYOUT and s != ';']
        previous = None
        for i, unit in enumerate(group):
            k, s, _ = unit
            if s == ';' and (not semantic or not semantic[0] < i < semantic[-1] or previous in (';', ':')):
                continue
            yield unit
            if k not in _LAYOUT:
                previous = s
        if semantic:
            # A removed trailing string can end on a later physical line.
            # Keep the logical terminator with the surviving code it closes.
            _, last_string, last_start = group[semantic[-1]]
            end = (last_start[0] + last_string.count('\n'), 0)
            yield kind, string, end if removed else start
        group, removed = [], False
    yield from group


def _digest(src, span):
    if not src.python:
        rows = (' '.join(line.split()) for line in src.lines[span.first - 1:span.last])
        return _sha('\n'.join(row for row in rows if row))
    kept = [(kind, string) for kind, string, start in _code_units(src)
            if span.first <= start[0] <= span.last]
    lo, hi = 0, len(kept)
    while lo < hi and kept[lo][0] in _LAYOUT:
        lo += 1
    while hi > lo and kept[hi - 1][0] in _LAYOUT:
        hi -= 1
    return _sha('\n'.join(tokenize.tok_name[kind] if kind in _BARE else f'{tokenize.tok_name[kind]}:{string}'
                          for kind, string in kept[lo:hi]))


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def _render(span):
    return f'{span.path}:{span.first}' if span.first == span.last else f'{span.path}:{span.first}-{span.last}'


def resolve(anchor, root, text=None):
    """The Span `anchor` names in the tree at `root` (or in `text`, the file's content at some other revision)."""
    return _resolve(_source(root, anchor.path, text), anchor)


def digest(anchor_or_span, root, text=None):
    """12 hex chars over the span's code: see the module docstring for what it ignores."""
    src = _source(root, anchor_or_span.path, text)
    span = _resolve(src, anchor_or_span) if isinstance(anchor_or_span, Anchor) else anchor_or_span
    return _digest(src, span)


def render(anchor_or_span, root=None, text=None):
    """'path:840' or 'path:840-845', from the tree at hand. Never commit the result."""
    span = anchor_or_span if isinstance(anchor_or_span, Span) else resolve(anchor_or_span, root, text)
    return _render(span)


def span_lines(span, root, text=None):
    """[(line number, raw text)] of the span."""
    lines = _source(root, span.path, text).lines
    return [(n, lines[n - 1]) for n in range(span.first, span.last + 1)]


def find_citations(text):
    """[(path, id)] of every `<path>#<id>` citation in `text`, continuations included, in order, duplicates kept."""
    found, pos = [], 0
    while True:
        match = _HEAD.search(text, pos)
        if match is None:
            return found
        path, pos = match.group(1), match.end()
        found.append((path, match.group(2)))
        while True:
            more = _MORE.match(text, pos)
            if more is None:
                break
            found.append((path, more.group(1)))
            pos = more.end()


def render_citations(text, root):
    """Render prose citations and their continuations at the current tree's lines, for generated reports only."""
    stores, rendered = {}, {}

    def location(path, ident):
        cite = (path, ident)
        if cite not in rendered:
            if path not in stores:
                file = store_file(root, path)
                stores[path] = load_store(file)['anchors'] if file.is_file() else {}
            if ident not in stores[path]:
                raise AnchorMissing(f'{path}#{ident}: no stored anchor')
            try:
                rendered[cite] = render(Anchor.from_entry(path, stores[path][ident]), root)
            except AnchorError as error:
                raise type(error)(f'{path}#{ident}: {error}') from None
        return rendered[cite]

    pieces, pos = [], 0
    while (match := _HEAD.search(text, pos)):
        path = match.group(1)
        pieces.extend((text[pos:match.start()], location(path, match.group(2))))
        pos = match.end()
        while (more := _MORE.match(text, pos)):
            pieces.extend((text[pos:more.start(1) - 1], location(path, more.group(1))))
            pos = more.end()
    return ''.join(pieces) + text[pos:]


def store_file(root, path):
    """The store holding `path`'s entries: tests/source_anchors/<path>.json."""
    return Path(root) / STORE_DIR / (path + '.json')


def store_paths(root):
    """Every cited path with a store file, by the file's location."""
    base = Path(root) / STORE_DIR
    if not base.is_dir():
        return []
    return sorted(file.relative_to(base).as_posix()[:-len('.json')] for file in base.rglob('*.json'))


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'duplicate key {key!r}')
        out[key] = value
    return out


def parse_store(text):
    """The store in `text`. ValueError unless it is {"path", "anchors"} with every entry holding exactly FIELDS."""
    store = json.loads(text, object_pairs_hook=_unique)
    if not (isinstance(store, dict) and set(store) == {'path', 'anchors'} and isinstance(store['path'], str)
            and isinstance(store['anchors'], dict)):
        raise ValueError('not {"path": ..., "anchors": {...}}')
    for key, entry in store['anchors'].items():
        if not isinstance(entry, dict) or set(entry) != set(FIELDS):
            raise ValueError(f'entry {key!r} does not hold exactly {", ".join(FIELDS)}')
        if not (all(isinstance(entry[f], str) for f in ('scope', 'needle', 'digest'))
                and isinstance(entry['cited_by'], list) and all(isinstance(f, str) for f in entry['cited_by'])):
            raise ValueError(f'entry {key!r}: scope, needle and digest must be strings and cited_by a list of paths')
    return store


def load_store(path):
    """The store in the file at `path`."""
    return parse_store(Path(path).read_text(encoding='utf-8'))


def dump_store(store):
    """The canonical text: one entry per line, keys in FIELDS order, sorted by id, cited_by sorted, final newline."""
    rows = [f'  {json.dumps(key, ensure_ascii=False)}: ' + json.dumps(
        {f: sorted(set(entry[f])) if f == 'cited_by' else entry[f] for f in FIELDS},
        ensure_ascii=False, separators=(', ', ': ')) for key, entry in sorted(store['anchors'].items())]
    anchors = '{\n' + ',\n'.join(rows) + '\n }' if rows else '{}'
    return '{\n "path": ' + json.dumps(store['path'], ensure_ascii=False) + ',\n "anchors": ' + anchors + '\n}\n'


def load_stores(root):
    """{cited path: store} for every store file under `root`."""
    return {path: load_store(store_file(root, path)) for path in store_paths(root)}


def _relative(root, file):
    file = Path(file)
    return file.resolve().relative_to(Path(root).resolve()).as_posix() if file.is_absolute() else file.as_posix()


def problems(root, consumer=None, paths=None):
    """Every Problem in the stores under `root`, in store, entry and citing-file order.

    FORMAT     a store file that does not parse, sits at the wrong path, or is not `dump_store` of itself
    MISSING / AMBIGUOUS / INVALID   the entry no longer resolves (INVALID: a bad scope, extent or needle)
    CHANGED    the span's digest moved; the message carries the current span
    ORPHAN     an entry's cited_by file is gone or no longer cites it (or cited_by is empty)
    UNANCHORED a citation in a cited_by file (or in `consumer`) with no entry, or whose entry does not list that file

    `consumer` (a repo file) narrows every check to the entries it lists or cites, the store files holding them, and
    the citations in it, so a consumer's own test owns its rows. `paths` narrows to entries of those cited paths.
    """
    root = Path(root)
    mine = _relative(root, consumer) if consumer is not None else None
    wanted = set(paths) if paths else None
    texts, sources, found, entries = {}, {}, [], {}

    def cites(file):
        if file not in texts:
            path = root / file
            texts[file] = (set(find_citations(path.read_text(encoding='utf-8', errors='replace')))
                           if path.is_file() else None)
        return texts[file]
    own = (cites(mine) or set()) if mine is not None else set()

    for cited in store_paths(root):
        if wanted is not None and cited not in wanted:
            continue
        file = store_file(root, cited)
        name = file.relative_to(root).as_posix()
        text = file.read_text(encoding='utf-8')
        try:
            store = parse_store(text)
        except ValueError as error:
            store, problem = None, Problem('FORMAT', name, f'FORMAT {name}: {error}')
        else:
            problem = (Problem('FORMAT', name, f'FORMAT {name}: "path" is {store["path"]!r}, not {cited!r}')
                       if store['path'] != cited else None if text == dump_store(store) else
                       Problem('FORMAT', name, f'FORMAT {name}: not canonical (refresh_source_anchors.py --write '
                               'rewrites it)', fixable=True))
        rows = store['anchors'] if store else {}
        if problem and (mine is None or any(path == cited for path, _ in own)
                        or any(mine in entry['cited_by'] for entry in rows.values())):
            found.append(problem)
        entries.update(((cited, key), entry) for key, entry in rows.items())

    for (cited, key), entry in sorted(entries.items()):
        cite = f'{cited}#{key}'
        if mine is not None and mine not in entry['cited_by'] and (cited, key) not in own:
            continue
        if mine is None and not entry['cited_by']:
            found.append(Problem('ORPHAN', cite, f'ORPHAN {cite} (cited_by nothing)'))
        for file in sorted(set(entry['cited_by'])) if mine is None else [mine] if mine in entry['cited_by'] else []:
            if cites(file) is None:
                found.append(Problem('ORPHAN', cite, f'ORPHAN {cite} (cited_by {file}, which does not exist)'))
            elif (cited, key) not in cites(file):
                found.append(Problem('ORPHAN', cite, f'ORPHAN {cite} (cited_by {file})'))
        anchor = Anchor.from_entry(cited, entry)
        try:
            if cited not in sources:
                sources[cited] = _source(root, cited)
            span = _resolve(sources[cited], anchor)
            now = _digest(sources[cited], span)
        except AnchorMissing as error:
            found.append(Problem('MISSING', cite, f'MISSING {cite}: {error}'))
        except AnchorAmbiguous as error:
            found.append(Problem('AMBIGUOUS', cite, f'AMBIGUOUS {cite}: {error}'))
        except AnchorError as error:
            found.append(Problem('INVALID', cite, f'INVALID {cite}: {error}'))
        else:
            if now != entry['digest']:
                body = ''.join(f'\n    {n}: {sources[cited].lines[n - 1]}' for n in range(span.first, span.last + 1))
                stored = entry['digest'] or 'nothing'
                found.append(Problem('CHANGED', cite, f'CHANGED {cite}: stored {stored}, now {now}, '
                                     f'at {_render(span)}{body}', fixable=True))

    for file in [mine] if mine is not None else sorted({f for entry in entries.values() for f in entry['cited_by']}):
        for cited, key in sorted(cites(file) or ()):
            if wanted is not None and cited not in wanted:
                continue
            entry, cite = entries.get((cited, key)), f'{cited}#{key}'
            if entry is None:
                found.append(Problem('UNANCHORED', cite, f'UNANCHORED {cite} in {file}'))
            elif file not in entry['cited_by']:
                found.append(Problem('UNANCHORED', cite, f'UNANCHORED {cite} in {file}: its entry does not list '
                                     f'{file} in cited_by'))
    return found


def verify(root, consumer=None):
    """Every problem as a human-readable string naming its citation 'path#id' (FORMAT: its store file); [] if clean."""
    return [problem.message for problem in problems(root, consumer)]
