"""Content-anchored citations (gpu_engine/source_anchor.py, scripts/refresh_source_anchors.py); pure stdlib.

Every case runs on a scratch root built from the small fixtures below. The negative controls (a)-(h) carry the design's
guarantee: a moved citation passes, a changed cited statement fails, a deleted or duplicated needle fails, and
comment or docstring edits pass on .py but a comment edit fails on any other file type.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / 'scripts'), str(ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)
from gpu_engine import source_anchor as sa  # noqa: E402
import refresh_source_anchors as refresh  # noqa: E402

PY, CJS = 'gpu_engine/fixture.py', 'gpu_engine/hooks.cjs'
A, B = 'scripts/consumer_a.py', 'tests/test_consumer_b.py'
FIXTURE_PY = '''\
"""Fixture module for source anchors."""
import functools

LIMIT = 3


@functools.cache
def recharge_turn(state, bonus):
    """Return the recharge turn."""
    total = state + bonus  # carry the bonus
    if total > LIMIT:
        return LIMIT
    return total


class Holder:
    """Holds a value."""

    class Inner:
        def method(self, x):
            return x * 2

    def method(self, x):
        y = x + 1
        return y


def outer(a):
    def inner(b):
        return a + b
    return inner


def label(a):
    return f"v{a!r:>3}" + "s"  # labelled


if LIMIT > 2:
    def pick(x):
        return x
else:
    def pick(x):
        return -x
'''
FIXTURE_CJS = '''\
// Fixture hooks.
function rechargeTurn(state, bonus) {
  const total = state + bonus;
  // cap at the limit
  return Math.min(total, 3);
}
module.exports = { rechargeTurn };
'''
CONSUMERS = {
    A: '"""Consumer A: the bonus is carried at gpu_engine/fixture.py#recharge-sum, #recharge-cap and #recharge-def.\n'
       '\nThe inner method doubles (gpu_engine/fixture.py#inner-method); the hook sums at '
       'gpu_engine/hooks.cjs#cjs-total.\n"""\n',
    B: '"""Consumer B cites gpu_engine/fixture.py#holder-method only."""\n',
}
# {cited path: {id: (scope, needle, extent, cited_by)}}
ENTRIES = {
    PY: {
        'recharge-sum': ('recharge_turn', 'total = state', 'line', [A]),
        'recharge-cap': ('recharge_turn', 'if total > LIMIT', 3, [A]),
        'recharge-def': ('', 'def recharge_turn(', 'stmt', [A]),
        'inner-method': ('Holder.Inner.method', 'return x *', 'line', [A]),
        'holder-method': ('Holder.method', 'y = x', 'line', [B]),
    },
    CJS: {'cjs-total': ('', 'const total = state + bonus', 3, [A])},
}


def at(needle, text=FIXTURE_PY):
    """The fixture's 1-based line holding `needle`; the expectations below are stated by content too."""
    hits = [n for n, line in enumerate(text.split('\n'), 1) if needle in line]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def build(root):
    for path, text in ((PY, FIXTURE_PY), (CJS, FIXTURE_CJS), *CONSUMERS.items()):
        (root/path).parent.mkdir(parents=True, exist_ok=True)
        (root/path).write_text(text, encoding='utf-8')
    for path, rows in ENTRIES.items():
        anchors = {key: dict(scope=scope, needle=needle, extent=extent, cited_by=cited_by,
                             digest=sa.digest(sa.Anchor(path, scope, needle, extent), root))
                   for key, (scope, needle, extent, cited_by) in rows.items()}
        file = sa.store_file(root, path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(sa.dump_store({'path': path, 'anchors': anchors}), encoding='utf-8')


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = refresh.main(list(argv))
    return code, out.getvalue()


def anchor(path, key):
    scope, needle, extent, _ = ENTRIES[path][key]
    return sa.Anchor(path, scope, needle, extent)


class Scratch(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='source-anchor-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        build(self.root)

    def edit(self, path, old, new):
        file = self.root/path
        text = file.read_text(encoding='utf-8')
        self.assertEqual(text.count(old), 1, f'{old!r} must occur once in {path}')
        file.write_text(text.replace(old, new), encoding='utf-8')

    def span(self, path, key):
        return sa.resolve(anchor(path, key), self.root)

    def stores(self):
        return {path: sa.store_file(self.root, path).read_bytes() for path in ENTRIES}

    def kinds(self, found):
        return sorted({problem.split(' ', 1)[0] for problem in found})


class ResolveTests(Scratch):
    def test_through_extent_tracks_its_endpoint_and_rejects_missing_or_ambiguous_ends(self):
        file = self.root/'span.py'
        file.write_text('def f():\n    start = 1\n    finish = 2\n    return finish\n')
        a = sa.Anchor('span.py', 'f', 'start =', {'through': dict(scope='f', needle='finish =', extent='stmt')})
        before = sa.digest(a, self.root)
        file.write_text(file.read_text().replace('    finish', '    # documentation expands\n    finish', 1))
        self.assertEqual(sa.resolve(a, self.root), sa.Span('span.py', 2, 4))
        self.assertEqual(sa.digest(a, self.root), before)
        file.write_text(file.read_text().replace('finish = 2', 'finish = 3'))
        self.assertNotEqual(sa.digest(a, self.root), before)
        source = file.read_text()
        file.write_text(source.replace('finish = 3', 'last = 3'))
        with self.assertRaises(sa.AnchorMissing):sa.resolve(a, self.root)
        file.write_text(source.replace('finish = 3', 'finish = 3\n    finish = 4'))
        with self.assertRaises(sa.AnchorAmbiguous):sa.resolve(a, self.root)

    def test_line_int_and_stmt_extents(self):
        line = at('total = state')
        self.assertEqual(self.span(PY, 'recharge-sum'), sa.Span(PY, line, line))
        self.assertEqual(self.span(PY, 'recharge-cap'), sa.Span(PY, at('if total > LIMIT'), at('return total')))
        self.assertEqual(self.span(CJS, 'cjs-total'), sa.Span(CJS, at('const total', FIXTURE_CJS),
                                                               at('return Math.min', FIXTURE_CJS)))
        stmt = sa.Anchor(PY, 'recharge_turn', 'if total > LIMIT', 'stmt')
        self.assertEqual(sa.resolve(stmt, self.root), sa.Span(PY, at('if total > LIMIT'), at('return LIMIT')))
        holder = sa.Anchor(PY, '', 'class Holder', 'stmt')
        self.assertEqual(sa.resolve(holder, self.root), sa.Span(PY, at('class Holder'), at('return y')))

    def test_block_is_the_innermost_compound_statement_holding_the_needle(self):
        """For a guard whose own first line is not unique: anchor its body's distinctive line, extent block."""
        guard = sa.Anchor(PY, 'recharge_turn', 'return LIMIT', 'block')
        self.assertEqual(sa.resolve(guard, self.root), sa.Span(PY, at('if total > LIMIT'), at('return LIMIT')))
        # Inner.method's body: the innermost holder is that def (on the line above), not class Inner or Holder
        self.assertEqual(sa.resolve(sa.Anchor(PY, '', 'return x * 2', 'block'), self.root),
                         sa.Span(PY, at('return x * 2') - 1, at('return x * 2')))
        with self.assertRaisesRegex(sa.AnchorError, 'no compound statement holds'):
            sa.resolve(sa.Anchor(PY, '', 'LIMIT = 3', 'block'), self.root)
        with self.assertRaisesRegex(sa.AnchorError, 'block on a non-.py file'):
            sa.resolve(sa.Anchor(CJS, '', 'const total', 'block'), self.root)
        # the guard's own line is part of the digest: changing the condition changes it
        before = sa.digest(guard, self.root)
        (self.root/PY).write_text(FIXTURE_PY.replace('if total > LIMIT', 'if total >= LIMIT'), encoding='utf-8')
        self.assertNotEqual(sa.digest(guard, self.root), before)

    def test_stmt_on_a_decorated_def_starts_at_its_first_decorator(self):
        whole = sa.Span(PY, at('@functools.cache'), at('return total'))
        self.assertEqual(self.span(PY, 'recharge-def'), whole)
        # A needle on the decorator line selects the same statement.
        self.assertEqual(sa.resolve(sa.Anchor(PY, '', '@functools.cache', 'stmt'), self.root), whole)

    def test_scope_by_dotted_qualname_including_nested_class_method_and_inner_function(self):
        self.assertEqual(self.span(PY, 'inner-method').first, at('return x * 2'))
        self.assertEqual(self.span(PY, 'holder-method').first, at('y = x + 1'))
        inner = sa.Anchor(PY, 'outer.inner', 'return a', 'line')
        self.assertEqual(sa.resolve(inner, self.root).first, at('return a + b'))
        self.assertEqual(sa.resolve(sa.Anchor(PY, 'recharge_turn', 'cache', 'line'), self.root).first,
                         at('@functools.cache'), 'a def scope starts at its first decorator')

    def test_a_duplicate_qualname_is_ambiguous(self):
        with self.assertRaises(sa.AnchorAmbiguous) as caught:
            sa.resolve(sa.Anchor(PY, 'pick', 'return', 'line'), self.root)
        self.assertIn("'pick'", str(caught.exception))

    def test_a_duplicate_needle_is_ambiguous_and_lists_each_candidate(self):
        with self.assertRaises(sa.AnchorAmbiguous) as caught:
            sa.resolve(sa.Anchor(PY, 'recharge_turn', 'return', 'line'), self.root)
        self.assertIn(f'{PY}:{at("return LIMIT")}: return LIMIT', str(caught.exception))
        self.assertIn(f'{PY}:{at("return total")}: return total', str(caught.exception))

    def test_missing_file_scope_and_needle_are_missing(self):
        for missing in (sa.Anchor('gpu_engine/gone.py', '', 'x', 'line'), sa.Anchor(PY, 'nope', 'x', 'line'),
                        sa.Anchor(PY, 'Holder.nope', 'x', 'line'), sa.Anchor(PY, '', 'not in the file', 'line')):
            with self.assertRaises(sa.AnchorMissing, msg=missing):
                sa.resolve(missing, self.root)

    def test_bad_scope_extent_and_needle_are_anchor_errors(self):
        for bad in (sa.Anchor(CJS, 'rechargeTurn', 'const total', 'line'),   # scope on a non-.py file
                    sa.Anchor(CJS, '', 'const total', 'stmt'),              # stmt on a non-.py file
                    sa.Anchor(PY, 'recharge_turn', 'if total > LIMIT', 4),  # int extent past the scope
                    sa.Anchor(PY, '', 'else:', 'stmt'),                     # no statement starts on that line
                    sa.Anchor(PY, '', 'total = state', 0), sa.Anchor(PY, '', 'total = state', True),
                    sa.Anchor(PY, '', 'total = state', 'paragraph'), sa.Anchor(PY, '', '', 'line')):
            with self.assertRaises(sa.AnchorError, msg=bad) as caught:
                sa.resolve(bad, self.root)
            self.assertIs(type(caught.exception), sa.AnchorError, bad)

    def test_render(self):
        self.assertEqual(sa.render(anchor(PY, 'recharge-sum'), self.root), f'{PY}:{at("total = state")}')
        self.assertEqual(sa.render(anchor(PY, 'recharge-cap'), self.root),
                         f'{PY}:{at("if total > LIMIT")}-{at("return total")}')
        self.assertEqual(sa.render(sa.Span(CJS, 3, 3)), f'{CJS}:3')

    def test_rendered_prose_tracks_moves_and_expands_continuations(self):
        text = f'see {PY}#recharge-sum, #recharge-cap and #recharge-sum; then {CJS}#cjs-total.'
        def expected():
            one = sa.render(anchor(PY, 'recharge-sum'), self.root)
            two = sa.render(anchor(PY, 'recharge-cap'), self.root)
            three = sa.render(anchor(CJS, 'cjs-total'), self.root)
            return f'see {one}, {two} and {one}; then {three}.'
        before = sa.render_citations(text, self.root)
        self.assertEqual(before, expected())
        file = self.root / PY
        file.write_text('\n\n' + file.read_text())
        after = sa.render_citations(text, self.root)
        self.assertEqual(after, expected())
        self.assertNotEqual(after, before)

    def test_rendered_prose_names_an_unanchored_citation(self):
        with self.assertRaisesRegex(sa.AnchorMissing, 'missing-entry'):
            sa.render_citations(f'{PY}#missing-entry', self.root)


class DigestTests(Scratch):
    def test_preexisting_optional_semicolon_is_stable_under_docstring_insertion(self):
        file = self.root/'docs.py'
        for original, annotated in [
            ('def f():\n    value = 1;\n', 'def f():\n    value = 1; "doc"\n'),
            ('def f(): value = 1;\n', 'def f(): value = 1; "doc"\n'),
            ('def f():\n    value = 1;\n', 'def f():\n    value = 1; """doc\n    expanded"""\n'),
        ]:
            for extent in ('line', 'stmt'):
                file.write_text(original)
                a = sa.Anchor('docs.py', 'f', 'value =', extent)
                before = sa.digest(a, self.root)
                file.write_text(annotated)
                self.assertEqual(sa.digest(a, self.root), before)

    def test_multiline_inline_documentation_keeps_line_and_endpoint_terminators(self):
        file = self.root/'docs.py'
        original = 'def f():\n    start = 1\n    finish = 2\n'
        for extent in ('line', {'through': dict(scope='f', needle='finish =', extent='line')}):
            file.write_text(original)
            a = sa.Anchor('docs.py', 'f', 'start =', extent)
            before = sa.digest(a, self.root)
            for target in ('start = 1', 'finish = 2'):
                file.write_text(original.replace(target, target+'; """documentation\n    expanded"""'))
                self.assertEqual(sa.digest(a, self.root), before)

    def test_multiline_executable_string_keeps_its_ending_line_after_noop_insertion(self):
        file = self.root/'docs.py'
        original = 'def f():\n    start = 1\n    value = """first\n    second"""\n'
        for a in (sa.Anchor('docs.py', 'f', 'value =', 'line'),
                  sa.Anchor('docs.py', 'f', 'start =', {'through': dict(scope='f', needle='value =', extent='line')})):
            file.write_text(original);before = sa.digest(a, self.root)
            for doc in ('"doc"', '"""doc\n    expanded"""'):
                file.write_text(original.replace('second"""', 'second"""; '+doc))
                self.assertEqual(sa.digest(a, self.root), before)

    def test_docstring_insertion_removal_and_parentheses_preserve_code_digest(self):
        file = self.root/'docs.py'
        original = 'def f():\n    value = 1\n    return value\n'
        file.write_text(original)
        a = sa.Anchor('docs.py', 'f', 'def f():', 'stmt')
        before = sa.digest(a, self.root)
        for inserted in ('    "documentation"\n', '    ("documentation")\n',
                         '    """documentation\n    expanded"""\n', '    (\n        "documentation"\n    )\n'):
            with self.subTest(inserted=inserted):
                file.write_text(original.replace('    value', inserted+'    value', 1))
                self.assertEqual(sa.digest(a, self.root), before)
        file.write_text(original)
        self.assertEqual(sa.digest(a, self.root), before)

    def test_inline_noop_strings_do_not_leave_separators_in_the_digest(self):
        file = self.root/'docs.py'
        a = sa.Anchor('docs.py', 'f', 'def f():', 'stmt')
        for original, annotated in [
            ('def f(): return 1\n', 'def f(): "doc"; return 1\n'),
            ('def f():\n    if True: return 1\n', 'def f():\n    if True: "doc"; return 1\n'),
            ('def f():\n    if True: x = 1; return x\n', 'def f():\n    if True: "doc"; x = 1; "more"; return x\n'),
            ('def f():\n    return 1\n', 'def f():\n    "doc"; return 1\n'),
            ('def f():\n    x = 1; return x\n', 'def f():\n    x = 1; "a"; "b"; return x\n'),
            ('def f():\n    x = 1\n    return x\n', 'def f():\n    x = 1; "a"; "b"\n    return x\n'),
        ]:
            file.write_text(original);before = sa.digest(a, self.root)
            file.write_text(annotated);self.assertEqual(sa.digest(a, self.root), before)

    def test_the_python_digest_is_its_token_stream_with_fstrings_joined_as_one_string(self):
        # The algorithm is part of the store format: pinned here by construction, the same on 3.11 and 3.12+.
        tokens = ['NAME:return', 'STRING:f"v{a!r:>3}"', 'OP:+', 'STRING:"s"', 'NEWLINE']   # leading INDENT dropped
        expected = hashlib.sha256('\n'.join(tokens).encode()).hexdigest()[:12]
        self.assertEqual(sa.digest(sa.Anchor(PY, 'label', 'return f"', 'line'), self.root), expected)

    def test_the_other_digest_is_its_whitespace_normalised_nonempty_lines(self):
        rows = ['const total = state + bonus;', '// cap at the limit', 'return Math.min(total, 3);']
        self.assertEqual(sa.digest(anchor(CJS, 'cjs-total'), self.root),
                         hashlib.sha256('\n'.join(rows).encode()).hexdigest()[:12])

    def test_digest_of_a_span_equals_digest_of_its_anchor(self):
        self.assertEqual(sa.digest(self.span(PY, 'recharge-def'), self.root),
                         sa.digest(anchor(PY, 'recharge-def'), self.root))


class CitationTests(unittest.TestCase):
    def test_continuations_order_and_duplicates(self):
        text = ('see gpu_engine/a.py#first, #second and #third; then gpu_engine/b.cjs#x and (docs/c.md#y-2), '
                'gpu_engine/a.py#first again,#tail')
        self.assertEqual(sa.find_citations(text), [
            ('gpu_engine/a.py', 'first'), ('gpu_engine/a.py', 'second'), ('gpu_engine/a.py', 'third'),
            ('gpu_engine/b.cjs', 'x'), ('docs/c.md', 'y-2'), ('gpu_engine/a.py', 'first')])

    def test_a_citation_never_starts_inside_a_longer_path(self):
        # The lookbehind: `xgpu_engine/a.py#b` does not cite gpu_engine/a.py. It is read whole or not at all.
        self.assertEqual(sa.find_citations('xgpu_engine/a.py#b'), [('xgpu_engine/a.py', 'b')])
        self.assertEqual(sa.find_citations('see lib/gpu_engine/a.py#b'), [('lib/gpu_engine/a.py', 'b')])

    def test_urls_uppercase_ids_other_suffixes_and_detached_ids_are_not_citations(self):
        for text in ('https://github.com/o/r/blob/main/gpu_engine/a.py#L10', 'gpu_engine/a.py#Upper',
                     'gpu_engine/a.txt#b', 'gpu_engine/a.py #b'):
            self.assertEqual(sa.find_citations(text), [], text)


class StoreTests(Scratch):
    def test_canonical_text_and_round_trip(self):
        store = {'path': 'gpu_engine/a.py', 'anchors': {
            'b': {'scope': 'f', 'needle': 'y = "é"', 'extent': 2, 'digest': '0123456789ab',
                  'cited_by': ['z.py', 'a.md']},
            'a': {'scope': '', 'needle': 'x = 1', 'extent': 'line', 'digest': 'ba9876543210', 'cited_by': ['s.py']}}}
        text = sa.dump_store(store)
        self.assertEqual(text, '{\n "path": "gpu_engine/a.py",\n "anchors": {\n'
                         '  "a": {"scope": "", "needle": "x = 1", "extent": "line", "digest": "ba9876543210", '
                         '"cited_by": ["s.py"]},\n'
                         '  "b": {"scope": "f", "needle": "y = \\"é\\"", "extent": 2, "digest": "0123456789ab", '
                         '"cited_by": ["a.md", "z.py"]}\n }\n}\n')
        self.assertEqual(sa.dump_store(sa.parse_store(text)), text)
        file = sa.store_file(self.root, PY)
        self.assertEqual(sa.dump_store(sa.load_store(file)), file.read_text(encoding='utf-8'))
        self.assertEqual(sorted(sa.load_stores(self.root)), sorted(ENTRIES))
        self.assertEqual(sa.verify(self.root), [])

    def test_a_hand_reformatted_store_is_format(self):
        file = sa.store_file(self.root, PY)
        file.write_text(json.dumps(sa.load_store(file), indent=2) + '\n', encoding='utf-8')
        self.assertEqual(sa.verify(self.root), [f'FORMAT tests/source_anchors/{PY}.json: not canonical '
                                                '(refresh_source_anchors.py --write rewrites it)'])

    def test_unparseable_duplicate_id_and_misplaced_stores_are_format(self):
        file = sa.store_file(self.root, PY)
        text = file.read_text(encoding='utf-8')
        row = next(line for line in text.splitlines() if line.startswith('  "holder-method"'))
        file.write_text(text.replace(row, row + '\n' + row), encoding='utf-8')
        self.assertIn("duplicate key 'holder-method'", '\n'.join(sa.verify(self.root)))
        file.write_text(text.replace(f'"path": "{PY}"', '"path": "gpu_engine/elsewhere.py"'), encoding='utf-8')
        self.assertIn(f"FORMAT tests/source_anchors/{PY}.json: \"path\" is 'gpu_engine/elsewhere.py'",
                      '\n'.join(sa.verify(self.root)))


class VerifyTests(Scratch):
    def test_orphans(self):
        self.edit(A, '(gpu_engine/fixture.py#inner-method)', '(the inner method)')
        (self.root/B).unlink()
        self.assertEqual(sa.verify(self.root), [
            f'ORPHAN {PY}#holder-method (cited_by {B}, which does not exist)',
            f'ORPHAN {PY}#inner-method (cited_by {A})'])

    def test_unanchored(self):
        self.edit(A, 'Consumer A:', f'Consumer A, see {PY}#ghost and {PY}#holder-method:')
        self.assertEqual(sa.verify(self.root), [
            f'UNANCHORED {PY}#ghost in {A}',
            f'UNANCHORED {PY}#holder-method in {A}: its entry does not list {A} in cited_by'])

    def test_consumer_filter_owns_its_rows(self):
        self.edit(PY, 'y = x + 1', 'y = x + 2')
        self.edit(B, 'only.', f'only, and {PY}#nowhere.')
        self.assertEqual(sa.verify(self.root, consumer=A), [])
        self.assertEqual(sa.verify(self.root, consumer=self.root/A), [])
        mine = sa.verify(self.root, consumer=self.root/B)
        self.assertEqual(self.kinds(mine), ['CHANGED', 'UNANCHORED'])
        self.assertTrue(mine[0].startswith(f'CHANGED {PY}#holder-method: stored '), mine)
        self.assertEqual(mine[1], f'UNANCHORED {PY}#nowhere in {B}')
        self.assertEqual(sa.verify(self.root, consumer=B), mine)
        self.assertEqual(sorted(sa.verify(self.root)), sorted(mine))


class NegativeControls(Scratch):
    """Each control mutates a fresh copy of the fixture tree. These are the design's guarantee."""

    def test_a_moved_citation_still_passes(self):
        before = sa.render(anchor(PY, 'recharge-def'), self.root)
        self.edit(PY, '@functools.cache\n', '\n\n\ndef unrelated(z):\n    return z * z\n\n\n@functools.cache\n')
        self.assertNotEqual(sa.render(anchor(PY, 'recharge-def'), self.root), before, 'the control must move it')
        self.assertEqual(sa.verify(self.root), [], 'inserting lines and a def above a citation must not fail it')

    def test_b_a_citation_moved_to_the_end_of_the_file_still_passes(self):
        file = self.root/PY
        text = file.read_text(encoding='utf-8')
        start, end = text.index('@functools.cache\n'), text.index('    return total\n') + len('    return total\n')
        file.write_text(text[:start] + text[end:] + '\n\n' + text[start:end], encoding='utf-8')
        self.assertEqual(self.span(PY, 'recharge-def').last, len(file.read_text().splitlines()), 'it must be last')
        self.assertEqual(sa.verify(self.root), [], 'moving a cited def to the end of the file must not fail it')

    def test_c_a_changed_cited_statement_fails_changed(self):
        self.edit(PY, 'total = state + bonus', 'total = state - bonus')
        found = sa.verify(self.root)
        self.assertEqual(self.kinds(found), ['CHANGED'], found)
        self.assertTrue(any(p.startswith(f'CHANGED {PY}#recharge-sum: stored ') for p in found),
                        f'an operator change inside a cited statement must fail CHANGED: {found}')

    def test_d_a_deleted_needle_fails_missing(self):
        self.edit(PY, '    total = state + bonus  # carry the bonus\n', '')
        found = sa.verify(self.root)
        self.assertIn(f"MISSING {PY}#recharge-sum: 'total = state' is not in {PY} recharge_turn", found,
                      'deleting the needle line must fail MISSING')

    def test_e_a_duplicated_needle_fails_ambiguous(self):
        line = '    total = state + bonus  # carry the bonus\n'
        self.edit(PY, line, line + line)
        found = '\n'.join(sa.verify(self.root))
        self.assertIn(f"AMBIGUOUS {PY}#recharge-sum: 'total = state' is on 2 lines of {PY} recharge_turn:", found,
                      'a duplicated needle must fail AMBIGUOUS, never pick one')
        self.assertIn(f'{PY}:{at("total = state") + 1}: total = state + bonus', found)

    def test_f_a_comment_edit_on_a_cited_python_line_still_passes(self):
        self.edit(PY, '# carry the bonus', '# carry the bonus, then cap it below')
        self.assertEqual(sa.verify(self.root), [], 'a comment edit on a cited .py line must not fail it')

    def test_g_a_docstring_edit_inside_a_cited_stmt_def_still_passes(self):
        self.edit(PY, '"""Return the recharge turn."""', '"""Return the recharge turn,\n    capped at LIMIT."""')
        self.assertEqual(sa.verify(self.root), [], 'a docstring edit inside a cited stmt def must not fail it')

    def test_h_a_comment_edit_inside_a_cited_cjs_span_fails_changed(self):
        self.edit(CJS, '// cap at the limit', '// cap at three')
        found = sa.verify(self.root)
        self.assertEqual(len(found), 1, found)
        self.assertTrue(found[0].startswith(f'CHANGED {CJS}#cjs-total: stored '),
                        f'a comment edit in a non-.py span must fail CHANGED: {found}')
        self.assertIn('// cap at three', found[0], 'CHANGED carries the current span')


class RefreshTests(Scratch):
    def test_check_exit_codes(self):
        self.assertEqual(run('--root', str(self.root), '--check'), (0, 'anchors current\n'))
        self.edit(PY, 'return x * 2', 'return x * 3')
        code, out = run('--root', str(self.root), '--check')
        self.assertEqual(code, 1, out)
        self.assertTrue(out.startswith(f'CHANGED {PY}#inner-method: stored '), out)
        self.assertTrue(out.endswith('1 anchor problems\n'), out)

    def test_write_refreshes_only_the_changed_digest(self):
        before = self.stores()
        self.edit(PY, 'return x * 2', 'return x * 3')
        code, out = run('--root', str(self.root), '--write')
        self.assertEqual(code, 0, out)
        self.assertIn(f'refreshed {PY}#inner-method: ', out)
        self.assertIn('return x * 3', out)
        after = self.stores()
        self.assertEqual(after[CJS], before[CJS])
        changed = [(a, b) for a, b in zip(before[PY].decode().splitlines(), after[PY].decode().splitlines()) if a != b]
        self.assertEqual(len(changed), 1, changed)
        self.assertTrue(changed[0][1].startswith('  "inner-method": '), changed)
        old, new = (sa.parse_store(blob.decode()) for blob in (before[PY], after[PY]))
        self.assertNotEqual(old['anchors']['inner-method'].pop('digest'), new['anchors']['inner-method'].pop('digest'))
        self.assertEqual(old, new)
        self.assertEqual(run('--root', str(self.root), '--check')[0], 0)

    def test_write_restricted_to_paths(self):
        before = self.stores()
        self.edit(PY, 'return x * 2', 'return x * 3')
        self.edit(CJS, '// cap at the limit', '// cap at three')
        code, out = run('--root', str(self.root), '--write', CJS)
        self.assertEqual(code, 0, out)
        after = self.stores()
        self.assertEqual(after[PY], before[PY])
        self.assertNotEqual(after[CJS], before[CJS])
        code, out = run('--root', str(self.root), '--check')
        self.assertEqual((code, out.splitlines()[0].split(':')[0]), (1, f'CHANGED {PY}#inner-method'), out)

    def test_write_refuses_on_missing_and_writes_nothing(self):
        before = self.stores()
        self.edit(PY, 'return x * 2', 'return x * 3')
        self.edit(PY, '        y = x + 1\n', '')
        code, out = run('--root', str(self.root), '--write')
        self.assertEqual(code, 1, out)
        self.assertIn(f'MISSING {PY}#holder-method', out)
        self.assertIn('refusing to write', out)
        self.assertNotIn('CHANGED', out, 'a digest refresh cannot fix what it refuses on, so only the blockers print')
        self.assertEqual(self.stores(), before)

    def test_write_fills_a_new_entry_written_with_an_empty_digest(self):
        file = sa.store_file(self.root, PY)
        store = sa.load_store(file)
        store['anchors']['holder-return'] = dict(scope='Holder.method', needle='return y', extent='line', digest='',
                                                 cited_by=[B])
        file.write_text(sa.dump_store(store), encoding='utf-8')
        self.edit(B, 'only.', f'only, and {PY}#holder-return.')
        code, out = run('--root', str(self.root), '--check')
        self.assertTrue(out.startswith(f'CHANGED {PY}#holder-return: stored nothing, now '), out)
        self.assertEqual(run('--root', str(self.root), '--write')[0], 0)
        self.assertEqual(run('--root', str(self.root), '--check'), (0, 'anchors current\n'))

    def test_show(self):
        code, out = run('--root', str(self.root), '--show', f'{PY}#recharge-cap')
        self.assertEqual(code, 0, out)
        first, last = at('if total > LIMIT'), at('return total')
        self.assertEqual(out.splitlines()[:2], [f'{PY}#recharge-cap -> {PY}:{first}-{last}',
                                                f'    {first}:     if total > LIMIT:'])
        self.assertEqual(run('--root', str(self.root), '--show', f'{PY}#nope'), (1, f'no entry {PY}#nope\n'))

    @unittest.skipUnless(shutil.which('git'), 'git is not installed')
    def test_write_with_base_prints_old_against_new(self):
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='t',
                   GIT_AUTHOR_EMAIL='t@example.invalid', GIT_COMMITTER_NAME='t',
                   GIT_COMMITTER_EMAIL='t@example.invalid')
        for argv in (['init', '-q'], ['add', '-A'], ['commit', '-qm', 'base']):
            subprocess.run(['git', *argv], cwd=self.root, env=env, check=True, capture_output=True)
        self.edit(PY, 'class Holder:\n', 'class Holder:\n    pass\n\n')   # moves inner-method down two lines
        self.edit(PY, 'return x * 2', 'return x * 3')
        stored = sa.load_store(sa.store_file(self.root, PY))['anchors']['inner-method']['digest']
        code, out = run('--root', str(self.root), '--write', '--base', 'HEAD')
        self.assertEqual(code, 0, out)
        old, new = at('return x * 2'), at('return x * 2') + 2
        self.assertIn(f'    {new}:             return x * 3', out)
        self.assertIn(f'  at HEAD: {PY}:{old}, digest {stored}\n    {old}:             return x * 2', out)
        # A citation whose needle is newer than the ref says so rather than failing.
        self.edit(PY, '        return y\n', '        z = 9\n        return y\n')
        file = sa.store_file(self.root, PY)
        store = sa.load_store(file)
        store['anchors']['holder-z'] = dict(scope='Holder.method', needle='z = 9', extent='line', digest='',
                                            cited_by=[B])
        file.write_text(sa.dump_store(store), encoding='utf-8')
        code, out = run('--root', str(self.root), '--show', f'{PY}#holder-z', '--base', 'HEAD')
        self.assertEqual(code, 0, out)
        self.assertIn(f"  at HEAD: does not resolve (AnchorMissing: 'z = 9' is not in {PY} Holder.method)", out)


class HeadTests(unittest.TestCase):
    def test_the_repository_stores_are_current(self):
        self.assertEqual(sa.verify(ROOT), [])
        self.assertEqual(run('--root', str(ROOT), '--check'), (0, 'anchors current\n'))


if __name__ == '__main__':
    unittest.main()
