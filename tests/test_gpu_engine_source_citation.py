"""Paired movement/change controls for each regenerated citation consumer."""
import contextlib
import importlib.util
import io
import json
import keyword
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tokenize
import unittest
from unittest.mock import patch

from gpu_engine import source_citation as citations
from scripts import refresh_citation_artifacts as refresh
from scripts import schema_container_census as schema

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CitationTests(unittest.TestCase):
    def test_rust_visibility_impl_and_same_named_methods_have_distinct_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);file = root/'sample.rs'
            file.write_text('impl One {\n    pub(crate) fn build() {\n        call("}");\n    }\n}\n'
                            'impl Two {\n    pub fn build() {\n        call("}");\n    }\n}\n')
            one, two = [citations.citation_at('sample.rs', n, root) for n in (3, 8)]
            self.assertIn('::One.build[', one);self.assertIn('::Two.build[', two)
            self.assertEqual(citations.verify_text(one+' '+two, root), [])

    def test_python_move_documentation_and_changed_multiline_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root/'sample.py'
            original = 'def f():\n    return call(\n        1,\n    )\n'
            file.write_text(original)
            cite = citations.citation_at('sample.py', 2, root)
            moved = '\n\ndef unrelated():\n    pass\n\n'+original.replace('    return', '    "documentation"\n    # comment\n    return')
            file.write_text(moved)
            self.assertEqual(citations.verify_text(cite, root), [])
            self.assertEqual(citations.citation_at('sample.py', moved.splitlines().index('    return call(')+1, root), cite)
            file.write_text(moved.replace('        1,', '        2,'))
            self.assertEqual(citations.verify_text(cite, root), ['CHANGED '+cite.rsplit('@', 1)[0]])

    def test_duplicate_and_removed_python_statements_name_the_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);file = root/'sample.py'
            file.write_text('def f():\n    value = 1\n')
            cite = citations.citation_at('sample.py', 2, root)
            for source, failure in [('def f():\n    value = 1\n    value = 1\n', 'AMBIGUOUS'),
                                    ('def f():\n    pass\n', 'MISSING')]:
                file.write_text(source)
                errors = citations.verify_text(cite, root)
                self.assertEqual(len(errors), 1)
                self.assertIn(failure, errors[0]);self.assertIn(cite.rsplit('@', 1)[0], errors[0])

    def test_rust_patch_call_covers_arguments_and_has_no_hunk_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);file = root/'sample.patch'
            original = '@@ -10,2 +10,5 @@ fn f() {\n+    call(\n+        1,\n+    );\n'
            file.write_text(original)
            cite = citations.citation_at('sample.patch', 2, root)
            self.assertNotIn('@@', cite)
            self.assertIn('1,', cite)
            file.write_text('\n'+original.replace('-10,2 +10,5', '-20,2 +20,5'))
            self.assertEqual(citations.verify_text(cite, root), [])
            self.assertEqual(citations.citation_at('sample.patch', 3, root), cite)
            file.write_text(file.read_text().replace('+        1,', '+        2,'))
            self.assertIn(cite.rsplit('@', 1)[0], citations.verify_text(cite, root)[0])


class ArtifactControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c153 = json.loads((ROOT/refresh.C153).read_text())
        cls.c154 = json.loads((ROOT/refresh.C154).read_text())
        cls.before153 = refresh.regenerate_c153(cls.c153)
        cls.before154 = refresh.regenerate_c154(cls.c154)
        cls.before155 = refresh.c155.source_citation_facts()
        text = '\n'.join(refresh.strings([cls.before153, cls.before154, cls.before155]))
        cls.paths = {m['path'] for m in citations.PATTERN.finditer(text)}
        cls.paths.update(['scripts/c153_wide_negative_census.py', 'scripts/c154_unreachable_readjudication.py',
                          'src/pokezero/randbat_vocab.py'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in self.paths:
            target = self.root/name;target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)

    def fresh(self, number):
        # Re-execute the actual generators under the moved source tree, including
        # their import-time demonstrations; no citation field is masked.
        with patch.object(sys, 'path', list(sys.path)):
            c153 = load(self.root/'scripts/c153_wide_negative_census.py', 'scratch_c153')
            with patch.dict(sys.modules, {'c153_wide_negative_census': c153}):
                with patch.object(refresh, 'c153', c153), patch.object(refresh, 'ROOT', self.root):
                    if number == 153:
                        return refresh.regenerate_c153(self.c153)
                    c154 = load(self.root/'scripts/c154_unreachable_readjudication.py', 'scratch_c154')
                    with patch.object(refresh, 'c154', c154):
                        return refresh.regenerate_c154(self.c154)

    def changed_statement_fails(self, record, path):
        text = '\n'.join(refresh.strings(record))
        cite = next(m.group(0) for m in citations.PATTERN.finditer(text) if m['path'] == path)
        span, source, _ = citations.resolve(cite, self.root)
        rows = source.splitlines(keepends=True)
        if path.endswith('.py'):
            token = next(t for t in tokenize.generate_tokens(io.StringIO(source).readline)
                         if span.first <= t.start[0] <= span.last and t.type == tokenize.NAME
                         and not keyword.iskeyword(t.string))
            line, col = token.start;end = token.end[1]
            rows[line-1] = rows[line-1][:col]+token.string+'_changed'+rows[line-1][end:]
        else:
            rows[span.first-1] = re.sub(r'\b(?!fn\b|pub\b|if\b|return\b|let\b|mut\b)[A-Za-z_]\w*',
                                      lambda m: m[0]+'_changed', rows[span.first-1], count=1)
        changed = ''.join(rows)
        self.assertNotEqual(changed, source)
        (self.root/path).write_text(changed)
        errors = citations.verify_text(text, self.root)
        self.assertTrue(any(cite.rsplit('@', 1)[0] in error for error in errors), errors)

    def shift(self, path):
        file = self.root/path
        file.write_text('\n\n\n'+file.read_text())

    def test_c153_shift_rederives_identically_and_statement_change_names_citation(self):
        path = 'scripts/engine_transition_differential.py'
        self.shift(path)
        self.assertEqual(self.fresh(153), self.before153)
        self.changed_statement_fails(self.before153, path)

    def test_c154_shift_rederives_identically_and_statement_change_names_citation(self):
        path = 'scripts/engine_transition_differential.py'
        self.shift(path)
        self.assertEqual(self.fresh(154), self.before154)
        self.changed_statement_fails(self.before154, path)

    def test_c155_shift_rederives_identically_and_statement_change_names_citation(self):
        path = refresh.c155.EVENTS
        for name in (path, refresh.c155.THRESHOLD_PATCH, refresh.c155.LEDGER):self.shift(name)
        after = refresh.c155.source_citation_facts(self.root)
        self.assertEqual(after, self.before155)
        self.assertEqual(citations.verify_text('\n'.join(after.values()), self.root), [])
        self.changed_statement_fails(self.before155, path)

    def test_c154_unrelated_rust_string_continuation_keeps_the_citing_function(self):
        path = refresh.c154.EV
        text = '\n'.join(refresh.strings(self.before154))
        calls = {m.group(0) for m in citations.PATTERN.finditer(text)
                 if m['path'] == path and json.loads(m['needle']).startswith('render_action_phase(')}
        self.assertEqual(len(calls), 2)
        for cite in calls:self.assertIn('::render_branch_events[', cite)
        file = self.root/path
        source = file.read_text()
        continuation = '; '+chr(92)+'\n             add it to SUBCASE_VOCABULARY'
        self.assertEqual(source.count(continuation), 1)
        file.write_text(source.replace(continuation, '; add it to SUBCASE_VOCABULARY'))
        self.assertEqual(citations.verify_text('\n'.join(calls), self.root), [])
        self.assertEqual(self.fresh(154), self.before154)

    def test_c154_r9_predicate_inversion_changes_its_citation_and_digest(self):
        text = '\n'.join(refresh.strings(self.before154['verdicts']['R9']))
        cite = next(m.group(0) for m in citations.PATTERN.finditer(text)
                    if m['scope'] == 'ResidualPlan.build' and 'Abilities::LIQUIDOOZE' in json.loads(m['needle']))
        span, source, _ = citations.resolve(cite, self.root)
        rows = source.splitlines(keepends=True)
        self.assertIn('!= Abilities::LIQUIDOOZE', rows[span.first-1])
        rows[span.first-1] = rows[span.first-1].replace('!= Abilities::LIQUIDOOZE', '== Abilities::LIQUIDOOZE')
        (self.root/span.path).write_text(''.join(rows))
        updated = citations.citation_at(span.path, span.first, self.root)
        self.assertNotEqual(citations.PATTERN.fullmatch(cite)['digest'], citations.PATTERN.fullmatch(updated)['digest'])
        errors = citations.verify_text(cite, self.root)
        self.assertTrue(any(cite.rsplit('@', 1)[0] in error for error in errors), errors)


class SchemaControls(unittest.TestCase):
    def test_real_inline_key_survives_shift_but_changed_member_names_owner(self):
        spec = schema.classification()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in {key[0] for key in spec}:
                target = root/path;target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, target)
            spec_path = root/'tests/data/schema_drill_schema_containers.txt'
            spec_path.parent.mkdir(parents=True);shutil.copyfile(schema.SPEC, spec_path)
            target = root/'scripts/export_encoder_tables.py'
            target.write_text('\n\n'+target.read_text())
            with patch.object(schema, 'REPO', root), patch.object(schema, 'SPEC', spec_path), patch.object(sys, 'argv', ['census']):
                with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(schema.main(), 0)
                found, _ = schema.containers()
                row = next(row for row in found if row[0] == str(target.relative_to(root)) and row[3].startswith('<inline:'))
                lines = target.read_text().splitlines(keepends=True)
                # Change a member of this actual inline container while keeping
                # two recognized schema names, so its classification must fail.
                import ast
                tree = ast.parse(''.join(lines))
                node = next(n for n in ast.walk(tree) if isinstance(n, (ast.List, ast.Tuple, ast.Set)) and n.lineno == row[1])
                chosen = next(n for n in node.elts if isinstance(n, ast.Constant) and n.value == 'v2.2')
                lines[chosen.lineno-1] = lines[chosen.lineno-1][:chosen.col_offset]+"'v2.1'"+lines[chosen.end_lineno-1][chosen.end_col_offset:]
                target.write_text(''.join(lines))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):self.assertEqual(schema.main(), 15)
                self.assertIn(row[3], output.getvalue())


if __name__ == '__main__':
    unittest.main()
