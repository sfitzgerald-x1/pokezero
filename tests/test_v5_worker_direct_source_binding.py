"""Evaluate the worker's actual compiler expression with original static inputs.

No worker execution, constructor, model, forward, game or sampler is invoked.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import unittest

from pokezero import v5_worker as worker
from gpu_engine import engine_direct_hit_controls as direct
from gpu_engine import engine_request_source as source
import test_gpu_engine_no_damage_owning as fixtures


class WorkerDirectSourceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NoDamageOwningTests.setUpClass();cls.f=fixtures.NoDamageOwningTests
        cls.rules=source._checked(cls.f.req)['rules']

    def compiler_call(self):
        tree=ast.parse(Path(worker.__file__).read_text())
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
            and n.func.attr=='compile_direct_hit_controls']
        self.assertEqual(len(calls),1)
        return tree,calls[0]

    def actual(self,*,rules=None,metadata=None):
        _,call=self.compiler_call()
        return eval(compile(ast.Expression(deepcopy(call)),'<actual worker compiler>','eval'),dict(
            hit=direct,lifecycle=self.f.producer,metadata=self.f.metadata if metadata is None else metadata,
            rules=self.rules if rules is None else rules))

    def test_worker_actual_expression_enables_only_original_bound_secondary_source(self):
        compiled=self.actual();data=direct.check_compiled(compiled)
        self.assertIsNotNone(data.secondary_status)
        index=self.f.metadata['vocabulary']['moves'].index('thunderbolt')
        self.assertTrue(bool(data.secondary_status.candidates[index]))
        legacy=direct.compile_direct_hit_controls(engine_catalog=self.f.producer)
        self.assertIsNone(legacy.device.secondary_status)
        with self.assertRaises(ValueError):self.actual(rules=source.SourceRules(self.rules.device))
        altered=json.loads(json.dumps(self.f.metadata));altered['vocabulary']['moves'][index]='foreign'
        with self.assertRaises(ValueError):self.actual(metadata=altered)

    def test_worker_entire_ast_delta_only_original_metadata_and_rules_arguments(self):
        tree,call=self.compiler_call()
        self.assertEqual({k.arg:ast.unparse(k.value)for k in call.keywords},
            dict(engine_catalog='lifecycle',metadata='metadata',source_rules='rules'))
        call.keywords=[k for k in call.keywords if k.arg=='engine_catalog']
        baseline=ast.parse(subprocess.check_output(['git','show',
            '40204d5254ac08427b138c29aea1ba15fcd2dbe7:src/pokezero/v5_worker.py'],text=True))
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(baseline,include_attributes=False))


if __name__=='__main__':unittest.main()
