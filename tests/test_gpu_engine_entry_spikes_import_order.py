"""Cold import/build regression only; no model, engine step or game."""
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
PARENT='5ca765f559cf67993f8bccb6b4a25962a9d01a2d'
MODULES=('gpu_engine.benchmark_retained_games','gpu_engine.engine_lifecycle_event_device',
    'gpu_engine.engine_full_v4_event_device')


class EntrySpikesImportOrderTests(unittest.TestCase):
    def test_cold_module_orders_in_fresh_processes(self):
        script='import importlib,json,sys\nfor name in json.loads(sys.argv[1]):importlib.import_module(name)\n'
        for order in itertools.permutations(MODULES):
            with self.subTest(order=order):
                result=subprocess.run([sys.executable,'-B','-c',script,json.dumps(order)],cwd=ROOT,
                    env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
                        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT)),
                    text=True,capture_output=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_all_guard_and_mathematical_ast_unchanged_after_stripping_imports(self):
        path='gpu_engine/engine_entry_spikes_callbacks.py'
        old=subprocess.check_output(['git','show',PARENT+':'+path],cwd=ROOT,text=True)
        class Strip(ast.NodeTransformer):
            def visit_Import(self,node):return None
            def visit_ImportFrom(self,node):return None
        self.assertEqual(ast.dump(Strip().visit(ast.parse(old))),
            ast.dump(Strip().visit(ast.parse((ROOT/path).read_text()))))
        tree=ast.parse((ROOT/path).read_text())
        self.assertFalse(any(isinstance(n,ast.ImportFrom)and any(a.asname=='ex'for a in n.names)for n in tree.body))
        local=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)and any(
            isinstance(x,ast.ImportFrom)and any(a.asname=='ex'for a in x.names)for x in n.body)]
        self.assertEqual(local,['namespaces','claimed','begin','finish'])

    def test_single_owned_pin_and_unchanged_twenty_five_module_domain(self):
        gpu=ROOT/'gpu_engine'
        def assignment(path,name):
            return next(n.value for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
        pins=ast.literal_eval(assignment(gpu/'engine_source_frame_witness.py','SOURCE_PINS'))
        self.assertEqual(len(pins),76)
        for name,digest in pins.items():self.assertEqual(hashlib.sha256((gpu/name).read_bytes()).hexdigest(),digest,name)
        value=assignment(gpu/'chance_consumers.py','DOMAIN_MODULES')
        modules=value.args[0].func.value.value.split()
        self.assertEqual(len(modules),25);self.assertNotIn('engine_entry_spikes_callbacks',modules)
        actual=hashlib.sha256(b''.join(n.encode()+b'\0'+(gpu/(n+'.py')).read_bytes()+b'\0'for n in modules)).hexdigest()
        for file,name in (('chance_consumers.py','DOMAIN_SHA256'),('ordinary_law.py','DOMAIN_SHA256'),('conditional_law.py','KERNEL_SHA256')):
            self.assertEqual(ast.literal_eval(assignment(gpu/file,name)),actual)


if __name__=='__main__':unittest.main()
