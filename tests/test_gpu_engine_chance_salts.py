"""Distinct chance streams must never fold the same per-turn key with one salt."""
import ast
from pathlib import Path
import unittest


class ChanceSaltTests(unittest.TestCase):
    def test_every_named_stream_has_a_unique_salt(self):
        # This owner branch predates Attract. Reserve its existing integration salt
        # so the collision is caught here as well as after train composition.
        salts={'INFATUATION_SALT':2236067}
        origins={'INFATUATION_SALT':'existing Attract stream'}

        def bind(target,value,origin):
            if isinstance(target,(ast.Tuple,ast.List)):
                self.assertIsInstance(value,(tuple,list),origin)
                self.assertEqual(len(target.elts),len(value),origin)
                for child,item in zip(target.elts,value):bind(child,item,origin)
            elif isinstance(target,ast.Name) and target.id.endswith('_SALT'):
                self.assertIs(type(value),int,origin)
                # Independent tape/oracle copies of one named stream must agree.
                if target.id in salts:self.assertEqual(salts[target.id],value,origin)
                salts[target.id]=value;origins[target.id]=origin

        root=Path(__file__).resolve().parents[1]/'gpu_engine'
        for path in sorted(root.rglob('*.py')):
            for node in ast.walk(ast.parse(path.read_text(),filename=str(path))):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AnnAssign) else []
                if not any(isinstance(n,ast.Name) and n.id.endswith('_SALT') for t in targets for n in ast.walk(t)):continue
                value=ast.literal_eval(node.value)
                for target in targets:bind(target,value,f'{path.relative_to(root)}:{node.lineno}')
        self.assertIn('SHED_SKIN_SALT',salts)
        by_value={}
        for name,value in salts.items():
            self.assertNotIn(value,by_value,f'{name} ({origins[name]}) collides with {by_value.get(value)} at {value}')
            by_value[value]=name


if __name__=='__main__':unittest.main()
