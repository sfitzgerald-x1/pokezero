"""Static observer wiring checks: no engine import, device, or game execution."""
import ast
import builtins
import hashlib
from pathlib import Path
import subprocess
import symtable
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PARENT = "c1faadcc20345261b13fd9d0eab208e19577f77b"
SOURCE = ROOT / "gpu_engine/transition.py"


def _step(source):
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "step_with_chance")


def _calls(source):
    step = _step(source)
    found = {}
    for node in ast.walk(step):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "substitute_damage_callbacks"):
            if node.func.attr in ("route", "Pending"):
                if node.func.attr in found:
                    raise AssertionError("duplicate Substitute observer call")
                found[node.func.attr] = node
    if set(found) != {"route", "Pending"}:
        raise AssertionError("missing original Substitute observer calls")
    return found


def _undefined_step_globals(source):
    module = symtable.symtable(source, str(SOURCE), "exec")
    bound = {s.get_name() for s in module.get_symbols()
             if s.is_assigned() or s.is_imported() or s.is_namespace()}
    bound.update(vars(builtins))
    bound.update(("__name__", "__file__", "__package__", "__doc__",
                  "__builtins__", "__spec__", "__loader__"))
    step = next(t for t in module.get_children() if t.get_name() == "step_with_chance")

    def descend(table):
        yield table
        for child in table.get_children():
            yield from descend(child)

    referenced = {s.get_name() for table in descend(step) for s in table.get_symbols()
                  if s.is_referenced() and s.is_global()}
    return referenced - bound


class _Grid:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        rows, side = key
        if rows != (0,):
            raise AssertionError("foreign fixture row")
        return self.values[side]


def _lexical_fixture(source, resolve):
    """Evaluate the original type assignment/resolve with inert scalar operands."""
    move = 130
    work = SimpleNamespace(active=_Grid((3, 4)), happiness=(255, 0), level=(80, 81),
                           hidden_power_type=(7, 10), hidden_power_power=(63, 50),
                           hp=(100, 90), max_hp=(100, 100), substitute_hp=(0, 25))
    env = dict(table=SimpleNamespace(power={move: 70}, type={move: 1}, variable_moves=object()),
               work=work, move=move, side=0, foe=1, rows=(0,), physical=False,
               _active=lambda value, state, side: value[side],
               _direct_hit_context=object(), _sub_damage_ticket=object(),
               _sub_damage_requested=True, acts=True, hit=True, immune=False,
               _risk_protected=False, _risk_absorbed=False, bypass_sub=False,
               enduring=_Grid((False, False)), caller_move=move)

    def variable_resolve(table, actual_move, power, typ, physical, hp_type, hp_power,
                         happiness, hp, max_hp):
        if (actual_move, power, typ, hp_type, hp_power) != (move, 70, 1, 7, 63):
            raise AssertionError("original current-member resolver operands changed")
        return hp_power, hp_type, physical

    env["variable"] = SimpleNamespace(resolve=variable_resolve)
    step = _step(source)
    initial = next(n for n in ast.walk(step) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "(power, typ)")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[initial], type_ignores=[])),
                 str(SOURCE), "exec"), {}, env)
    if resolve:
        assignment = next(n for n in ast.walk(step) if isinstance(n, ast.Assign)
                          and isinstance(n.value, ast.Call)
                          and ast.unparse(n.value.func) == "variable.resolve")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                     str(SOURCE), "exec"), {}, env)
    env["substitute_damage_callbacks"] = SimpleNamespace(
        route=lambda *args, **kwargs: kwargs,
        Pending=lambda *args: args)
    return env


class TransitionObserverWiringTests(unittest.TestCase):
    def test_actual_observer_calls_resolve_current_type(self):
        source = SOURCE.read_text()
        for resolve, expected_type in ((False, 1), (True, 7)):
            env = _lexical_fixture(source, resolve)
            self.assertNotIn("move_type", env)
            for name, call in _calls(source).items():
                with self.subTest(resolve=resolve, observer=name):
                    result = eval(compile(ast.Expression(call), str(SOURCE), "eval"), {}, env)
                    actual = result["resolved_type"] if name == "route" else result[-1]
                    self.assertEqual(actual, expected_type)
                    self.assertEqual(actual, env["typ"])

    def test_current_function_has_no_undefined_observer_global(self):
        source = SOURCE.read_text()
        self.assertEqual(_undefined_step_globals(source), set())
        broken = source.replace("resolved_type=typ)", "resolved_type=observer_type_typo)", 1)
        self.assertNotEqual(source, broken)
        self.assertEqual(_undefined_step_globals(broken), {"observer_type_typo"})

    def test_parent_failure_and_only_two_name_repair(self):
        parent = subprocess.check_output(
            ["git", "show", PARENT + ":gpu_engine/transition.py"], cwd=ROOT, text=True)
        self.assertEqual(_undefined_step_globals(parent), {"move_type"})
        env = _lexical_fixture(parent, True)
        for name, call in _calls(parent).items():
            with self.subTest(parent_observer=name):
                with self.assertRaisesRegex(NameError, "move_type"):
                    eval(compile(ast.Expression(call), str(SOURCE), "eval"), {}, env)
        fixed = subprocess.check_output(
            ["git", "show", "afc80dc70525b7eb8723233015ce6495614bef08:gpu_engine/transition.py"],
            cwd=ROOT, text=True)
        self.assertEqual(parent.count("resolved_type=move_type)"), 1)
        self.assertEqual(parent.count(",move_type)"), 1)
        expected = parent.replace("resolved_type=move_type)", "resolved_type=typ)").replace(
            ",move_type)", ",typ)")
        self.assertEqual(fixed, expected)

    def test_owned_source_bindings_match_actual_files(self):
        def literal(file, name):
            tree = ast.parse((ROOT / "gpu_engine" / file).read_text())
            return ast.literal_eval(next(n.value for n in tree.body
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in n.targets)))

        for file, name, count in (
                # §6 pins residual_order.py (R1) beside transition.py: 89 and 14.
                ("engine_source_frame_witness.py", "SOURCE_PINS", 89),
                ("chance_consumers.py", "SOURCE_PINS", 14),
                ("ordinary_law.py", "MAPPING_PINS", 6),
                ("conditional_law.py", "MAPPING_PINS", 4)):
            pins = literal(file, name)
            self.assertEqual(len(pins), count)
            for target, expected in pins.items():
                with self.subTest(binding=file, target=target):
                    actual = hashlib.sha256((ROOT / "gpu_engine" / target).read_bytes()).hexdigest()
                    self.assertEqual(actual, expected)
        modules = ("action_history absorption baton combat critical damage_moves effects encore "
                   "entry focus_punch items linked_traps mechanics phazing pressure pursuit residual_order secondary "
                   "self_destruct sleep_talk solar_beam state sustain transition variable_moves weather").split()
        domain = hashlib.sha256(b"".join(name.encode() + b"\0" +
            (ROOT / "gpu_engine" / (name + ".py")).read_bytes() + b"\0"
            for name in modules)).hexdigest()
        for file, name in (("chance_consumers.py", "DOMAIN_SHA256"),
                           ("ordinary_law.py", "DOMAIN_SHA256"),
                           ("conditional_law.py", "KERNEL_SHA256")):
            self.assertEqual(literal(file, name), domain)


if __name__ == "__main__":
    unittest.main()
