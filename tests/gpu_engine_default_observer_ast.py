"""Test-only normalization of explicit, validated observer-disabled defaults.

Only predicates made from the listed private parameters and literal constants
can be evaluated. Boolean operators follow Python's ordered short circuit:
a known observer predicate may make its later operands unreachable. Unknown
operands that execute, numerical calls/keywords, classes and other parameters
remain in the comparison.
"""
import ast
from copy import deepcopy


OBSERVER_DEFAULTS = {
    '_public_trace': None, '_lifecycle_trace': False, '_weather_trace': None,
    '_status_trace': None, '_weather_hp_context': None,
    '_leftovers_context': None, '_status_residual_context': None,
    '_sleep_powder_context': None, '_thunder_wave_context': None, '_spikes_context': None, '_calm_mind_context': None, '_swords_dance_context': None, '_direct_hit_context': None, '_substitute_context': None, '_recoil_context': None,
}


class DefaultObserverProgram(ast.NodeTransformer):
    def truth(self, test):
        if isinstance(test, ast.BoolOp) and isinstance(test.op, (ast.And, ast.Or)):
            for operand in test.values:
                result = self.truth(operand)
                # An unknown prefix still executes, even if a later operand
                # would force the final boolean. Never erase that prefix.
                if result is None:
                    return None
                if isinstance(test.op, ast.And) and result is False:
                    return False
                if isinstance(test.op, ast.Or) and result is True:
                    return True
            return result
        allowed = (ast.Expression, ast.Name, ast.Load, ast.Constant, ast.Compare,
                   ast.Is, ast.IsNot, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not)
        nodes = list(ast.walk(test))
        names = {node.id for node in nodes if isinstance(node, ast.Name)}
        if not names or not names <= OBSERVER_DEFAULTS.keys():
            return None
        if any(not isinstance(node, allowed) for node in nodes):
            return None
        return bool(eval(compile(ast.Expression(test), '<observer-default-proof>', 'eval'),
                         {'__builtins__': {}}, OBSERVER_DEFAULTS))

    def visit_If(self, node):
        result = self.truth(node.test)
        if result is None:
            return self.generic_visit(node)
        statements = []
        for statement in node.body if result else node.orelse:
            value = self.visit(statement)
            if isinstance(value, list):
                statements.extend(value)
            elif value is not None:
                statements.append(value)
        return statements

    def visit_IfExp(self, node):
        result = self.truth(node.test)
        return (self.generic_visit(node) if result is None
                else self.visit(node.body if result else node.orelse))

    def visit_FunctionDef(self, node):
        keep = []
        for index, (arg, default) in enumerate(zip(node.args.kwonlyargs, node.args.kw_defaults)):
            if arg.arg not in OBSERVER_DEFAULTS:
                keep.append(index)
                continue
            expected = OBSERVER_DEFAULTS[arg.arg]
            if (not isinstance(default, ast.Constant) or type(default.value) is not type(expected)
                    or default.value != expected):
                raise ValueError('changed explicit observer default: ' + arg.arg)
        node.args.kwonlyargs = [node.args.kwonlyargs[i] for i in keep]
        node.args.kw_defaults = [node.args.kw_defaults[i] for i in keep]
        return self.generic_visit(node)


def default_program(tree):
    return DefaultObserverProgram().visit(deepcopy(tree))


def dump_default_program(tree):
    return ast.dump(default_program(tree), include_attributes=False)


def assert_hp_rng_mutations_detected(case, tree, baseline, dump=dump_default_program):
    """Poison real HP input, returned RNG and call keyword, never a trace."""
    # Choose writes in the retained default branch, rather than a disabled
    # observer branch that legitimately contains a separate keys assignment.
    hp = default_program(tree)
    call = next(node for node in ast.walk(hp) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'major_status_residual')
    call.args[0] = ast.BinOp(call.args[0], ast.Add(), ast.Constant(1))
    case.assertNotEqual(dump(hp), baseline, 'HP mutation escaped the default proof')
    rng = default_program(tree)
    call = next(node for node in ast.walk(rng) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == '_replace'
                and any(keyword.arg == 'keys' for keyword in node.keywords))
    keyword = next(keyword for keyword in call.keywords if keyword.arg == 'keys')
    keyword.value = ast.BinOp(keyword.value, ast.Add(), ast.Constant(1))
    case.assertNotEqual(dump(rng), baseline, 'RNG mutation escaped the default proof')
    keywords = default_program(tree)
    call = next(node for node in ast.walk(keywords) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'major_status_residual')
    keyword = next(keyword for keyword in call.keywords if keyword.arg == 'active')
    keyword.arg = 'changed_numerical_active_keyword'
    case.assertNotEqual(dump(keywords), baseline, 'call keyword mutation escaped the default proof')
