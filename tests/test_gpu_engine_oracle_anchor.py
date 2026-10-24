"""The authored anchor's exact partition law, not pool-frequency coverage."""
from fractions import Fraction
import importlib.util
import unittest
AVAILABLE=bool(importlib.util.find_spec('jax'))
if AVAILABLE:
    import numpy as np
    from gpu_engine.oracle_anchor import finite_rows,anchor_input,jax_distribution


def test_finite_partition_retains_all_damage_and_accuracy_mass():
    rows,weights=finite_rows()
    assert len(rows)==len(weights)==4096
    assert sum(weights)==1
    for side in range(2):
        assert sum(w for row,w in zip(rows,weights) if row[side,4]==0)==Fraction(1,16)
        assert sum(w for row,w in zip(rows,weights) if row[side,6]==0)==Fraction(95,100)
        for damage in range(16):
            assert sum(w for row,w in zip(rows,weights) if row[side,5]==damage)==Fraction(1,16)
    assert np.all(rows[:,:,5]<16)


def test_authored_anchor_preserves_exact_mass_and_all_four_hit_patterns():
    rows,weights=finite_rows();spec=anchor_input(0)
    result=jax_distribution(spec,rows,weights)
    assert sum(result.values())==1
    assert all(key[2:]==(7,7) for key in result)
    hit_pattern={(key[0]<spec['hp'],key[1]<spec['hp']) for key in result}
    assert hit_pattern=={(False,False),(True,False),(False,True),(True,True)}


def test_always_accurate_earthquake_has_no_zero_mass_support():
    rows,weights=finite_rows(100);spec=anchor_input(0,move='earthquake')
    assert len(weights)==1024 and sum(weights)==1 and all(w>0 for w in weights)
    result=jax_distribution(spec,rows,weights)
    assert sum(result.values())==1
    assert all(key[0]<spec['hp'] and key[1]<spec['hp'] for key in result)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(
        unittest.skipUnless(AVAILABLE,'requires isolated spike JAX')(function))
        for name,function in globals().items() if name.startswith('test_') and callable(function))
