"""Exact algebra checks against independent exhaustive small input programs.

These are authored mathematical tests, not engine consumer/coupling proof,
scientific roots, retrospective panel replay or coverage qualification.
"""
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
import itertools
import math
import unittest

from gpu_engine import carry_law as c
from gpu_engine.finite_law import Axis,COORDINATES,Observation,Plan,CardinalityRefusal

A='draws.0.0';B='draws.1.0'


def plan(active=()):
    return Plan(tuple(Axis(name,(Observation('identity'),),('authored toy consumer',))
        if name in active else Axis(name,(),inactive_reason='authored toy physical output has no consumer')
        for name,_ in COORDINATES))


def law(fn,active,carry,*,space=None):
    p=plan(active);space=space or c.OutputSpace((('left',2),('right',2)))
    terms=[]
    for cell in c.execution_cells(p,cap=16):
        values=dict(cell.case.values)
        terms.append(c.output_term(p,cell,('physical',fn(values[A],values[B])),
            tuple(zip((name for name,_ in space.bounds),carry,strict=True)),space))
    return c.OutputLaw(space,tuple(terms))


def dense_program(fn,carry):
    """Execute the tiny source program on all four original primitive inputs."""
    out=defaultdict(F)
    for a,b in itertools.product(range(2),repeat=2):
        values={A:a,B:b}
        result=tuple(e.value if isinstance(e,c.Constant) else values[e.coordinate] for e in carry)
        out[(('physical',fn(a,b)),result)]+=F(1,4)
    return dict(out)


def dense_metrics(reference,native):
    keys=set(reference)|set(native)
    differences=[abs(reference.get(k,F())-native.get(k,F())) for k in keys]
    return (len(set(reference)-set(native)),len(set(native)-set(reference)),
        max(differences,default=F()),sum(differences,F())/2)


class CarryLawTests(unittest.TestCase):
    def comparison(self,a,b,nodes=10000):
        return c.compare(a,b,mass_tolerance=F(),max_nodes=nodes)

    def test_same_physical_and_carry_marginals_do_not_hide_wrong_correlation(self):
        reference=law(lambda a,b:a,(A,),(c.Copy(A),c.Constant(0)))
        p=plan((A,));space=reference.space;terms=[]
        for cell in c.execution_cells(p,cap=2):
            value=dict(cell.case.values)[A]
            terms.append(c.output_term(p,cell,('physical',value),
                (('left',c.Constant(1-value)),('right',c.Constant(0))),space))
        result=self.comparison(reference,c.OutputLaw(space,tuple(terms)))
        self.assertEqual((result.reference_only,result.native_only,result.half_l1),(2,2,F(1)))

    def test_equivalent_refinements_and_closed_output_aggregation_compare_equal(self):
        a=law(lambda a,b:0,(),(c.Copy(A),c.Copy(B)))
        b=law(lambda a,b:0,(A,B),(c.Copy(A),c.Copy(B)))
        out=self.comparison(a,b);self.assertEqual(out.half_l1,0)
        self.assertEqual((out.reference_only,out.native_only),(0,0))
        # Different per-input physical behavior, same CLOSED output law.
        a=law(lambda a,b:a,(A,),(c.Constant(0),c.Constant(0)))
        b=law(lambda a,b:1-a,(A,),(c.Constant(0),c.Constant(0)))
        out=self.comparison(a,b);self.assertEqual(out.half_l1,0)
        self.assertIsNone(out.witness)

    def test_repeated_copy_is_one_variable_and_preserves_its_alias(self):
        same=law(lambda a,b:0,(),(c.Copy(A),c.Copy(A)))
        independent=law(lambda a,b:0,(),(c.Copy(A),c.Copy(B)))
        self.assertEqual(same.terms[0].support_cardinality,2)
        self.assertEqual(independent.terms[0].support_cardinality,4)
        result=self.comparison(same,independent)
        self.assertEqual((result.reference_only,result.native_only,result.half_l1),(0,2,F(1,2)))
        self.assertEqual(result.max_probability_difference,F(1,4))

    def test_complete_thirty_coordinate_bank_is_compact_and_copy_zero_is_detected(self):
        forwarded=[(name,bound) for name,bound in COORDINATES
            if name.startswith(('draws.','phazing.','secondary.','sleep_talk.'))]
        self.assertEqual(len(forwarded),30)
        space=c.OutputSpace(tuple(forwarded));p=plan();cell=next(c.execution_cells(p,cap=1))
        expressions=tuple((name,c.Copy(name)) for name,_ in forwarded)
        good=c.output_term(p,cell,('physical',0),expressions,space)
        count=math.prod(bound for _,bound in forwarded)
        self.assertEqual(good.support_cardinality,count);self.assertGreater(count,2**64)
        original=c.OutputLaw(space,(good,));result=self.comparison(original,original,nodes=1)
        self.assertEqual(result.visited_nodes,1);self.assertEqual(result.half_l1,0)
        broken=c.output_term(p,cell,('physical',0),
            tuple((name,c.Constant(0) if name==A else e) for name,e in expressions),space)
        result=self.comparison(original,c.OutputLaw(space,(broken,)),nodes=100)
        self.assertEqual(result.reference_only,count//2)
        self.assertEqual(result.native_only,0);self.assertEqual(result.half_l1,F(1,2))
        self.assertLessEqual(result.visited_nodes,64)

    def test_native_mass_is_never_normalized_and_incomplete_laws_refuse(self):
        a=law(lambda a,b:0,(),(c.Copy(A),c.Copy(B)))
        b=replace(a,terms=(replace(a.terms[0],mass=F(10001,10000)),))
        with self.assertRaisesRegex(ValueError,'MASS'):self.comparison(a,b)
        result=c.compare(a,b,mass_tolerance=F(1,1000),max_nodes=100)
        self.assertEqual(result.native_mass,F(10001,10000))
        self.assertEqual(result.half_l1,F(1,20000))
        complete=law(lambda a,b:a,(A,),(c.Copy(A),c.Copy(B)))
        with self.assertRaisesRegex(ValueError,'MASS'):
            self.comparison(replace(complete,terms=complete.terms[:1]),complete)

    def test_cell_domain_mass_and_certificate_are_bound_before_output(self):
        p=plan();cell=next(c.execution_cells(p,cap=1));space=c.OutputSpace((('x',2),))
        for bad in (replace(cell,case=replace(cell.case,mass=F(1,2))),
                    replace(cell,partition_identity='bad'),
                    replace(cell,domains=tuple((name,(0,)) if name==A else (name,members)
                        for name,members in cell.domains))):
            with self.assertRaises(ValueError):c.output_term(p,bad,(),(('x',c.Copy(A)),),space)
        with self.assertRaisesRegex(ValueError,'PARTITION'):
            c.output_term(plan((A,)),cell,(),(('x',c.Copy(A)),),space)
        with self.assertRaisesRegex(ValueError,'POSITIVE_COPY_DOMAIN'):
            c.output_term(p,cell,(),(('x',c.Copy('secondary.0')),),space)
        with self.assertRaises(CardinalityRefusal):c.execution_cells(plan((A,B)),cap=3)

    def test_symbolic_node_cap_returns_no_partial_comparison(self):
        a=law(lambda a,b:0,(),(c.Copy(A),c.Copy(B)))
        b=law(lambda a,b:0,(),(c.Constant(0),c.Copy(B)))
        with self.assertRaises(c.ComparisonNodeCap):self.comparison(a,b,nodes=1)
        self.assertEqual(self.comparison(a,b,nodes=20).half_l1,F(1,2))

    def test_output_space_rejects_nested_mutable_or_malformed_rows(self):
        for bounds in ((['x',2],),(('x',2,3),),(('x',True),),(('x',2),('x',2))):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):c.OutputSpace(bounds)
        value=law(lambda a,b:0,(),(c.Copy(A),c.Copy(B)))
        forged=object.__new__(c.OutputSpace);object.__setattr__(forged,'bounds',(['left',2],('right',2)))
        with self.assertRaises(ValueError):self.comparison(replace(value,space=forged),value)
        for tag in (False,True,0.0,1.0):
            term=replace(value.terms[0],pattern=((tag,0),(0,0)))
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):self.comparison(replace(value,terms=(term,)),value)

    def test_exact_comparison_matches_exhaustive_small_source_programs(self):
        functions=[(lambda a,b:0,()),(lambda a,b:1,()),(lambda a,b:a,(A,)),
            (lambda a,b:b,(B,)),(lambda a,b:1-a,(A,)),(lambda a,b:a^b,(A,B)),
            (lambda a,b:a&b,(A,B)),(lambda a,b:a|b,(A,B))]
        expressions=(c.Constant(0),c.Constant(1),c.Copy(A),c.Copy(B));programs=[]
        for fn,active in functions:
            for carry in itertools.product(expressions,repeat=2):
                programs.append((law(fn,active,carry),dense_program(fn,carry)))
        for index,(left,reference) in enumerate(programs):
            for offset in (1,17,63):
                right,native=programs[(index+offset)%len(programs)]
                actual=self.comparison(left,right)
                self.assertEqual((actual.reference_only,actual.native_only,
                    actual.max_probability_difference,actual.half_l1),dense_metrics(reference,native))
                if actual.witness is not None:
                    key=actual.witness
                    self.assertNotEqual(reference.get(key,F()),native.get(key,F()))
