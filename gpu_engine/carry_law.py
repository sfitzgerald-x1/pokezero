"""Exact cell-domain/copy/constant output laws, without raw-bank expansion.

This algebra supplies no engine coupling or consumer proof. Each execution cell
must come from a partition refined for BOTH engines, and physical outputs must
be constant on that cell. Each engine must independently produce its copy and
constant expressions. In particular, a JAX capsule cannot label native output.

Forwarded coordinates retain their complete cell domains. Repeated copies are
one random variable, not independent draws. Closed outputs can aggregate across
cells. Equality compares the resulting output law, never execution paths.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math

from .finite_law import COORDINATES, Case, CardinalityRefusal, Plan


@lru_cache(maxsize=128)
def _partition_identity(plan):
    def argument(value):
        return [value.numerator, value.denominator] if isinstance(value, Fraction) else value
    payload=[(axis.coordinate,[(o.kind,argument(o.argument)) for o in axis.observations],
              axis.source_uses,axis.inactive_reason) for axis in plan.axes]
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutionCell:
    case: Case
    domains: tuple
    partition_identity: str

    def validate(self, plan):
        if not isinstance(plan,Plan) or self.partition_identity!=_partition_identity(plan):
            raise ValueError('CELL_PARTITION_BINDING_CHANGED')
        if (not isinstance(self.case,Case) or type(self.case.mass) is not Fraction
                or not isinstance(self.domains,tuple) or not isinstance(self.case.values,tuple)):
            raise ValueError('EXACT_EXECUTION_CELL_REQUIRED')
        names=tuple(name for name,_ in COORDINATES)
        if tuple(name for name,_ in self.domains)!=names or tuple(name for name,_ in self.case.values)!=names:
            raise ValueError('COMPLETE_COORDINATE_DOMAINS_REQUIRED')
        mass=Fraction(1)
        for (name,members),(_,value),cells in zip(self.domains,self.case.values,plan.cells(),strict=True):
            if type(value) is not int or not isinstance(members,tuple) or any(type(v) is not int for v in members):
                raise ValueError('EXACT_IMMUTABLE_CELL_VALUES_REQUIRED')
            matching=[cell for cell in cells if cell.members==members and cell.representative==value]
            if len(matching)!=1:
                raise ValueError('CELL_DOMAIN_OR_REPRESENTATIVE_CHANGED: '+name)
            mass*=matching[0].mass
        if mass!=self.case.mass:
            raise ValueError('CELL_MASS_CHANGED')


def execution_cells(plan, *, cap):
    """Precount before yielding; legacy Case and its serialized schema stay intact."""
    if not isinstance(plan,Plan) or type(cap) is not int or cap<=0:
        raise ValueError('EXPLICIT_PLAN_AND_POSITIVE_CAP_REQUIRED')
    cells=plan.cells();count=math.prod(map(len,cells))
    if count>cap:raise CardinalityRefusal(count,cap)
    identity=_partition_identity(plan);names=tuple(name for name,_ in COORDINATES)
    return (ExecutionCell(Case(tuple(zip(names,(c.representative for c in row),strict=True)),
                    math.prod((c.mass for c in row),start=Fraction(1))),
                tuple(zip(names,(c.members for c in row),strict=True)),identity)
            for row in itertools.product(*cells))


@dataclass(frozen=True)
class Copy:
    coordinate: str


@dataclass(frozen=True)
class Constant:
    value: int


@dataclass(frozen=True)
class OutputSpace:
    bounds: tuple

    def __post_init__(self):
        self.validate()

    def validate(self):
        if (type(self.bounds) is not tuple or not self.bounds
                or any(type(row) is not tuple or len(row)!=2 for row in self.bounds)):
            raise ValueError('IMMUTABLE_UNIQUE_OUTPUT_COORDINATES_REQUIRED')
        if (any(type(name) is not str or not name or type(bound) is not int or bound<=0
                    for name,bound in self.bounds)
                or len({name for name,_ in self.bounds})!=len(self.bounds)):
            raise ValueError('IMMUTABLE_UNIQUE_OUTPUT_COORDINATES_REQUIRED')


def _canonical_pattern(pattern):
    """Variable names are internal; only output aliases and value sets are observable."""
    names={};out=[]
    for term in pattern:
        if term[0]==0:out.append(term)
        elif len(term[2])==1:out.append((0,term[2][0]))
        else:
            identity=names.setdefault(term[1],len(names))
            out.append((1,identity,term[2]))
    return tuple(out)


@dataclass(frozen=True)
class OutputTerm:
    physical: tuple
    pattern: tuple
    mass: Fraction

    @property
    def support_cardinality(self):
        domains={term[1]:term[2] for term in self.pattern if term[0]==1}
        return math.prod(map(len,domains.values()))


def output_term(plan, cell, physical, carry, space, *, branch_mass=Fraction(1)):
    """Bind a constant physical output and independently supplied carry expressions."""
    cell.validate(plan)
    if not isinstance(space,OutputSpace) or not isinstance(physical,tuple):
        raise ValueError('BOUND_OUTPUT_SPACE_AND_PHYSICAL_KEY_REQUIRED')
    hash(physical)
    if type(branch_mass) is not Fraction or branch_mass<=0:
        raise ValueError('POSITIVE_EXACT_BRANCH_MASS_REQUIRED')
    if not isinstance(carry,tuple) or tuple(name for name,_ in carry)!=tuple(name for name,_ in space.bounds):
        raise ValueError('COMPLETE_ORDERED_CARRY_EXPRESSIONS_REQUIRED')
    domains=dict(cell.domains);pattern=[]
    for (name,expression),(_,bound) in zip(carry,space.bounds,strict=True):
        if isinstance(expression,Constant):
            if type(expression.value) is not int or not 0<=expression.value<bound:
                raise ValueError('INVALID_CONSTANT_OUTPUT: '+name)
            pattern.append((0,expression.value))
        elif isinstance(expression,Copy):
            if expression.coordinate not in domains:
                raise ValueError('UNBOUND_COPY_COORDINATE: '+name)
            members=domains[expression.coordinate]
            if any(type(v) is not int or not 0<=v<bound for v in members):
                raise ValueError('INVALID_POSITIVE_COPY_DOMAIN: '+name)
            pattern.append((1,expression.coordinate,members))
        else:raise ValueError('UNSUPPORTED_CARRY_EXPRESSION: '+name)
    return OutputTerm(physical,_canonical_pattern(pattern),cell.case.mass*branch_mass)


@dataclass(frozen=True)
class OutputLaw:
    space: OutputSpace
    terms: tuple

    @property
    def mass(self):
        return sum((term.mass for term in self.terms),start=Fraction())

    def validate(self):
        if not isinstance(self.space,OutputSpace) or not isinstance(self.terms,tuple) or not self.terms:
            raise ValueError('COMPLETE_OUTPUT_LAW_REQUIRED')
        self.space.validate()
        for term in self.terms:
            if (not isinstance(term,OutputTerm) or type(term.mass) is not Fraction or term.mass<=0
                    or not isinstance(term.physical,tuple) or not isinstance(term.pattern,tuple)
                    or len(term.pattern)!=len(self.space.bounds)):
                raise ValueError('INVALID_POSITIVE_OUTPUT_TERM')
            identities={}
            for expression,(_,bound) in zip(term.pattern,self.space.bounds,strict=True):
                if not isinstance(expression,tuple) or not expression or type(expression[0]) is not int:
                    raise ValueError('INVALID_CARRY_EXPRESSION')
                if expression[0]==0:
                    if len(expression)!=2 or type(expression[1]) is not int or not 0<=expression[1]<bound:
                        raise ValueError('INVALID_CONSTANT_OUTPUT')
                elif expression[0]==1 and len(expression)==3:
                    _,identity,members=expression
                    if (type(identity) is not int or identity<0 or not isinstance(members,tuple)
                            or not members or tuple(sorted(set(members)))!=members
                            or any(type(v) is not int or not 0<=v<bound for v in members)
                            or (identity in identities and identities[identity]!=members)):
                        raise ValueError('INVALID_COPY_DOMAIN_OR_ALIAS')
                    identities[identity]=members
                else:raise ValueError('UNSUPPORTED_CARRY_EXPRESSION')


class ComparisonNodeCap(ValueError):
    def __init__(self,cap):
        self.cap=cap
        super().__init__('SYMBOLIC_COMPARISON_NODE_CAP: '+str(cap)+'; no partial comparison returned')


@dataclass(frozen=True)
class Comparison:
    reference_only: int
    native_only: int
    max_probability_difference: Fraction
    half_l1: Fraction
    reference_mass: Fraction
    native_mass: Fraction
    visited_nodes: int
    witness: tuple | None


def compare(reference, native, *, mass_tolerance, max_nodes):
    """Compare complete atomic pushforwards; equivalent refinements may merge.

    The bounded recursion groups primitive values with identical remaining
    constraints. It never expands an independent Cartesian bank. Complicated
    alias mixtures may still exceed the explicit node cap: that is a whole-law
    refusal, not a prefix result or a normalization.
    """
    reference.validate();native.validate()
    if reference.space!=native.space:raise ValueError('OUTPUT_SPACE_MISMATCH')
    if type(mass_tolerance) is not Fraction or mass_tolerance<0:
        raise ValueError('EXPLICIT_EXACT_MASS_TOLERANCE_REQUIRED')
    if type(max_nodes) is not int or max_nodes<=0:raise ValueError('POSITIVE_NODE_CAP_REQUIRED')
    if reference.mass!=1 or abs(native.mass-1)>mass_tolerance:
        raise ValueError('INCOMPLETE_OR_INVALID_OUTPUT_MASS')
    grouped={}
    for side,law in enumerate((reference,native)):
        for term in law.terms:
            rows=grouped.setdefault(term.physical,{})
            values=rows.setdefault(term.pattern,[Fraction(),Fraction()])
            values[side]+=term.mass/term.support_cardinality

    def signature(rows):
        return tuple((pattern,*values) for pattern,values in sorted(rows.items()) if any(values))

    visited=0
    @lru_cache(maxsize=None)
    def measure(rows):
        nonlocal visited
        visited+=1
        if visited>max_nodes:raise ComparisonNodeCap(max_nodes)
        if all(j==n for _,j,n in rows):return 0,0,Fraction(),Fraction(),None
        if not rows:return 0,0,Fraction(),Fraction(),None
        if not rows[0][0]:
            j=sum((j for _,j,_ in rows),start=Fraction())
            n=sum((n for _,_,n in rows),start=Fraction());delta=abs(j-n)
            return int(j>0 and n==0),int(n>0 and j==0),delta,delta,() if delta else None
        values=set()
        for pattern,_,_ in rows:
            term=pattern[0];values.update((term[1],) if term[0]==0 else term[2])
        children={}
        for value in sorted(values):
            child={}
            for pattern,j,n in rows:
                first=pattern[0]
                if first[0]==0:
                    if value!=first[1]:continue
                    rest=pattern[1:]
                else:
                    if value not in first[2]:continue
                    rest=tuple((0,value) if term[0]==1 and term[1]==first[1] else term
                               for term in pattern[1:])
                rest=_canonical_pattern(rest)
                weights=child.setdefault(rest,[Fraction(),Fraction()]);weights[0]+=j;weights[1]+=n
            key=signature(child);children.setdefault(key,[]).append(value)
        only_j=only_n=0;maximum=l1=Fraction();witness=None
        for child,values in children.items():
            j,n,delta,error,example=measure(child);count=len(values)
            only_j+=count*j;only_n+=count*n;maximum=max(maximum,delta);l1+=count*error
            if witness is None and example is not None:witness=(values[0],*example)
        return only_j,only_n,maximum,l1,witness

    only_j=only_n=0;maximum=l1=Fraction();witness=None
    for physical,rows in grouped.items():
        j,n,delta,error,example=measure(signature(rows))
        only_j+=j;only_n+=n;maximum=max(maximum,delta);l1+=error
        if witness is None and example is not None:witness=(physical,example)
    return Comparison(only_j,only_n,maximum,l1/2,reference.mass,native.mass,visited,witness)
