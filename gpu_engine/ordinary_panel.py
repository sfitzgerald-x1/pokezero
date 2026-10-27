"""Prospective routing and auditable complete ordinary-carry cell emissions.

Routing inspects only a source-bound root. An admitted path never falls back
because of a cap, output or comparison result. Terminal audit reconstructs the
same input partition, then lifts both independently retained outputs per cell.
"""
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import hashlib
import json
import math

from . import carry_law as algebra, future_state as mapping, ordinary_law as ordinary
from .finite_law import COORDINATES,CardinalityRefusal

SCHEMA = 'pokezero.complete-ordinary-carry-panel.v1'
EVIDENCE = 'pokezero.complete-ordinary-cell-emissions.v1'
CRITERION = 'exact-rational-full-output-law'


def contract(entry):
    if 'ordinary_carry' not in entry:return None
    cfg=entry['ordinary_carry']
    if (type(cfg) is not dict or set(cfg)!={'schema','criterion','max_output_nodes'}
            or cfg['schema']!=SCHEMA or cfg['criterion']!=CRITERION
            or type(cfg['max_output_nodes']) is not int or cfg['max_output_nodes']<=0
            or entry.get('constructor_provenance') is None):
        raise ValueError('EXPLICIT_COMPLETE_ORDINARY_PROTOCOL_AND_ORIGINALS_REQUIRED')
    return cfg


def dispatch(route):
    return dict(schema=SCHEMA,route=route,criterion=CRITERION if route=='ordinary_carry'
        else 'legacy-per-key-tolerance' if route in ('legacy','conditional') else 'whole-slot-unit-unknown')


@dataclass(frozen=True)
class Dispatch:
    admission: dict
    plan: object


def ordinary_plan(state,actions,table,metadata,native,entry):
    try:return ordinary.plan_for_state(state,actions,table,metadata,cap=entry['max_cases'],native_input=native)
    except CardinalityRefusal as error:
        return mapping.Refused((mapping.Gap('ordinary_plan','CHANCE_CARDINALITY_CAP',str(error)),),
            mapping._freeze(mapping._sample(state,0)))


def consumer_certificate(plan):
    return dict(kind='ordinary_carry',source=plan.source_certificate,table=plan.table_certificate,
        axes=[asdict(axis) for axis in plan.law.axes],cardinality=plan.law.cardinality,
        native_input=plan.native_input.raw_native,native_actions=asdict(plan.native_actions),address=plan.address)


def route_current(state,actions,table,metadata,native,entry):
    """Pure native admission precedes planning/caps and all engine cases."""
    domain=ordinary.root_domain(native,actions)
    if not domain['supported']:
        from .chance_consumers import plan_for_state
        plan=plan_for_state(state,0,actions,table,metadata,cap=entry['max_cases'],native_input=native)
        return Dispatch(dict(dispatch('legacy'),native_domain_refusal=domain['reason']),plan)
    plan=ordinary_plan(state,actions,table,metadata,native,entry)
    return Dispatch(dispatch('ordinary_carry'),plan)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _axes(plan):
    rows=[]
    for axis in plan.axes:
        row=asdict(axis)
        row['observations']=[dict(kind=o.kind,argument={'fraction':[o.argument.numerator,o.argument.denominator]}
            if type(o.argument) is Fraction else o.argument) for o in axis.observations]
        rows.append(row)
    return json.loads(_canonical(rows))


def cell_binding(cell):
    return hashlib.sha256(_canonical([cell.partition_identity,cell.case.values,
        str(cell.case.mass),cell.domains]).encode()).hexdigest()


def encode_output(output):
    expressions=[]
    for name,value in output.expressions:
        if isinstance(value,algebra.Copy):row=dict(name=name,copy=value.coordinate)
        elif isinstance(value,algebra.Constant):row=dict(name=name,constant=value.value)
        else:raise ValueError('UNCLASSIFIED_OUTPUT_EXPRESSION')
        expressions.append(row)
    return dict(physical=output.physical,capsule=dict(version=1,schema='pokezero.suspended-turn',
        **dict(output.capsule)),expressions=expressions)


def emission(index,cell,jax,native):
    return dict(index=index,cell_sha256=cell_binding(cell),mass=str(cell.case.mass),
        jax=encode_output(jax),native=encode_output(native))


def evidence(plan,rows):
    return dict(schema=EVIDENCE,criterion=CRITERION,axes=_axes(plan),
        space=ordinary.SPACE.bounds,emissions=rows)


def _finite(value):
    if isinstance(value,(list,tuple)):
        for item in value:_finite(item)
    elif type(value) is dict:
        for item in value.values():_finite(item)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('NONFINITE_PHYSICAL_OUTPUT')


def _physical_schema(plan):
    parent=mapping.canonical_native_future(plan.native_input.state,plan.native_input.certificate)
    if isinstance(parent,mapping.Refused):raise ValueError('INVALID_BOUND_PHYSICAL_SCHEMA')
    return tuple((name,type(value)) for name,value in parent.key)


def _decode_output(row,physical_schema,address):
    if type(row) is not dict or set(row)!={'physical','capsule','expressions'}:
        raise ValueError('COMPLETE_OUTPUT_EVIDENCE_REQUIRED')
    if type(row['physical']) is not list or type(row['expressions']) is not list:
        raise ValueError('OUTPUT_EVIDENCE_TYPES')
    _finite(row['physical']);physical=mapping._freeze(row['physical']);hash(physical)
    if (len(physical)!=len(physical_schema)
            or any(type(pair) is not tuple or len(pair)!=2 or type(pair[0]) is not str
                for pair in physical)
            or tuple((pair[0],type(pair[1])) for pair in physical)!=physical_schema):
        raise ValueError('COMPLETE_UNIQUE_TYPED_PHYSICAL_LAYOUT_REQUIRED')
    capsule=ordinary._capsule(row['capsule']);expressions=[]
    for value in row['expressions']:
        if type(value) is not dict or set(value) not in ({'name','copy'},{'name','constant'}):
            raise ValueError('OUTPUT_EXPRESSION_EVIDENCE_SCHEMA')
        if type(value['name']) is not str:raise ValueError('OUTPUT_EXPRESSION_NAME')
        if 'copy' in value:
            if type(value['copy']) is not str or value['copy'] not in dict(COORDINATES):raise ValueError('OUTPUT_COPY_COORDINATE')
            expression=algebra.Copy(value['copy'])
        else:
            if type(value['constant']) is not int:raise ValueError('OUTPUT_CONSTANT_INTEGER')
            expression=algebra.Constant(value['constant'])
        expressions.append((value['name'],expression))
    output=ordinary.Output(physical,capsule,tuple(expressions))
    ordinary.validate_output_address(output,address)
    return output


def audit_evidence(payload,plan,entry):
    """No case execution: rebuild exact cells and both full output pushforwards."""
    cfg=contract(entry)
    if cfg is None:raise ValueError('UNREGISTERED_ORDINARY_OUTPUT_METHOD')
    if (type(payload) is not dict or set(payload)!={'schema','criterion','axes','space','emissions'}
            or payload['schema']!=EVIDENCE or payload['criterion']!=CRITERION
            or _canonical(payload['axes'])!=_canonical(_axes(plan.law))
            or _canonical(payload['space'])!=_canonical(ordinary.SPACE.bounds)
            or type(payload['emissions']) is not list):
        raise ValueError('BOUND_OUTPUT_PARTITION_SPACE_AND_CRITERION_REQUIRED')
    cells=tuple(algebra.execution_cells(plan.law,cap=entry['max_cases']))
    if len(payload['emissions'])!=len(cells):raise ValueError('INCOMPLETE_CELL_EMISSIONS')
    reference=[];native=[];physical_schema=_physical_schema(plan)
    for index,(row,cell) in enumerate(zip(payload['emissions'],cells,strict=True)):
        if (type(row) is not dict or set(row)!={'index','cell_sha256','mass','jax','native'}
                or type(row['index']) is not int or row['index']!=index
                or row['cell_sha256']!=cell_binding(cell) or type(row['mass']) is not str
                or row['mass']!=str(cell.case.mass)):
            raise ValueError('CELL_COVERAGE_OR_MASS_CHANGED')
        reference.append(_decode_output(row['jax'],physical_schema,plan.address).term(plan.law,cell))
        native.append(_decode_output(row['native'],physical_schema,plan.address).term(plan.law,cell))
    return algebra.compare(algebra.OutputLaw(ordinary.SPACE,tuple(reference)),
        algebra.OutputLaw(ordinary.SPACE,tuple(native)),mass_tolerance=Fraction(),
        max_nodes=cfg['max_output_nodes'])


def _batched(raw):
    import numpy as np
    from .state import BattleState
    return BattleState(**{name:np.asarray([raw[name]],dtype=np.uint32 if name in ('keys','volatile')
        else None) for name in BattleState._fields})


def audit_dispatch(root,context,entry):
    """Retained constructors and native root preflight only; no transition replay."""
    contract(entry)
    if root['source_error'] or root['terminal']:return dispatch('source_failure'),None
    from . import source_provenance as source
    ctor=root['constructor_provenance']
    if set(ctor)!={f.name for f in fields(source.ConstructorProvenance)}:
        return dispatch('refused'),None
    original=source.ConstructorProvenance(**{k:mapping._freeze(v) for k,v in ctor.items()})
    bound=source.ConstructorContext(**context['context'])
    runtime=source.RuntimeAttestation(**context['runtime'])
    metadata=bound.metadata();raw=root['state'];registered=entry['constructor_provenance']
    if (context['root_start']!=registered['root_start']
            or _canonical(context['runtime'])!=_canonical(registered['runtime'])
            or bound.catalog_sha256!=source._sha(registered['catalog'])
            or bound.pool_sha256!=entry['pool_sha256']):
        raise ValueError('AUDIT_CONTEXT_DIFFERS_FROM_REGISTERED_ENTRY')
    if (root['constructor_context_sha256']!=bound.sha256
            or original.root_id!=context['root_start']+root['slot']
            or tuple(root['teams'])!=original.team_indices):
        raise ValueError('AUDIT_ORIGINAL_ROOT_BINDING_CHANGED')
    source._context(bound,runtime)
    values=asdict(original);digest=values.pop('sha256')
    if source._sha(values)!=digest:raise ValueError('AUDIT_ORIGINAL_PROVENANCE_CHANGED')
    initial=dict(original.initial_fields)
    if any(mapping._freeze(raw[name])!=initial[name] for name in source._IMMUTABLE_FIELDS):
        raise ValueError('AUDIT_IMMUTABLE_CONSTRUCTOR_IDENTITY_CHANGED')
    if raw['pending_residual']:return dispatch('conditional'),None
    current=source.native_input_with_provenance(raw,None,metadata,provenance=original,context=bound,
        runtime=runtime,expected_root_id=original.root_id,expected_pool_sha256=entry['pool_sha256'])
    if isinstance(current,mapping.Refused):return dispatch('refused'),None
    domain=ordinary.root_domain(current,tuple(root['actions']))
    if not domain['supported']:
        return dict(dispatch('legacy'),native_domain_refusal=domain['reason']),None
    from .transition import tables_from_metadata
    plan=ordinary_plan(_batched(raw),tuple(root['actions']),tables_from_metadata(metadata),metadata,current,entry)
    return dispatch('ordinary_carry'),plan
