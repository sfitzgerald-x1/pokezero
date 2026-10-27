"""Opt-in complete ordinary output laws with independently produced carry.

This adapter changes no panel admission or historical result. Native domain
cuts and actual JAX-table cuts refine the SAME physical coordinates. Source
proofs lift forwarded coordinates into copy/constant expressions; the complete
pushforward is compared without enumerating a Cartesian saved random bank.
Unsupported positive outputs, caps and source drift refuse the whole slot.
The owning supervisor must retain its operation deadline.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
from importlib import import_module
import json
from pathlib import Path

from . import carry_law as algebra, chance_consumers, conditioned_future, future_state as mapping
from .fidelity_panel import JaxCaseExecutor, WholeSlotUnknown, chunks
from .finite_law import Axis, COORDINATES, DRAW_BOUNDS, CardinalityRefusal, Observation, Plan


# Complete transition-domain proof, including the four copy/zero assignments
# in transition.step_with_chance. A numeric representative alone is not proof.
DOMAIN_SHA256 = '81b43f80a616f5102d0f07b5bc980328004e3911785e2d28d1d8285e585b8e3b'
# Filled only after reviewed implementation stabilizes; drift must refuse.
MAPPING_PINS = {'carry_law.py': '19bb86991bc09212a46d849bdd3d7629044512c1d8b78f0e79487ddeae45237e', 'chance_consumers.py': '049ff5c20fa3ce98b9820808eb7dccfe8c8113336dfa55f3d2c59dc7853ef1df', 'conditioned_future.py': 'b3d8c647f0aca36204e1dcd521a1ce17fee7463f683217f3d0830eb9e44c2e11', 'future_state.py': '77fe87f3af696dcc8e3785dcf6481e83075b781c464fa3c736c305745628b0a8', 'fidelity_panel.py': 'ce67f0d036d0daf8b6d72e1727a7fc5aa22e73dacb2e4dbd4d496a94a8c1973e', 'finite_law.py': 'd999cdbaddebc33cf3f84583e584e2110d7ee79f37bf365f715d0eb7f424ae8c'}
BANK = tuple([(f'residual_draws.{s}.{i}', f'draws.{s}.{i}', bound)
    for s in range(2) for i,bound in enumerate(DRAW_BOUNDS)]
    + [(f'{field}_draws.{s}', f'{field}.{s}', bound)
        for field,bound in (('phazing',60),('secondary',100),('sleep_talk',60)) for s in range(2)])
SPACE = algebra.OutputSpace(tuple((name,bound) for name,_,bound in BANK))
BANK_FIELDS = {'residual_draws','phazing_draws','secondary_draws','sleep_talk_draws'}


def _source_certificate():
    root=Path(__file__).parent
    digest=hashlib.sha256(b''.join(name.encode()+b'\0'+(root/(name+'.py')).read_bytes()+b'\0'
        for name in chance_consumers.DOMAIN_MODULES)).hexdigest()
    if digest!=DOMAIN_SHA256:raise ValueError('ORDINARY_CONSUMER_DOMAIN_DRIFT')
    if not MAPPING_PINS:raise ValueError('ORDINARY_MAPPING_PROOF_NOT_REGISTERED')
    for name,expected in MAPPING_PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('ORDINARY_MAPPING_DRIFT: '+name)
    from scripts import engine_build_fingerprint as fingerprint
    issues=fingerprint.check()
    if issues:raise ValueError('ORDINARY_NATIVE_BUILD_NOT_CURRENT: '+repr(issues))
    native_fingerprint=fingerprint.compute_fingerprint()['fingerprint']
    if native_fingerprint!='481094fe7d3b6efc2e2b6d34c9ae4ca8c662fb35f52dfc40e048a55c9ffaaa53':
        raise ValueError('ORDINARY_NATIVE_CONSUMER_DOMAIN_DRIFT')
    return (digest,tuple(MAPPING_PINS.items()),native_fingerprint,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def refine(jax_plan, native_requirements, *, cap):
    """Both engines must be constant on each resulting execution cell."""
    if not isinstance(jax_plan,Plan) or type(cap) is not int or cap<=0:
        raise ValueError('BOUND_PLAN_AND_POSITIVE_CAP_REQUIRED')
    cuts={name:[] for name,_ in COORDINATES}
    if type(native_requirements) is not list:raise ValueError('NATIVE_CONSUMERS_REQUIRED')
    for row in native_requirements:
        if type(row) is not dict or set(row)!={'coordinate','kind','argument'}:
            raise ValueError('NATIVE_CONSUMER_SCHEMA')
        name,kind,arg=(row[x] for x in ('coordinate','kind','argument'))
        if type(name) is not str or name not in cuts or type(kind) is not str or type(arg) is not int:
            raise ValueError('NATIVE_CONSUMER_COORDINATE_OR_TYPE')
        cuts[name].append(Observation(kind,arg))
    axes=[]
    for axis in jax_plan.axes:
        observations=tuple(dict.fromkeys((*axis.observations,*cuts[axis.coordinate])))
        uses=(*axis.source_uses,*(('native.conditioned_step.requirements',) if cuts[axis.coordinate] else ()))
        axes.append(Axis(axis.coordinate,observations,tuple(uses)) if observations else
            Axis(axis.coordinate,(),inactive_reason='JAX source domain and admitted native ordinary domain have no physical consumer'))
    result=Plan(tuple(axes))
    # Precount the common refinement before either engine executes a prefix.
    result.bounded_cases(cap)
    return result


def _capsule(values):
    if type(values) is not dict or set(values)!={*conditioned_future.CAPSULE_FIELDS,'version','schema'}:
        raise ValueError('ORDINARY_CAPSULE_FIELDS')
    if type(values['version']) is not int or values['version']!=1 or values['schema']!='pokezero.suspended-turn':
        raise ValueError('ORDINARY_CAPSULE_SCHEMA')
    out={}
    for name in conditioned_future.CAPSULE_FIELDS:
        value=mapping._freeze(values[name]);wanted=bool if name in ('pending_residual','forced_switch','baton_pass') else int
        if mapping._shape(value)!=mapping.SHAPES[name] or any(type(v) is not wanted for v in conditioned_future._leaves(value)):
            raise ValueError('ORDINARY_CAPSULE_TYPE_OR_SHAPE: '+name)
        out[name]=value
    if out['queued_actor'] not in (-1,0,1) or any(not 0<=v<4 for v in out['queued_actions']):
        raise ValueError('ORDINARY_CAPSULE_QUEUE')
    if any(v not in (0,2,4,8) for v in out['queued_stall']):raise ValueError('ORDINARY_CAPSULE_STALL')
    for name in ('keys','turn','replacement_count'):
        if any(not 0<=v<2**32 for v in conditioned_future._leaves(out[name])):
            raise ValueError('ORDINARY_CAPSULE_ADDRESS: '+name)
    for name,_,bound in BANK:
        if not 0<=_bank_value(out,name)<bound:raise ValueError('ORDINARY_CAPSULE_BANK: '+name)
    return tuple((name,out[name]) for name in conditioned_future.CAPSULE_FIELDS)


def _bank_value(capsule,name):
    field,*indices=name.split('.');value=capsule[field]
    for index in indices:value=value[int(index)]
    return value


@dataclass(frozen=True)
class Output:
    physical: tuple
    capsule: tuple
    expressions: tuple

    def term(self,plan,cell):
        values=dict(cell.case.values);capsule=dict(self.capsule)
        for (name,expression),(wanted,_,_) in zip(self.expressions,BANK,strict=True):
            if name!=wanted:raise ValueError('INCOMPLETE_ORDERED_BANK_OUTPUT')
            value=values[expression.coordinate] if isinstance(expression,algebra.Copy) else expression.value
            if _bank_value(capsule,name)!=value:
                raise ValueError('NUMERIC_OUTPUT_DOES_NOT_MATCH_COPY_CONSTANT_PROOF: '+name)
        constant={name:value for name,value in self.capsule if name not in BANK_FIELDS}
        physical=tuple(sorted((*self.physical,*mapping._flat({'capsule':constant}))))
        return algebra.output_term(plan,cell,physical,self.expressions,SPACE)


def _jax_output(state,index,metadata,certificate):
    data=mapping._sample(state,index)
    expected,gaps,_,last,damage=mapping._prepare(data,metadata,certificate,boundary='coupled-output')
    if gaps:raise WholeSlotUnknown('UNSUPPORTED_POSITIVE_JAX_OUTPUT',asdict(mapping.Refused(tuple(gaps))))
    if (last,damage)!=(certificate.conditional_last,certificate.conditional_damage):
        raise ValueError('ORDINARY_CONDITIONAL_HISTORY_CHANGED')
    capsule=_capsule({**{name:data[name] for name in conditioned_future.CAPSULE_FIELDS},
        'version':1,'schema':'pokezero.suspended-turn'})
    # Source proof: admitted roots are live ordinary requests, hence not
    # resuming; transition copies these exact fresh inputs iff defer_residual.
    # This proof is pinned transitively, rather than inferred from values at
    # a representative. The native expressions are produced independently.
    pending=data['pending_residual']
    expressions=tuple((name,algebra.Copy(coordinate) if pending else algebra.Constant(0)) for name,coordinate,_ in BANK)
    return Output(tuple(mapping._flat(expected)),capsule,expressions)


def _native_output(branch,certificate):
    if type(branch) is not dict or set(branch)!={'percentage','state','capsule','bank','instructions'}:
        raise ValueError('ORDINARY_NATIVE_BRANCH_SCHEMA')
    if type(branch['percentage']) not in (int,float) or branch['percentage']!=100:
        raise ValueError('FIXED_NATIVE_CASE_MUST_BE_DETERMINISTIC')
    physical=mapping._canonical_native_future(branch['state'],certificate,boundary='coupled-output')
    if isinstance(physical,mapping.Refused):
        raise WholeSlotUnknown('UNSUPPORTED_POSITIVE_NATIVE_OUTPUT',asdict(physical))
    capsule=_capsule(branch['capsule'])
    if type(branch['bank']) is not list or len(branch['bank'])!=len(BANK):
        raise ValueError('COMPLETE_NATIVE_BANK_OUTPUT_REQUIRED')
    expressions=[]
    for row,(name,_,_) in zip(branch['bank'],BANK,strict=True):
        if type(row) is not dict or set(row) not in ({'value','copy'},{'value','constant'}):
            raise ValueError('NATIVE_BANK_EXPRESSION_SCHEMA')
        if type(row['value']) is not int or row['value']!=_bank_value(dict(capsule),name):
            raise ValueError('NATIVE_BANK_VALUE_MISMATCH')
        if 'copy' in row:
            if type(row['copy']) is not str or row['copy'] not in dict(COORDINATES):
                raise ValueError('NATIVE_BANK_UNBOUND_COPY')
            expression=algebra.Copy(row['copy'])
        else:
            if type(row['constant']) is not int:raise ValueError('NATIVE_BANK_CONSTANT_TYPE')
            expression=algebra.Constant(row['constant'])
        expressions.append((name,expression))
    return Output(physical.key,capsule,tuple(expressions))


@dataclass(frozen=True)
class OrdinaryPlan:
    law: Plan
    jax_plan: chance_consumers.ConsumerPlan
    address: tuple
    source_certificate: tuple

    @property
    def native_input(self):return self.jax_plan.native_input
    @property
    def native_actions(self):return self.jax_plan.native_actions
    @property
    def table_certificate(self):return self.jax_plan.table_certificate
    @property
    def metadata(self):return self.jax_plan.metadata

    def canonical_successor(self,state,index):
        return _jax_output(state,index,self.metadata,self.native_input.certificate)


def plan_for_state(state,actions,table,metadata,*,cap,native_input):
    source=_source_certificate()
    if not isinstance(native_input,mapping.NativeInput):
        raise ValueError('RETAINED_CURRENT_ROOT_NATIVE_INPUT_REQUIRED')
    native=native_input
    data=mapping._sample(state,0)
    slots=mapping._freeze(actions)
    if len(slots)!=2 or any(type(v) is not int or not 0<=v<4 for v in slots):
        raise ValueError('ORDINARY_EXACT_MOVE_SLOTS_REQUIRED')
    if data['done'] or data['pending_residual'] or any(data['forced_switch']) or any(data['baton_pass']):
        raise ValueError('LIVE_ORDINARY_ROOT_REQUIRED')
    if not 0<=data['turn']<2**31-1:raise ValueError('ORDINARY_JAX_TURN_RANGE')
    jax_plan=chance_consumers.plan_for_state(state,0,slots,table,metadata,cap=cap,native_input=native,
        exact_move_slots=True)
    if isinstance(jax_plan,mapping.Refused):return jax_plan
    module=import_module('pokezero_search')
    report=json.loads(module.conditioned_step_requirements(native.raw_native,json.dumps(slots)))
    if set(report)!={'version','schema','requirements'} or report['version']!=1 or report['schema']!='pokezero.native-step-consumers':
        raise ValueError('NATIVE_REQUIREMENTS_SCHEMA')
    law=refine(jax_plan.law,report['requirements'],cap=cap)
    import jax
    # Bind advancement before any transition is executed. Never read it from
    # a JAX successor to label a native one.
    next_keys=tuple(int(v) for v in jax.random.split(state.keys[0],2)[0].tolist())
    address=(('turn',data['turn']),('keys',data['keys']),('next_keys',next_keys))
    return OrdinaryPlan(law,jax_plan,address,source)


def compare_slot(state,actions,table,metadata,*,cap,batch_size,max_nodes,native_input):
    """No partial or normalized laws; an UNKNOWN has zero identified mass."""
    try:
        plan=plan_for_state(state,actions,table,metadata,cap=cap,native_input=native_input)
        if isinstance(plan,mapping.Refused):
            raise WholeSlotUnknown('INPUT_OR_PLAN_REFUSED',asdict(plan))
        cells=tuple(algebra.execution_cells(plan.law,cap=cap))
        execute=JaxCaseExecutor(state,actions,table,plan,batch_size=batch_size)
        reference=[];native=[];module=import_module('pokezero_search')
        for batch in chunks(iter(cells),batch_size):
            outputs=execute(tuple(cell.case for cell in batch))
            for cell,output in zip(batch,outputs,strict=True):
                reference.append(output.term(plan.law,cell))
                v=dict(cell.case.values)
                chance={'draws':[[v[f'draws.{s}.{i}'] for i in range(12)] for s in range(2)],
                    **{field:[v[f'{field}.{s}'] for s in range(2)] for field in ('phazing','secondary','sleep_talk','entry_ties')},
                    'initial_tie':v['initial_tie']}
                request={'version':1,'schema':'pokezero.ordinary-step','actions':plan.native_actions.raw_actions,
                    'chance':chance,'address':dict(plan.address)}
                report=json.loads(module.conditioned_step(plan.native_input.raw_native,json.dumps(request)))
                if set(report)!={'version','schema','branches'} or report['version']!=1 or report['schema']!='pokezero.ordinary-step-result':
                    raise ValueError('ORDINARY_NATIVE_REPORT_SCHEMA')
                if type(report['branches']) is not list or len(report['branches'])!=1:
                    raise ValueError('FIXED_NATIVE_CASE_MUST_BE_DETERMINISTIC')
                native.append(_native_output(report['branches'][0],plan.native_input.certificate).term(plan.law,cell))
        if _source_certificate()!=plan.source_certificate:raise ValueError('ORDINARY_SOURCES_CHANGED_DURING_EXECUTION')
        result=algebra.compare(algebra.OutputLaw(SPACE,tuple(reference)),algebra.OutputLaw(SPACE,tuple(native)),
            mass_tolerance=Fraction(),max_nodes=max_nodes)
        return dict(status='AGREE' if result.half_l1==0 else 'DIVERGE',identified_mass='1',unknown_mass='0',
            execution_cells=len(cells),comparison=asdict(result),scientific_qualification=False)
    except (ValueError,WholeSlotUnknown,CardinalityRefusal) as error:
        return dict(status='UNKNOWN',identified_mass='0',unknown_mass='1',partial_laws_accepted=False,
            category=getattr(error,'category',type(error).__name__),detail=str(error),scientific_qualification=False)
