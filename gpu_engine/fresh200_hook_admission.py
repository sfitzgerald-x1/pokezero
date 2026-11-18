"""Independent source-catalog and installed-hook folds, never physical parity.

The source-only producer must supply parsed FULL function spans from pinned
files before launching Native. This module verifies those exact spans; it is
not a JavaScript parser and does not manufacture declarations from a receipt.
Present source hashes are not historical attestations. Pinned parser/source
closure admission and the real Native catalog builder remain separate gates.
"""
import copy
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
import re
import struct
from types import MappingProxyType

from . import fresh200_witness_audit as native


def _need(value,message):
    if not value:raise ValueError(message)


def _wire(value,*,sort=False):
    return (json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=sort,allow_nan=False)+'\n').encode('utf8')


def _function(value,sha,token):
    return type(value) is dict and set(value)=={'$function','function_token'} and value['$function']==sha and value['function_token']==token


@dataclass(frozen=True)
class FunctionLocation:
    source: str
    start: int
    end: int
    declaration: str
    source_form: str


@dataclass(frozen=True)
class FunctionEntry:
    operation: str
    originalSha256: str
    kind: str
    stage: str
    locations: tuple


@dataclass(frozen=True)
class SourceCatalog:
    parser_sha256: str
    source_pins: tuple
    entries: tuple
    sha256: str = field(init=False)
    _lookup: object = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        _need(native._hex(self.parser_sha256) and type(self.source_pins) is tuple and self.source_pins
            and type(self.entries) is tuple and self.entries,'sealed source/parser/function catalog required')
        _need(len({x.operation for x in self.entries})==len(self.entries),'duplicate catalog operation')
        document=dict(schema='pokezero.native_function_catalog.v1',parser_sha256=self.parser_sha256,
            sources=[dict(path=p,sha256=s) for p,s in self.source_pins],
            functions=[dict(operation=e.operation,originalSha256=e.originalSha256,kind=e.kind,stage=e.stage,
                locations=[vars(x) for x in e.locations]) for e in self.entries])
        object.__setattr__(self,'sha256',native.digest(_wire(document,sort=True)))
        object.__setattr__(self,'_lookup',MappingProxyType({x.operation:x for x in self.entries}))

    def entry(self,operation):
        _need(operation in self._lookup,'hook operation is absent from independent source catalog')
        return self._lookup[operation]


def seal_parsed_catalog(source_pins,functions,*,parser_sha256,read_source=None):
    """Verify parser-declared complete function spans from exact known paths.

    The independently pinned source parser is responsible for declaration and
    inheritance resolution; byte-substring discovery is not substituted for it.
    Files are never enumerated. Ambiguous identical bodies retain all locations.
    """
    _need(type(source_pins) is dict and source_pins and len(source_pins)<=2000
        and native._hex(parser_sha256),'independent bounded source/parser pins required')
    _need(type(functions) in (list,tuple) and 0<len(functions)<=100000,'bounded parsed function declarations required')
    loader=(lambda path:Path(path).read_bytes()) if read_source is None else read_source
    sources={};total=0
    for path,sha in source_pins.items():
        _need(type(path) is str and path and native._hex(sha),'exact known source path and SHA required')
        data=loader(path);_need(type(data) is bytes and len(data)<=16*1024**2 and native.digest(data)==sha,'admitted source file identity differs')
        total+=len(data);_need(total<=64*1024**2,'bounded source catalog closure required');data.decode('utf8');sources[path]=data
    entries=[]
    for value in functions:
        _need(type(value) is dict and set(value)=={'operation','originalSha256','kind','stage','locations'},'closed parsed function declaration required')
        operation=value['operation'];sha=value['originalSha256'];kind=value['kind'];stage=value['stage']
        _need(type(operation) is str and native._hex(sha),'literal function operation and full-source SHA required')
        if kind=='callback-own-apply':_need(stage=='callback' and operation=='Callback:'+sha,'exact admitted callback source operation required')
        else:_need(kind=='method' and stage in ('bootstrap-method','instance-method')
            and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',operation),'exact admitted method role required')
        locations=value['locations'];_need(type(locations) in (list,tuple) and locations,'full parsed function origins required');origins=[]
        for location in locations:
            _need(type(location) is dict and set(location)=={'source','start','end','declaration','source_form'},'closed parsed source span required')
            p=location['source'];start=location['start'];end=location['end']
            _need(p in sources and type(start) is int and type(end) is int and 0<=start<end<=len(sources[p])
                and type(location['declaration']) is str and location['declaration']
                and location['source_form'] in ('function-expression','function-declaration','method-definition','arrow-function'),
                'unsupported or unbound parsed full function span')
            body=sources[p][start:end];body.decode('utf8');_need(native.digest(body)==sha,'full function span/hash differs')
            origins.append(FunctionLocation(**location))
        _need(len(set(origins))==len(origins),'duplicate catalog origin is not independent ambiguity')
        entries.append(FunctionEntry(operation,sha,kind,stage,tuple(origins)))
    return SourceCatalog(parser_sha256,tuple(source_pins.items()),tuple(entries))


@dataclass(frozen=True)
class HookPolicy:
    catalog: SourceCatalog
    bootstrap_operations: tuple
    instance_operations: tuple
    instance_discovery_operation: str = 'Battle.setPlayer'
    sha256: str = field(init=False)

    def __post_init__(self):
        _need(type(self.catalog) is SourceCatalog and type(self.bootstrap_operations) is tuple and self.bootstrap_operations
            and type(self.instance_operations) is tuple,'independent immutable installation order required')
        methods=self.bootstrap_operations+self.instance_operations
        _need(len(set(methods))==len(methods),'duplicate expected method installation')
        _need(set(methods)=={e.operation for e in self.catalog.entries if e.kind=='method'},'complete independent method roster required')
        for stage,operations in (('bootstrap-method',self.bootstrap_operations),('instance-method',self.instance_operations)):
            _need(all(self.catalog.entry(op).stage==stage for op in operations),'catalog and independently declared installation stage differ')
        _need(not self.instance_operations or self.instance_discovery_operation in self.bootstrap_operations,'declared instance discovery seam required')
        object.__setattr__(self,'sha256',native.digest(_wire(dict(schema='pokezero.native_hook_policy.v1',catalog_sha256=self.catalog.sha256,
            bootstrap_operations=self.bootstrap_operations,instance_operations=self.instance_operations,
            instance_discovery_operation=self.instance_discovery_operation,discovery_rules='native-passive-registration-v1'),sort=True)))

    @property
    def operations(self):return frozenset(e.operation for e in self.catalog.entries)


@dataclass(frozen=True)
class HookAdmissionBinding:
    policy_sha256: str
    catalog_sha256: str
    previous_registration_head: str
    previous_roster_digest: str

    def __post_init__(self):
        _need(all(native._hex(x) for x in vars(self).values()),'exact independently sealed hook admission binding required')


class HookCursor:
    """Candidate-only registration fold; caller commits with causal/IO cursors."""
    def __init__(self):
        self.hooks=[];self.head=native.ZERO;self.bootstrap=0;self.instances=0;self.commands=0
        self.method_targets={};self.target_roles={};self.callbacks={};self.first_instance_call=None;self.method_operations=set()

    @property
    def digest(self):
        rows=[dict(operation=e['operation'],originalSha256=e['originalSha256'],kind=e['kind']) for e in self.hooks]
        rows.sort(key=lambda e:e['operation']);return native.digest(_wire(rows))

    def binding(self,policy):return HookAdmissionBinding(policy.sha256,policy.catalog.sha256,self.head,self.digest)

    def admit_binding(self,binding,policy):
        _need(type(binding) is HookAdmissionBinding and native._same(vars(binding),vars(self.binding(policy))),
            'pre-submission hook policy/catalog/registration parents differ')

    def candidate_command(self,rows,*,policy,phase):
        _need(type(policy) is HookPolicy and type(rows) is list and rows,'bounded source-bound causal command rows required')
        candidate=copy.deepcopy(self);candidate._consume(rows,policy,phase);return candidate

    def _consume(self,rows,policy,phase):
        _need(self.commands or phase=='start','registration composition requires explicit start')
        enters={};returns={}
        for _,row in rows:
            frame=row.get('frame');operation=row.get('operation')
            if operation in ('CALL_ENTER','CALL_RETURN','CALL_THROW'):
                _need(type(frame) is dict and type(frame.get('callId')) is int,'literal causal discovery frame required')
                target=enters if operation=='CALL_ENTER' else returns
                _need(frame['callId'] not in target,'duplicate causal discovery frame');target[frame['callId']]=row
        open_frames=[];seen_enter=False
        for wire,row in rows:
            operation=row['operation'];frame=row.get('frame')
            if operation=='CALL_ENTER':
                seen_enter=True;_need(self.bootstrap==len(policy.bootstrap_operations),'missing bootstrap registration before original call')
                open_frames.append(frame)
                if frame['operation']==policy.instance_discovery_operation and self.first_instance_call is None:self.first_instance_call=frame['callId']
                if frame['operation'].startswith('Callback:'):
                    token=row['evidence'].get('function_token');_need(token in self.callbacks and self.callbacks[token]==frame['operation'],
                        'callback invocation lacks earlier exact function-object registration')
                else:_need(frame['operation'] in self.method_operations,'original method invoked before admitted registration')
            elif operation in ('CALL_RETURN','CALL_THROW'):
                _need(open_frames and native._same(open_frames[-1],frame),'registration discovery call stack differs')
                if frame['operation'].startswith('Callback:'):
                    _need(row['evidence'].get('function_token')==enters[frame['callId']]['evidence'].get('function_token'),
                        'actual callback receiver changed between enter and return')
                open_frames.pop()
            elif operation=='HOOK_REGISTERED':
                e=row['evidence'];_need(type(wire) is bytes and wire.endswith(b'\n') and type(e) is dict
                    and set(e)=={'hook_id','operation','originalSha256','kind','target_token','discovery_call_id','discovery_site'},'closed literal hook registration evidence required')
                _need(type(e['hook_id']) is int and e['hook_id']==len(self.hooks),'hook registration gap or duplicate')
                token=e['target_token'];_need(type(token) is str and re.fullmatch(r'h(?:0|[1-9][0-9]*)',token),'stable registration object token required')
                entry=policy.catalog.entry(e['operation']);_need(entry.originalSha256==e['originalSha256'] and entry.kind==e['kind'],'installed original function differs from independent catalog')
                discovery=e['discovery_site'];_need(type(discovery) is dict,'closed literal hook discovery site required')
                if entry.stage=='bootstrap-method':
                    _need(self.commands==0 and phase=='start' and not seen_enter and frame is None and e['discovery_call_id'] is None
                        and self.bootstrap<len(policy.bootstrap_operations) and policy.bootstrap_operations[self.bootstrap]==entry.operation,
                        'bootstrap installation order/context differs');self.bootstrap+=1
                    wanted=dict(kind='bootstrap-method',owner=entry.operation.split('.')[0]);_need(native._same(discovery,wanted),'bootstrap declaration differs')
                else:
                    _need(open_frames and native._same(frame,open_frames[-1]) and type(e['discovery_call_id']) is int
                        and e['discovery_call_id']==frame['callId'],'registration is outside the actual discovery call')
                    if entry.stage=='instance-method':
                        _need(self.commands==0 and phase=='start' and frame['operation']==policy.instance_discovery_operation
                            and frame['callId']==self.first_instance_call and self.instances<len(policy.instance_operations)
                            and policy.instance_operations[self.instances]==entry.operation,'instance installation order/context differs');self.instances+=1
                        _need(native._same(discovery,dict(kind='instance-method',owner=entry.operation.split('.')[0])),'instance declaration differs')
                    else:self._callback_discovery(e,frame,discovery,enters,returns)
                if entry.kind=='method':
                    role=(entry.stage,entry.operation.split('.')[0]);existing=self.method_targets.get(role)
                    _need(token not in self.callbacks and (existing is None or existing==token)
                        and (token not in self.target_roles or self.target_roles[token]==role),'method target object identity differs')
                    self.method_targets[role]=token;self.target_roles[token]=role;self.method_operations.add(entry.operation)
                else:
                    _need(token not in self.target_roles and token not in self.callbacks,'callback target object token reused')
                    self.callbacks[token]=entry.operation
                self.hooks.append(dict(e));self.head=native.digest(bytes.fromhex(self.head)+struct.pack('<Q',len(wire))+wire)
        _need(not open_frames and self.bootstrap==len(policy.bootstrap_operations) and self.instances==len(policy.instance_operations),
            'incomplete source-bound installation or discovery call closure')
        self.commands+=1

    def _callback_discovery(self,e,frame,site,enters,returns):
        call=frame['callId'];_need(call in enters and call in returns and returns[call]['operation']=='CALL_RETURN','successful actual callback discovery return required')
        operation=frame['operation'];before=enters[call]['evidence'];after=returns[call]['evidence'];sha=e['originalSha256'];token=e['target_token']
        if site==dict(kind='getCallback-result'):
            _need(operation=='Battle.getCallback' and _function(after.get('result'),sha,token),'callback registration differs from actual getCallback result')
        elif set(site)=={'kind','index'} and site['kind']=='handler-result':
            _need(operation in ('Battle.findEventHandlers','Battle.findPokemonEventHandlers','Battle.findBattleEventHandlers',
                'Battle.findSideEventHandlers','Battle.findFieldEventHandlers') and type(site['index']) is int and site['index']>=0,'actual handler discovery role/index required')
            result=after.get('result');_need(type(result) is dict and type(result.get('items')) is list and site['index']<len(result['items']),'literal handler return array required')
            handler=result['items'][site['index']];fields=handler.get('fields') if type(handler) is dict else None
            _need(type(fields) is list,'literal handler callback property required')
            matches=[p[1] for p in fields if type(p) is list and len(p)==2 and p[0]=='callback']
            _need(len(matches)==1 and _function(matches[0],sha,token),'callback registration differs from exact handler result position')
        elif site==dict(kind='singleEvent-selection'):
            selection=before.get('callbackSelection');_need(operation=='Battle.singleEvent' and type(selection) is dict
                and set(selection)=={'function_token','originalSha256','selector'} and selection['function_token']==token and selection['originalSha256']==sha,
                'callback registration differs from original singleEvent selection')
            args=before.get('arguments');_need(type(args) is dict and type(args.get('items')) is list,'actual singleEvent arguments required');args=args['items']
            custom=args[7] if len(args)>7 else {'$native':'undefined'}
            if selection['selector']=='custom':_need(_function(custom,sha,token),'custom callback differs from actual original argument')
            else:
                false=custom is None or custom is False or type(custom) is int and custom==0 or custom==''
                false=false or native._same(custom,{'$native':'undefined'}) or native._same(custom,{'$nativeNumber':'NaN'}) or native._same(custom,{'$number':'-0'})
                _need(selection['selector']=='effect-property' and false and len(args)>1 and type(args[0]) is str,
                    'effect callback selection has a different actual custom operand')
        else:raise ValueError('unadmitted passive callback discovery site')
