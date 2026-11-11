"""The reviewed genuine protocol constructor, independent of an evaluator.

This class owns only the existing B2 compiled functions, public sessions and
fixed sampling inputs. Search shares these exact holders. Constructing worlds
requires neither a champion archive nor a forward, search object or GPU.
"""
import hashlib
import json
from pathlib import Path
import time

from pokezero.live_protocol_actor import ActorContract,_packet
from pokezero.recorded_encoder_runner import JaxRecordedAdapter
from pokezero.live_protocol_runtime import CompiledEncoderFunctions
from pokezero.phase_d_search_server import SCHEMA
from pokezero.v5_contract import DecisionAddress
from collections import OrderedDict
from pokezero.v5_teacher_public_root import RootSession,capture_root
from . import phase_d_live_worlds as worlds,phase_d_protocol_world as protocol


HELD_ROOTS=64   # roots kept for classify, most recent first (a batch larger than this recomputes the oldest, same value)


class ProtocolConstructor:
    def __init__(self,*,catalog,source,metadata,owning_config,count,seed=20261103,
                 rung='R1',saved_root_admission=True,functions=None,seen=None,sessions=None,incremental_roots=False):
        if count not in (2,8,16,32)or type(seed)is not int or type(saved_root_admission)is not bool:
            raise ValueError('original fixed world budget, seed and saved-root mode required')
        if type(incremental_roots)is not bool or(incremental_roots and saved_root_admission):
            raise ValueError('incremental roots need per-battle sessions (saved-root admission off)')
        self.catalog,self.source,self.metadata=catalog,source,metadata
        self.owning_config,self.count,self.seed,self.rung=owning_config,count,seed,rung
        self.saved_root_admission=saved_root_admission
        self.functions=CompiledEncoderFunctions(catalog)if functions is None else functions
        if self.functions.catalog is not catalog:raise ValueError('same original compiled B2 catalog required')
        self.seen=set()if seen is None else seen
        self.sessions={}if sessions is None else sessions
        # Incremental public roots (off by default): one RootSession per session key, and the roots prepare captured, by address,
        # for classify (public_root). Outputs are capture_root's; only the cost changes.
        self.incremental_roots=incremental_roots;self.root_sessions={};self.roots=OrderedDict()

    def prepare(self,request,index):
        required={'schema','request_id','battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line'}
        if set(request)!=required or request['schema']!=SCHEMA or request['owner']not in ('p1','p2'):
            raise worlds.Refusal('MALFORMED_SEARCH_REQUEST')
        if (type(request['stream_id'])is not int or not 0<request['stream_id']<2**31
                or type(request['decision_index'])is not int or request['decision_index']<0
                or type(request['battle_id'])is not str or not request['battle_id'].strip()
                or type(request['generation'])is not int or request['generation']<=0):
            raise worlds.Refusal('MALFORMED_SEARCH_ADDRESS')
        lines=tuple(request['lines']);prefix=hashlib.sha256(''.join(x+'\n'for x in lines).encode()).hexdigest()
        if prefix!=request['prefix_sha256']or not lines or lines[-1]!=request['own_request_line']:
            raise worlds.Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH')
        key=(self.rung,request['battle_id'],request['owner'],request['stream_id']);address=(key,request['decision_index'],prefix)
        if address in self.seen:raise worlds.Refusal('SEARCH_REQUEST_ALREADY_ATTEMPTED')
        self.seen.add(address)
        owner=int(request['owner'][1])-1
        prior=None if self.saved_root_admission else self.sessions.get(key)
        if prior is None:
            adapter=JaxRecordedAdapter(self.catalog,owner_side=owner,compiled_functions=self.functions)
            adapter.reset(stream_id=request['stream_id'],battle_id=request['battle_id']);old=()
        else:adapter,old=prior
        if lines[:len(old)]!=old or len(lines)<=len(old):raise worlds.Refusal('SEARCH_PREFIX_NOT_ORIGINAL_SUCCESSOR')
        with adapter.checked_catalog():   # the compiled catalog's structure walk once per request, not on every feed/encode
            # Original Python public-root capture checks literal owning request and
            # exact-foe/split contamination before either constructor can consume it.
            public=self._capture(key,address,request,lines,owner)
            packet=_packet(request['own_request_line'],request['owner'])
            adapter.feed(tuple(x for x in lines[len(old):]if not x.startswith('|request|')))
            if not self.saved_root_admission:self.sessions[key]=(adapter,lines)
            facts=protocol.tail.facts(public,packet,lines)
            detached=worlds.from_b2(adapter,packet,self.source,self.metadata,address=public.address,defer_sleep=True,protocol_facts=facts)
            made=protocol.sample(public,detached,lines=lines,catalog=self.catalog,source=self.source,metadata=self.metadata,
                config=self.owning_config,battle_id=request['battle_id'],seed=self.seed,count=self.count)
            return made.root,made,tuple(map(bool,adapter.encode(packet,owner_side=owner)['legal_action_mask']))


    def _capture(self,key,address,request,lines,owner):
        """capture_root's PublicRoot for this request: from scratch, or with incremental_roots through the key's RootSession (a
        new one, folding from the start, when there is none or the request does not extend it; any error drops it)."""
        kwargs=dict(contract=ActorContract.champion(),lines=lines,address=DecisionAddress(request['stream_id'],request['decision_index'],owner),
            battle_id=request['battle_id'],seed=self.seed&0xffffffff)
        if not self.incremental_roots:return capture_root(config=self.owning_config,**kwargs)
        session=self.root_sessions.get(key)
        if session is None or not session.extends(lines):
            if session is not None:session.close()
            session=self.root_sessions[key]=RootSession(config=self.owning_config,contract=kwargs['contract'],battle_id=request['battle_id'],
                owner=request['owner'],stream_id=request['stream_id'])
        try:public=session.root(**kwargs)
        except BaseException:
            self.root_sessions.pop(key,None);raise
        self.roots[address]=public;self.roots.move_to_end(address)
        while len(self.roots)>HELD_ROOTS:self.roots.popitem(last=False)
        return public

    def public_root(self,request):
        """The PublicRoot of a protocol request (classify's): the one prepare captured for this very address when it is held
        (incremental_roots), else capture_root's, recomputed. The same value either way."""
        lines=tuple(request['lines']);prefix=hashlib.sha256(''.join(x+'\n'for x in lines).encode()).hexdigest()
        if type(request['stream_id'])is int and type(request['decision_index'])is int:   # True == 1: a bool never matches a held root
            held=self.roots.get(((self.rung,request['battle_id'],request['owner'],request['stream_id']),request['decision_index'],prefix))
            if held is not None:return held
        return capture_root(config=self.owning_config,contract=ActorContract.champion(),lines=lines,battle_id=request['battle_id'],
            address=DecisionAddress(request['stream_id'],request['decision_index'],int(request['owner'][1])-1),seed=self.seed&0xffffffff)

    def drop_roots(self,battles=None):
        """Close and forget the incremental root sessions and held roots of these battle ids (every one when None)."""
        for key in[k for k in self.root_sessions if battles is None or k[1]in battles]:self.root_sessions.pop(key).close()
        for address in[a for a in self.roots if battles is None or a[0][1]in battles]:del self.roots[address]

    def construct(self,request,index=0):
        """Same full-world test as Search's constructor-only path; no evaluator."""
        started=time.monotonic();made=None
        try:
            detached,made,legal=self.prepare(request,index)
            if not made.fully_constructed:
                raise worlds.Refusal('WORLD_CONSTRUCTOR_REFUSED',';'.join(sorted({r for w in made.worlds for r in w.refusals})))
            return dict(schema=SCHEMA,request_id=request.get('request_id'),status='COMPLETE',
                action_index=None,q_values=None,refusal=None,requested_worlds=self.count,
                constructed_worlds=self.count,decision_seconds=None,constructor_seconds=time.monotonic()-started,
                mapper_diagnostics=list(getattr(made,'mapper_diagnostics',())),
                scope='HOST_CONSTRUCTOR_ONLY_NO_MODEL_FORWARD_SEARCH_TIME_OR_GPU_CLAIM')
        except (ValueError,KeyError,TypeError,IndexError,AttributeError)as error:
            return dict(schema=SCHEMA,request_id=request.get('request_id'),status='REFUSED',
                action_index=None,q_values=None,refusal=str(error),requested_worlds=self.count,
                constructed_worlds=0 if made is None else sum(w.state is not None for w in made.worlds),decision_seconds=None,
                mapper_diagnostics=[] if made is None else list(getattr(made,'mapper_diagnostics',())),
                world_refusals=[]if made is None else[dict(world=w.index,kinds=list(w.refusals))for w in made.worlds if w.refusals],
                scope='HOST_CONSTRUCTOR_ONLY_NO_MODEL_FORWARD_SEARCH_TIME_OR_GPU_CLAIM')


def read_bound(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('original bound startup input differs')
    return json.loads(raw)


def vocabulary_from_document(document,contract):
    """Resolve the original worker's provenance-wrapped vocabulary format."""
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.live_protocol_actor import _vocab_sha
    if (document['checkpoint_sha256']!=contract.checkpoint_sha256
            or document['category_vocab_sha256']!=contract.category_vocab_sha256
            or document['source_hash']!=contract.belief_set_source_hash):
        raise ValueError('original champion category/source provenance differs')
    # Provenance fields are not CategoryVocabulary constructor arguments.
    # Keep the ordered tokens, aliases and OOV band exactly as published.
    vocab=CategoryVocabulary(tuple(document['tokens']),document['oov_buckets'],document['aliases'])
    if (_vocab_sha(vocab)!=contract.category_vocab_sha256 or vocab.size!=contract.categorical_vocab_size
            or type(document['size'])is not int or document['size']!=vocab.size):
        raise ValueError('literal original category vocabulary differs')
    return vocab


def load_inputs(args):
    """Same pinned static catalog/source/config for host coverage and GPU search."""
    import subprocess
    from pokezero.live_protocol_actor import ActorContract
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.local_showdown import LocalShowdownConfig
    from pokezero.observation import ObservationFeatureMasks
    from pokezero import v5_contract
    from gpu_engine import public_belief,v4_encoder,device_public_worlds
    contract=ActorContract.champion();revision=subprocess.check_output(['git','-C',args.showdown_root,'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',args.showdown_root,'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=contract.showdown_revision or dirty:raise ValueError('original clean pinned Showdown required')
    vocab=vocabulary_from_document(read_bound(args.vocabulary,args.expected_vocabulary_sha256),contract)
    metadata=read_bound(args.metadata,args.expected_metadata_sha256)
    source=load_gen3_randbat_source_cached(args.showdown_root)
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),load_showdown_dex_cached(args.showdown_root),
        vocab,checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    bound=device_public_worlds.compile_source(source,metadata,expected_source_hash=contract.belief_set_source_hash,
        expected_source_rows_sha256=v5_contract.source_variant_census(source)['source_rows_sha256'])
    masks=dict(contract.feature_masks);masks['opponent_tendency_stats_block']=masks.pop('stats_block')
    config=LocalShowdownConfig(showdown_root=args.showdown_root,node_binary=args.node_binary,observation_spec=v5_contract.V4_SPEC,
        category_vocab=vocab,feature_masks=ObservationFeatureMasks(**masks),set_belief_source=True)
    return catalog,bound,metadata,config
