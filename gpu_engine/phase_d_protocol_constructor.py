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
from pokezero.v5_teacher_public_root import capture_root
from . import phase_d_live_worlds as worlds,phase_d_protocol_world as protocol


class ProtocolConstructor:
    def __init__(self,*,catalog,source,metadata,owning_config,count,seed=20261103,
                 rung='R1',saved_root_admission=True,functions=None,seen=None,sessions=None):
        if count not in (2,8,16,32)or type(seed)is not int or type(saved_root_admission)is not bool:
            raise ValueError('original fixed world budget, seed and saved-root mode required')
        self.catalog,self.source,self.metadata=catalog,source,metadata
        self.owning_config,self.count,self.seed,self.rung=owning_config,count,seed,rung
        self.saved_root_admission=saved_root_admission
        self.functions=CompiledEncoderFunctions(catalog)if functions is None else functions
        if self.functions.catalog is not catalog:raise ValueError('same original compiled B2 catalog required')
        self.seen=set()if seen is None else seen
        self.sessions={}if sessions is None else sessions

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
        # Original Python public-root capture checks literal owning request and
        # exact-foe/split contamination before either constructor can consume it.
        public=capture_root(config=self.owning_config,contract=ActorContract.champion(),lines=lines,
            address=DecisionAddress(request['stream_id'],request['decision_index'],owner),battle_id=request['battle_id'],seed=self.seed&0xffffffff)
        packet=_packet(request['own_request_line'],request['owner'])
        adapter.feed(tuple(x for x in lines[len(old):]if not x.startswith('|request|')))
        if not self.saved_root_admission:self.sessions[key]=(adapter,lines)
        detached=worlds.from_b2(adapter,packet,self.source,self.metadata,address=public.address)
        made=protocol.sample(public,detached,lines=lines,catalog=self.catalog,source=self.source,metadata=self.metadata,
            config=self.owning_config,battle_id=request['battle_id'],seed=self.seed,count=self.count)
        return detached,made,tuple(map(bool,adapter.encode(packet,owner_side=owner)['legal_action_mask']))


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
                scope='HOST_CONSTRUCTOR_ONLY_NO_MODEL_FORWARD_SEARCH_TIME_OR_GPU_CLAIM')
        except (ValueError,KeyError,TypeError,IndexError,AttributeError)as error:
            return dict(schema=SCHEMA,request_id=request.get('request_id'),status='REFUSED',
                action_index=None,q_values=None,refusal=str(error),requested_worlds=self.count,
                constructed_worlds=0 if made is None else sum(w.state is not None for w in made.worlds),decision_seconds=None,
                world_refusals=[]if made is None else[dict(world=w.index,kinds=list(w.refusals))for w in made.worlds if w.refusals],
                scope='HOST_CONSTRUCTOR_ONLY_NO_MODEL_FORWARD_SEARCH_TIME_OR_GPU_CLAIM')


def read_bound(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('original bound startup input differs')
    return json.loads(raw)


def load_inputs(args):
    """Same pinned static catalog/source/config for host coverage and GPU search."""
    import subprocess
    from pokezero.live_protocol_actor import ActorContract,_vocab_sha
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.local_showdown import LocalShowdownConfig
    from pokezero.observation import ObservationFeatureMasks
    from pokezero import v5_contract
    from gpu_engine import public_belief,v4_encoder,device_public_worlds
    contract=ActorContract.champion();revision=subprocess.check_output(['git','-C',args.showdown_root,'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',args.showdown_root,'status','--porcelain','--untracked-files=no'],text=True)
    if revision!=contract.showdown_revision or dirty:raise ValueError('original clean pinned Showdown required')
    vocab=CategoryVocabulary(**read_bound(args.vocabulary,args.expected_vocabulary_sha256))
    if _vocab_sha(vocab)!=contract.category_vocab_sha256:raise ValueError('original9375 vocabulary required')
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
