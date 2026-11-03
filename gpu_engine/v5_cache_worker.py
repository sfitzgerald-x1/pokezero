"""CPU-only cache worker bootstrap. No model, engine or journal writer.

Executed as a file so CPU affinity and library limits precede project imports.
Requests contain only immutable spool references and one declared game row.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


class Assembler:
    def __init__(self):self.contexts=OrderedDict()

    def assemble(self,request):
        started=time.perf_counter()
        from .v5_device_spool import Spool,_stat
        from .v5_device_rollout import ActorBank,Bindings
        from pokezero.v5_contract import LabelVocabulary
        from pokezero.v5_loss import V5LossConfig
        from pokezero.v5_gumbel import Recipe
        key=(request['context'],request['sha256'])
        if key not in self.contexts:
            path=Path(key[0]);data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=key[1]:raise ValueError('cache worker context changed')
            context=json.loads(data);bank=dict(context['bank']);binding=dict(bank['bindings'])
            binding['label_vocabulary']=LabelVocabulary(**binding['label_vocabulary'])
            bank['bindings']=Bindings(**binding)
            spool=Spool.__new__(Spool);spool.bank=ActorBank(**bank)
            spool.store=SimpleNamespace(path=Path(context['store']))
            spool.root=path.parent;spool.bank_sha256=context['bank_sha256']
            spool.teacher_recipe=None if context['teacher_seed']is None else Recipe(context['teacher_seed'])
            spool.records=[];spool.stats={};spool.parts=[]
            context.update(spool=spool,loss=V5LossConfig(**context['loss']),loaded=[],references=[])
            self.contexts[key]=context
            if len(self.contexts)>2:self.contexts.popitem(last=False)
        context=self.contexts[key];self.contexts.move_to_end(key)
        spool=context['spool'];prefix=request['prefix']
        if type(prefix)is not int or prefix<=0:raise ValueError('literal immutable chunk prefix required')
        loaded=context['loaded'];references=request['records']
        if len(references)!=prefix or context['references'][:prefix]!=references[:len(loaded)]:
            raise ValueError('immutable cache chunk prefix changed')
        while len(loaded)<prefix:
            ordinal=len(loaded);path=spool.root/str(ordinal)/'metadata.json'
            data=path.read_bytes();reference=references[ordinal]
            if (spool.store.path/reference['path']!=path
                    or hashlib.sha256(data).hexdigest()!=reference['sha256']):
                raise ValueError('cache chunk metadata changed')
            record=json.loads(data)
            if record['chunk']!=ordinal or record['bank_sha256']!=spool.bank_sha256:
                raise ValueError('cache worker chunk binding differs')
            for desc in (*record['frames'].values(),*record['final'].values(),record.get('teacher')):
                if desc is not None:
                    file=spool.store.path/desc['path'];identity=tuple(desc['identity'])
                    if _stat(file)!=identity:raise ValueError('owned spool file changed before worker read')
                    spool.stats[str(file)]=identity
            loaded.append(record)
            context['references'].append(reference)
        # Early endings use exactly the prefix present at their disposition,
        # even when another worker has already read a later chunk.
        spool.records=loaded[:prefix];spool.parts=[]
        row=request['row'];game=SimpleNamespace(**request['game'])
        if type(row)is not int or not 0<=row<len(spool.bank.streams)or game.stream_id!=spool.bank.streams[row]:
            raise ValueError('original declared cache row required')
        import numpy as np
        spool.has_training=np.zeros(len(spool.bank.streams),bool);spool.has_training[row]=request['has_training']
        spool.finalize_game(row,game,bindings=spool.bank.bindings,loss_config=context['loss'],discount=context['discount'])
        return dict(part=None if not spool.parts else [row,str(spool.parts[0][1]),spool.parts[0][2]],
            seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--core',type=int,required=True);args=parser.parse_args()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{args.core})
    # Parent also sets these before interpreter startup. Do not mutate the GPU
    # owner's environment or fork its live CUDA runtime.
    if os.environ.get('JAX_PLATFORMS')!='cpu'or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('cache workers must be CPU-only')
    import importlib.util
    if importlib.util.find_spec('torch')is not None:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assembler=Assembler()
    for line in sys.stdin:
        try:response=dict(ok=True,result=assembler.assemble(json.loads(line)))
        except Exception as error:response=dict(ok=False,error=type(error).__name__+': '+str(error))
        print(json.dumps(response,allow_nan=False),flush=True)
        if not response['ok']:return 1
    return 0


if __name__=='__main__':
    # File bootstrap has no package context until after affinity is installed.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    __package__='gpu_engine'
    raise SystemExit(main())
