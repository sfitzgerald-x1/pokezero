"""Disk-backed raw chunks and original-schema v5 cache assembly.

Raw fields keep EVERY row, including inactive/refused/historical rows. Only
one game's dense history is materialized for original terminal/GAE labels.
Final arrays are disk memmaps in original (frame, declared row, owner) order.
The trainer and its ordinary cache reader are not replaced.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import uuid
import tempfile
import numpy as np
from pokezero import v5_data
from . import v5_device_rollout as rollout


def _stat(path):
    s=path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def retain_raw_bank(ordinal,every):
    if type(ordinal)is not int or ordinal<0 or type(every)is not int or every<=0:
        raise ValueError('declared global bank ordinal and positive raw interval required')
    return (ordinal+1)%every==0


class Spool:
    def __init__(self,store,bank,*,bank_sha256,teacher_recipe=None,transient_raw_root=None):
        self.store=store;self.bank=bank;self.bank_sha256=bank_sha256;self.teacher_recipe=teacher_recipe
        self.root=store.path/'rollouts'/('spool-'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self._raw_directory=(None if transient_raw_root is None else
            tempfile.TemporaryDirectory(prefix='v5-raw-',dir=transient_raw_root))
        self.raw_bytes=0
        self.records=[];self.record_refs=[];self.stats={};self.parts=[];self.has_training=np.zeros(len(bank.streams),bool)

    def _save(self,path,value):
        if self._raw_directory is not None:
            path=Path(self._raw_directory.name)/path.relative_to(self.root)
            path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as stream:np.save(stream,value,allow_pickle=False)
        path.chmod(0o444);self.stats[str(path)]=_stat(path)
        self.raw_bytes+=path.stat().st_size
        return dict(path=str(path if self._raw_directory is not None else path.relative_to(self.store.path)),shape=list(value.shape),dtype=value.dtype.str,
            sha256=v5_data._file_sha(path),identity=self.stats[str(path)])

    def release_raw(self):
        """Only this object's temporary files; caller first joins all readers."""
        directory=getattr(self,'_raw_directory',None)
        if directory is not None:directory.cleanup()

    def append(self,host,*,teacher_records=None):
        """One host chunk -> immutable batch-major .npy files, no full copy."""
        ordinal=len(self.records);folder=self.root/str(ordinal);folder.mkdir()
        record=dict(schema='pokezero.device_rollout_spool.v1',bank_sha256=self.bank_sha256,chunk=ordinal,
            steps=host.frames.streams.shape[0],frames={},final={},storage='batch,chunk_step,...; original logical order is chunk_step,batch,...')
        if self._raw_directory is not None:record['raw_retention']='TRANSIENT_ARRAYS_NOT_RESUMABLE; game caches and teacher labels durable'
        for name,value in zip(host.frames._fields,host.frames):
            record['frames'][name]=None if value is None else self._save(folder/(name+'.npy'),np.swapaxes(value,0,1))
        for name in ('streams','frames','status','winner','terminal_frame','terminal_hp','terminal_max_hp'):
            record['final'][name]=self._save(folder/('final_'+name+'.npy'),getattr(host.carry,name))
        if self.teacher_recipe is not None:
            from .v5_device_teacher import validate_records
            if teacher_records is None:raise ValueError('D spool requires every requested teacher slot')
            validate_records(host.frames,self.bank,self.teacher_recipe,teacher_records)
            data=(json.dumps(teacher_records,sort_keys=True,allow_nan=False)+'\n').encode()
            description=self.store.artifact(str((folder/'teacher.json').relative_to(self.store.path)),data,kind='DEVICE_TEACHER_CHUNK',bank_sha256=self.bank_sha256)
            file=self.store.path/description['path'];file.chmod(0o444);self.stats[str(file)]=_stat(file)
            record['teacher']=dict(path=description['path'],sha256=description['sha256'],identity=self.stats[str(file)])
        elif teacher_records is not None:raise ValueError('teacher records require original Recipe')
        self.has_training|=np.any(host.frames.trainable,axis=(0,2))
        reference=self.store.artifact(str((folder/'metadata.json').relative_to(self.store.path)),
            (json.dumps(record,sort_keys=True)+'\n').encode(),kind='DEVICE_ROLLOUT_CHUNK',bank_sha256=self.bank_sha256)
        (self.store.path/reference['path']).chmod(0o444)
        self.record_refs.append(dict(path=reference['path'],sha256=reference['sha256']))
        self.records.append(record)

    def _read_row(self,description,row):
        path=self.store.path/description['path'];before=_stat(path)
        if before!=self.stats[str(path)]:raise ValueError('owned spool file changed')
        mapped=np.load(path,mmap_mode='r',allow_pickle=False)
        try:
            if list(mapped.shape)!=description['shape']or mapped.dtype.str!=description['dtype']:
                raise ValueError('owned spool geometry changed')
            value=np.array(mapped[row:row+1],copy=True)
        finally:mapped._mmap.close()
        if _stat(path)!=before:raise ValueError('owned spool file changed during read')
        return value

    def game(self,row):
        """Materialize only [all recorded steps,1,2,...], never [T,B,...]."""
        if not self.records or type(row)is not int or not 0<=row<len(self.bank.streams):raise ValueError('spooled declared game required')
        total=sum(r['steps']for r in self.records);values={}
        for name in rollout.Frames._fields:
            first=self.records[0]['frames'][name]
            if first is None:
                if any(r['frames'][name]is not None for r in self.records):raise ValueError('spool optional field changed')
                values[name]=None;continue
            value=np.empty((total,1)+tuple(first['shape'][2:]),np.dtype(first['dtype']));offset=0
            for record in self.records:
                piece=np.swapaxes(self._read_row(record['frames'][name],row),0,1)
                value[offset:offset+record['steps']]=piece;offset+=record['steps']
            values[name]=value
        final={name:self._read_row(desc,row)for name,desc in self.records[-1]['final'].items()}
        carry=rollout.Carry(**{name:final.get(name)for name in rollout.Carry._fields})
        return rollout.Rollout(carry,rollout.Frames(**values))

    def finalize_game(self,row,game,*,bindings,loss_config,discount):
        if not self.has_training[row]:return
        bank=replace(self.bank,streams=(self.bank.streams[row],),actor_indices=(self.bank.actor_indices[row],),learner_mask=(self.bank.learner_mask[row],))
        options={}
        if self.teacher_recipe is not None:
            labels=[]
            for record in self.records:
                description=record['teacher'];file=self.store.path/description['path'];before=_stat(file)
                if before!=self.stats[str(file)]or v5_data._file_sha(file)!=description['sha256']:
                    raise ValueError('owned teacher records changed')
                items=json.loads(file.read_text())
                if _stat(file)!=before:raise ValueError('owned teacher records changed during read')
                labels.extend(x for x in items if x['address']['stream']==game.stream_id)
            options=dict(teacher_records=labels,teacher_recipe=self.teacher_recipe)
        batch=rollout.training_batch((self.game(row),),bindings,actor_bank=bank,battle_ids=(game.battle_id,),seeds=(game.seed,),
            loss_config=loss_config,discount=discount,**options)
        folder=self.root/'games';folder.mkdir(exist_ok=True);path=folder/str(row)
        v5_data.write_cache(path,batch)
        for p in path.iterdir():p.chmod(0o444)
        self.parts.append((row,path,v5_data._file_sha(path/'metadata.json')))

    def merge(self,path):
        return merge_game_parts(path,self.parts,self.bank.streams)


def merge_game_parts(path,game_parts,streams):
    """Original cache format, bounded per-game scatter into disk arrays.

    JSON row metadata remains O(N), as required by the unchanged reader.
    Dense arrays are never concatenated or held in whole-cohort RAM.
    """
    if not game_parts:return None
    parts=[]
    for row,p,expected in game_parts:
        if v5_data._file_sha(p/'metadata.json')!=expected:raise ValueError('ordinary game cache metadata changed')
        meta=json.loads((p/'metadata.json').read_text());check=dict(meta);claimed=check.pop('metadata_sha256',None)
        if meta.get('schema_version')!=v5_data.CACHE_SCHEMA or claimed!=v5_data._sha(check):
            raise ValueError('ordinary game cache schema/metadata changed')
        parts.append((row,p,meta))
    first=parts[0][2];n=sum(m['example_count']for _,_,m in parts)
    coordinates=np.empty(n,dtype=[('frame','i4'),('row','i4'),('owner','i1')]);offset=0
    for row,_,meta in parts:
        count=meta['example_count'];addresses=meta['v5_rows']['addresses']
        if len(addresses)!=count or any(a['stream']!=streams[row]for a in addresses):raise ValueError('game cache source address differs')
        coordinates['frame'][offset:offset+count]=[a['frame']for a in addresses]
        coordinates['row'][offset:offset+count]=row;coordinates['owner'][offset:offset+count]=[a['owner']for a in addresses];offset+=count
        if (meta['v5_static']!=first['v5_static']or meta['base_none']!=first['base_none']
                or set(meta['arrays'])!=set(first['arrays'])or set(meta['base_json'])!=set(first['base_json'])):
            raise ValueError('ordinary game cache namespaces differ')
    order=np.lexsort((coordinates['owner'],coordinates['row'],coordinates['frame']))
    ordered=coordinates[order]
    if n>1 and np.any(ordered[1:]==ordered[:-1]):raise ValueError('duplicate original training address')
    destinations=np.empty(n,np.int64);destinations[order]=np.arange(n,dtype=np.int64)
    metadata=dict(schema_version=v5_data.CACHE_SCHEMA,example_count=n,v5_static=first['v5_static'],
        v5_rows={},base_json={},base_none=first['base_none'],arrays={})
    for section in ('v5_rows','base_json'):
        for name in first[section]:
            values=[None]*n;offset=0
            for _,_,meta in parts:
                count=meta['example_count'];items=meta[section][name]
                if len(items)!=count:raise ValueError('ordinary cache row metadata differs')
                for target,value in zip(destinations[offset:offset+count],items):values[int(target)]=value
                offset+=count
            metadata[section][name]=values
    path=Path(path);path.mkdir()
    for name in first['arrays']:
        descriptions=[m['arrays'][name]for _,_,m in parts];dtype=np.result_type(*(np.dtype(d['dtype'])for d in descriptions))
        shape=(n,)+tuple(descriptions[0]['shape'][1:])
        if any(tuple(d['shape'][1:])!=shape[1:]or np.dtype(d['dtype']).kind!=dtype.kind
                or (np.dtype(d['dtype'])!=dtype and dtype.kind not in 'US')for d in descriptions):
            raise ValueError('ordinary cache field geometry/dtype differs')
        target=path/(name+'.npy');output=np.lib.format.open_memmap(target,mode='w+',dtype=dtype,shape=shape);offset=0
        try:
            for (_,folder,meta),description in zip(parts,descriptions):
                file=folder/(name+'.npy')
                if v5_data._file_sha(file)!=description['sha256']:raise ValueError('ordinary game cache bytes changed')
                source=np.load(file,mmap_mode='r',allow_pickle=False)
                try:
                    count=meta['example_count'];output[destinations[offset:offset+count]]=source;offset+=count
                finally:source._mmap.close()
            output.flush()
        finally:output._mmap.close()
        metadata['arrays'][name]=dict(shape=list(shape),dtype=dtype.str,sha256=v5_data._file_sha(target))
    # Same canonical metadata SHA as original _sha, streamed to avoid a
    # second whole JSON string. Reader/trainer format remains unchanged.
    digest=hashlib.sha256()
    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(metadata):digest.update(piece.encode())
    metadata['metadata_sha256']=digest.hexdigest()
    with (path/'metadata.json').open('x')as stream:json.dump(metadata,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    return metadata
