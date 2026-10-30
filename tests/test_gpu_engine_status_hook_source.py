"""Verified static Dex callback inventory only; no Battle/Teams/game/model."""
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import weakref

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import engine_status_hook_source as h
from gpu_engine import engine_request_source as s, engine_status_residual_callbacks as r
from gpu_engine import engine_public_events as p, transition
import test_gpu_engine_request_source as fixtures


class StatusHookSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RequestSourceTests.setUpClass();cls.f=fixtures.RequestSourceTests
        cls.metadata=deepcopy(cls.f.metadata)
        cls.metadata['vocabulary']['abilities'] += ['shedskin','waterveil','pressure','synchronize']
        cls.metadata['vocabulary']['items'] += ['lumberry']
        cls.engine=p.compile_catalog(cls.metadata,transition.tables_from_metadata(cls.metadata),
            public_moves=cls.f.lifecycle_catalog.public_moves,dex=cls.f.dex,
            exposure=p.HealthExposure(s.SHOWDOWN,'gen3randombattle',True,False))
        cls.rules=s.load_source_rules(cls.f.root,cls.metadata,node=cls.f.node)
        cls.helper=r.compile_status_residual(cls.metadata,engine_catalog=cls.engine)
        cls.compiled=h.compile_hook_source(cls.rules,cls.helper,engine_catalog=cls.engine)
        cls.data=h.check_compiled(cls.compiled)
        cls.manifest=json.loads(h.manifest_json(cls.compiled))

    def test_all_rows_exact_source_identity_dispatch_and_digests(self):
        payload=self.manifest
        self.assertEqual(payload['revision'],s.SHOWDOWN)
        self.assertEqual(payload['dispatch'],h.DISPATCH)
        self.assertEqual(payload['dispatch']['Residual'],['onResidual','onSideResidual','onFieldResidual'])
        self.assertEqual(payload['dispatch']['Update'],['onUpdate'])
        count=0
        for kind in h.KINDS:
            values=payload['rows'][kind];ids=self.metadata['vocabulary'][kind]
            self.assertEqual([x['requested'] for x in values],ids)
            self.assertEqual(len(values),len(getattr(self.data,kind).classification))
            for row,digest in zip(values,getattr(self.data,kind).row_digest):
                np.testing.assert_array_equal(digest,h._words(h._digest(row)))
                for entry in row['entries']:
                    self.assertEqual(len(entry['sha256']),64);int(entry['sha256'],16)
                    if entry['metadata']:self.assertNotEqual(entry['type'],'function')
                count+=1
        self.assertEqual(count,sum(len(self.metadata['vocabulary'][kind]) for kind in h.KINDS))
        self.assertGreater(len(payload['conditions']),100)
        self.assertTrue(payload['rules']);self.assertTrue(payload['rule_table'])
        self.assertEqual(payload['format_row']['id'],'gen3randombattle')
        self.assertEqual(int(self.data.items.classification[0]),h.ABSENT)
        self.assertEqual(payload['rows']['items'][0]['effect_type'],'Absent')
        self.assertNotIn('onAnyResidual',payload['dispatch']['Residual'])
        self.assertEqual(h._digest(payload),hashlib.sha256(h.manifest_json(self.compiled).encode()).hexdigest())

    def test_original_shedskin_immunity_waterveil_lum_and_leftovers_are_not_admission(self):
        ids=self.metadata['vocabulary']
        for ability,flag,key in (('shedskin',h.RESIDUAL,'onResidual'),
                ('immunity',h.SET_STATUS|h.UPDATE,'onSetStatus'),
                ('waterveil',h.SET_STATUS|h.UPDATE,'onSetStatus')):
            i=ids['abilities'].index(ability);row=self.manifest['rows']['abilities'][i]
            self.assertEqual(int(self.data.abilities.relevant_flags[i])&flag,flag)
            self.assertTrue(any(x['key']==key and x['type']=='function' for x in row['entries']))
            self.assertEqual(int(self.data.abilities.classification[i]),h.RELEVANT_UNREVIEWED)
        i=ids['abilities'].index('pressure')
        self.assertEqual(int(self.data.abilities.classification[i]),h.NO_RELEVANT_HOOKS)
        self.assertTrue(self.manifest['rows']['abilities'][i]['entries']) # onStart etc retained
        i=ids['items'].index('lumberry')
        self.assertTrue(int(self.data.items.relevant_flags[i])&h.UPDATE)
        self.assertTrue(int(self.data.items.relevant_flags[i])&h.EAT)
        self.assertEqual(int(self.data.items.classification[i]),h.RELEVANT_UNREVIEWED)
        i=ids['items'].index('leftovers')
        self.assertEqual(int(self.data.items.classification[i]),h.LEFTOVERS_TRANSPORT_CANDIDATE)
        row=self.manifest['rows']['items'][i]
        self.assertEqual(next((x['Order'],x['SubOrder']) for x in row['entries'] if x['key']=='onResidual'),(10,4))
        self.assertFalse(hasattr(self.data,'admitted'));self.assertFalse(hasattr(h,'admit'))

    def test_entire_original_generator_vocabulary_static_dex_only_and_rule_carriers(self):
        # Same generic finite vocabulary expansion as generate_teams.cjs;
        # only Dex is loaded. No generator/Teams/Battle/RNG is invoked.
        script=r'''
const {Dex}=require(process.argv[1]+'/dist/sim/dex');const d=Dex.mod('gen3');
const ids=xs=>['_none',...[...new Set(xs.map(x=>x.id).filter(Boolean))].sort()];
const T=['Normal','Fighting','Flying','Poison','Ground','Rock','Bug','Ghost','Steel',
 'Fire','Water','Grass','Electric','Psychic','Ice','Dragon','Dark'];
const species=d.species.all().flatMap(s=>[s,...(s.cosmeticFormes||[]).map(n=>d.species.get(n))])
 .filter(s=>s.exists&&s.gen<=3);
const moves=new Set(d.moves.all().filter(x=>x.exists&&x.gen<=3).map(x=>x.id));
moves.add('struggle');for(const type of T.slice(1))moves.add('hiddenpower'+type.toLowerCase());
const vocabulary={species:ids(species),moves:['_none',...[...moves].sort()],
 items:ids(d.items.all().filter(x=>x.exists&&x.gen<=3)),
 abilities:ids(d.abilities.all().filter(x=>x.exists&&x.gen<=3)),
 types:['_none',...T.map(x=>x.toLowerCase())],natures:ids(d.natures.all())};
const rows=vocabulary.moves.map(id=>{if(id==='_none')return null;const m=d.moves.get(id);
 return {id,type:m.type==='???'?0:vocabulary.types.indexOf(m.type.toLowerCase()),source_type:m.type,
 category:m.category==='Status'?'Status':['Fire','Water','Grass','Electric','Psychic','Ice','Dragon','Dark'].includes(m.type)?'Special':'Physical',
 base_power:m.basePower,pp:m.pp,accuracy:m.accuracy,priority:m.priority,flags:m.flags};});
console.log(JSON.stringify({vocabulary,moves:rows}));'''
        static=json.loads(subprocess.check_output([self.f.node,'-e',script,str(self.f.root)],text=True))
        metadata=deepcopy(self.metadata);metadata.update(static)
        engine=p.compile_catalog(metadata,transition.tables_from_metadata(metadata),
            public_moves=self.f.lifecycle_catalog.public_moves,dex=self.f.dex,
            exposure=p.HealthExposure(s.SHOWDOWN,'gen3randombattle',True,False))
        rules=s.load_source_rules(self.f.root,metadata,node=self.f.node)
        compiled=h.compile_hook_source(rules,r.compile_status_residual(metadata,engine_catalog=engine),engine_catalog=engine)
        data=h.check_compiled(compiled);manifest=json.loads(h.manifest_json(compiled))
        for kind in h.KINDS:
            self.assertEqual(len(manifest['rows'][kind]),len(metadata['vocabulary'][kind]))
            known=getattr(data,kind).manifest_known.tolist()
            if kind!='moves':self.assertTrue(all(known),kind)
            for row,ok,flag,classification in zip(manifest['rows'][kind],known,
                    getattr(data,kind).relevant_flags.tolist(),getattr(data,kind).classification.tolist()):
                self.assertTrue(row['exists'])
                if not ok:self.assertEqual((flag,classification),(-1,h.UNSUPPORTED))
        aliases=[row for row in manifest['rows']['moves'] if row.get('real_move')]
        self.assertTrue(aliases)
        self.assertTrue(all(row['id']=='hiddenpower' for row in aliases))
        self.assertTrue(all(row['known'] for row in aliases))
        # String-valued lock callbacks are inventoried, but deliberately outside
        # this compiler's typed function/bool/number qualification vocabulary.
        self.assertEqual([row['requested'] for row,known in zip(manifest['rows']['moves'],
            data.moves.manifest_known.tolist()) if not known],['bide','iceball','rollout','uproar'])
        self.assertGreater(len(manifest['rows']['abilities']),70)
        self.assertGreater(len(manifest['rows']['items']),100)
        self.assertGreater(len(manifest['rows']['moves']),350)
        table=[key for key,_ in manifest['rule_table'] if not key.startswith(('+','*','-','!'))]
        self.assertEqual([row['requested'] for row in manifest['rules']],table)
        sleep=next(row for row in manifest['rules'] if row['id']=='sleepclausemod')
        self.assertTrue(sleep['installed_as_pseudo_weather'])
        self.assertTrue(sleep['flags']&h.SET_STATUS)
        self.assertTrue(any(row['key']=='onSetStatus' for row in sleep['entries']))
        self.assertGreater(len(h.manifest_json(compiled)),100000)
        type(self).full_summary=dict(rows={kind:len(manifest['rows'][kind]) for kind in h.KINDS},
            conditions=len(manifest['conditions']),rules=len(manifest['rules']),
            callback_entries=sum(len(row['entries']) for rows in manifest['rows'].values() for row in rows),
            unsupported_moves=[row['requested'] for row,known in zip(manifest['rows']['moves'],data.moves.manifest_known.tolist()) if not known],
            source_tree=manifest['source_tree'],compiled_tree=manifest['compiled_tree'],
            metadata_sha256=manifest['metadata_sha256'],vocabulary_sha256=manifest['vocabulary_sha256'],
            manifest_sha256=hashlib.sha256(h.manifest_json(compiled).encode()).hexdigest(),
            manifest_bytes=len(h.manifest_json(compiled).encode()),
            rule_field_carriers=[row['id'] for row in manifest['rules'] if row['installed_as_pseudo_weather']])

    def test_verified_loader_only_rejects_fixture_copy_and_mutated_origin(self):
        fixture=s._compile_rules(self.metadata,json.loads(s._rules(self.rules)[2]))
        for value in (fixture,replace(self.rules)):
            with self.assertRaises(ValueError):h.compile_hook_source(value,self.helper,engine_catalog=self.engine)
        origin=s._rules_origin(self.rules)
        for field in origin.__dataclass_fields__:
            old=getattr(origin,field)
            try:
                object.__setattr__(origin,field,old+'changed')
                with self.subTest(field=field),self.assertRaises(ValueError):h.check_compiled(self.compiled)
            finally:object.__setattr__(origin,field,old)
        self.assertIs(h.check_compiled(self.compiled),self.data)

    def test_nested_metadata_wrong_catalog_and_vocabulary_fail_closed(self):
        original=self.metadata['moves'][1]['flags'].copy()
        try:
            self.metadata['moves'][1]['flags']['changed']=True
            with self.assertRaises(ValueError):h.check_compiled(self.compiled)
        finally:self.metadata['moves'][1]['flags'].clear();self.metadata['moves'][1]['flags'].update(original)
        with self.assertRaises(ValueError):h.compile_hook_source(self.rules,self.helper,engine_catalog=self.f.lifecycle_catalog)
        bad=deepcopy(self.metadata);bad['vocabulary']['items'][1]='choiceband'
        result=subprocess.run([self.f.node,str(h._script()),str(self.f.root)],input=json.dumps(bad),text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        for field in ('vocabulary_sha256','rules_sha256','metadata_sha256','generator_sha256'):
            payload=deepcopy(self.manifest);payload[field]='0'*64
            with self.subTest(field=field),self.assertRaises(ValueError):h._validate(payload,self.metadata,self.rules,s._rules_origin(self.rules))
        payload=deepcopy(self.manifest);payload['rows']['items'].pop()
        with self.assertRaises(ValueError):h._validate(payload,self.metadata,self.rules,s._rules_origin(self.rules))

    def test_unknown_missing_types_flags_and_hash_cannot_be_absence(self):
        rows=self.manifest['rows']['abilities'];index=self.metadata['vocabulary']['abilities'].index('pressure')
        for edit in ('type','key','metadata','Order','flags'):
            altered=deepcopy(rows)
            if edit=='flags':altered[index]['flags']=h.DAMAGE
            else:altered[index]['entries'][0].pop(edit,None)
            data=h._rows(altered,'abilities')
            self.assertFalse(bool(data.manifest_known[index]))
            self.assertEqual(int(data.classification[index]),h.UNSUPPORTED)
        payload=deepcopy(self.manifest);payload['rows']['abilities'][index]['entries'][0].pop('sha256')
        with self.assertRaises(ValueError):h._validate(payload,self.metadata,self.rules,s._rules_origin(self.rules))
        # Exact non-undefined constants are callback presence too.
        rows=deepcopy(rows);entry=rows[index]['entries'][0]
        entry.update(key='onDamage',metadata=False,type='boolean',Order=None,Priority=None,SubOrder=None)
        rows[index]['flags']=h.DAMAGE
        self.assertEqual(int(h._rows(rows,'abilities').classification[index]),h.RELEVANT_UNREVIEWED)

    def test_original_holders_all_array_edges_and_weak_key_lifetime(self):
        with self.assertRaises(ValueError):h.check_compiled(replace(self.compiled))
        old=self.compiled.device
        for name in h.Data._fields:
            if name in h.KINDS:
                value=getattr(old,name);bad=old._replace(**{name:value._replace(relevant_flags=np.asarray(value.relevant_flags))})
            else:bad=old._replace(**{name:j.array(getattr(old,name),copy=True)})
            try:
                object.__setattr__(self.compiled,'device',bad)
                with self.subTest(field=name),self.assertRaises(ValueError):h.check_compiled(self.compiled)
            finally:object.__setattr__(self.compiled,'device',old)
        compiled=h.compile_hook_source(self.rules,self.helper,engine_catalog=self.engine)
        weak=weakref.ref(compiled);del compiled;gc.collect();self.assertIsNone(weak())

    def test_all_closed_host_apis_reject_before_subprocess_or_original_guards_and_hot_device_arrays(self):
        calls=(lambda:h.compile_hook_source(self.rules,self.helper,engine_catalog=self.engine),
            lambda:h.check_compiled(self.compiled),lambda:h.manifest_json(self.compiled))
        for call in calls:
            for mode in (0,1):
                fn=(lambda _:call()) if mode else (lambda:call())
                with patch.object(s,'_rules_origin',side_effect=AssertionError('late original guard')),patch.object(subprocess,'run',side_effect=AssertionError('late loader')):
                    with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'cannot be traced'):
                        jax.jit(fn)(j.array(1)) if mode else jax.jit(fn)()
        def device_read(index):
            return self.data.abilities.relevant_flags[index],self.data.items.classification[index]
        fast=jax.jit(device_read)
        for _ in range(2):
            result=fast(j.array([0,1],j.int32));np.testing.assert_array_equal(result[0],self.data.abilities.relevant_flags[:2])
        self.assertEqual(fast._cache_size(),1)
        self.assertIs(h.check_compiled(self.compiled),self.data)


if __name__=='__main__':unittest.main()
