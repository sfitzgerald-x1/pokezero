"""Ordered current-parent static checks; no engine/model/game calls."""
from dataclasses import dataclass
from unittest import mock
import unittest
from gpu_engine import v4_encoder as e
import test_gpu_engine_v4_encoder as fixture

READS=[]
@dataclass
class Box:
    child: object
    def __getattribute__(self,name):
        if name=='child':READS.append(('attr',name))
        return object.__getattribute__(self,name)

class CurrentMap(dict):
    def __iter__(self):
        READS.append(('keys',tuple(dict.keys(self))))
        return dict.__iter__(self)
    def __getitem__(self,key):
        READS.append(('key',key));return dict.__getitem__(self,key)

class CurrentList(list):
    def __len__(self):READS.append(('length',list.__len__(self)));return list.__len__(self)
    def __iter__(self):
        READS.append(('iterator',))
        for x in list.__iter__(self):
            READS.append(('next',));yield x

class StaticGuardPlanTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    fixture.V4EncoderTests.setUpClass();cls.catalog=fixture.V4EncoderTests.catalog

 def compare(self,root,record,plan):
    results=[]
    for checked in (lambda:[e._check_static(getattr(root,name),snapshot)for name,snapshot in record],lambda:e._check_catalog_plan(root,plan)):
        READS.clear()
        try:checked();out=('accepted',)
        except Exception as error:out=(type(error).__name__,str(error))
        results.append((out,tuple(READS)))
    self.assertEqual(results[0],results[1]);return results[0][0]

 def test_current_nested_reads_mutations_and_equal_replacements(self):
    first=10**50;second=10**50+1;leaf=Box(first);mapping=CurrentMap(a=CurrentList([leaf,second]),b=first);root=Box(mapping)
    record=(('child',e._static_snapshot(root.child,set())),);plan=e._compile_static_plan(record)
    self.assertEqual(self.compare(root,record,plan),('accepted',))
    mutations=[(lambda:setattr(leaf,'child',second),lambda:setattr(leaf,'child',first)),
      (lambda:mapping['a'].append(first),lambda:mapping['a'].pop()),
      (lambda:dict.__setitem__(mapping,'b',second),lambda:dict.__setitem__(mapping,'b',first)),
      (lambda:setattr(root,'child',CurrentMap(mapping)),lambda:setattr(root,'child',mapping))]
    for change,restore in mutations:
        change()
        try:self.assertNotEqual(self.compare(root,record,plan),('accepted',))
        finally:restore()
        self.assertEqual(self.compare(root,record,plan),('accepted',))
    # Key namespace equality remains equality, not newly required key identity.
    old=CurrentMap([('long-key-not-interned-123456',first)]);root=Box(old);record=(('child',e._static_snapshot(root.child,set())),);plan=e._compile_static_plan(record)
    key=next(iter(old));replacement=('!'+key)[1:];old.clear();old[replacement]=first
    self.assertEqual(self.compare(root,record,plan),('accepted',))

 def test_real_snapshot_current_edges_order_and_recursive_fallback(self):
    catalog=self.catalog;record=e._CATALOGS[catalog];plan=e._STATIC_PLANS[catalog]
    self.assertIs(plan.record,record)
    original_getattribute=type(catalog).__getattribute__;names={name for name,_ in record}
    def current_root(obj,name):
        if obj is catalog and name in names:READS.append(('catalog_attr',name))
        return original_getattribute(obj,name)
    with mock.patch.object(type(catalog),'__getattribute__',current_root):
        self.assertEqual(self.compare(catalog,record,plan),('accepted',))
        self.assertEqual([name for kind,name in READS if kind=='catalog_attr'],[name for name,_ in record])
    original=catalog.vocabulary.aliases
    object.__setattr__(catalog.vocabulary,'aliases',dict(original))
    try:
        self.assertNotEqual(self.compare(catalog,record,plan),('accepted',))
        with self.assertRaises(ValueError):e.check_catalog(catalog)
    finally:object.__setattr__(catalog.vocabulary,'aliases',original)
    index=catalog.vocabulary._index;key=next(iter(index));old=index[key];index[key]=old+1
    try:self.assertNotEqual(self.compare(catalog,record,plan),('accepted',))
    finally:index[key]=old
    saved=e._STATIC_PLANS.pop(catalog)
    try:
        with mock.patch.object(e,'_check_static',wraps=e._check_static) as checked:
            self.assertIs(e.check_catalog(catalog),catalog);self.assertGreater(checked.call_count,0)
    finally:e._STATIC_PLANS[catalog]=saved
    with mock.patch.object(e,'_check_static',side_effect=AssertionError('compiled plan should walk its own instructions')):
        self.assertIs(e.check_catalog(catalog),catalog)

 def test_sequence_iteration_and_mapping_failure_order_unchanged(self):
    value=10**50;sequence=CurrentList([Box(value),value]);mapping=CurrentMap(a=sequence,b=value);root=Box(mapping)
    record=(('child',e._static_snapshot(root.child,set())),);plan=e._compile_static_plan(record)
    # Same-length replacement container remains refused at its original edge.
    original=mapping['a'];mapping['a']=CurrentList(original)
    try:self.assertNotEqual(self.compare(root,record,plan),('accepted',))
    finally:mapping['a']=original
    old=list(dict.items(mapping));mapping.clear();mapping.update(reversed(old))
    try:self.assertEqual(self.compare(root,record,plan)[1],'original encoder catalog namespace required')
    finally:mapping.clear();mapping.update(old)
    self.assertEqual(self.compare(root,record,plan),('accepted',))

if __name__=='__main__':unittest.main()
