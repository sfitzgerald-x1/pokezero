"""Literal original public counter algebra; no engine step/model/game/replay."""
from types import SimpleNamespace as NS
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_device_public_hooks as h,engine_device_trackers as tr
from pokezero import showdown as reference
from pokezero.belief import PublicBattleBeliefEngine
from test_gpu_engine_device_public_hooks import state,tables


def raw_state(batch=1):
    s=state(batch)
    return s._replace(hp=j.full_like(s.hp,160),max_hp=j.full_like(s.max_hp,160))


def value(c,name,side=0,physical=0):
    slot=c.trackers.physical_to_public[:,side,physical]
    return c.trackers.mon_features[name][j.arange(len(slot)),side,j.clip(slot,0,5)]


class PublicRoundCounterTests(unittest.TestCase):
    def test_toxic_stage_follows_original_status_public_turn_and_switch_reset(self):
        s=raw_state();c=h.initialize(s,tables());use=j.ones(1,bool)
        parser=reference._ReplayParser('unit-toxic',complete_prefix=True,hp_visibility={'p1':'exact','p2':'percentage'})
        for line in ('|switch|p1a: Snorlax|Snorlax, L80|160/160','|switch|p2a: Absol|Absol, L80|100/100','|turn|1'):
            parser._feed_line(line)
        def same():self.assertEqual(int(value(c,'toxic_stage')[0]),parser.toxic_stage['p1'])
        same()
        s=s._replace(status=s.status.at[:,0,0].set(4));h.observe(c,s,use)
        parser._feed_line('|-status|p1a: Snorlax|tox');same()
        s=s._replace(hp=s.hp.at[:,0,0].set(150));h.hp_line(c,s,0,use,source_class=h.HP_POST_ITEM)
        parser._feed_line('|-damage|p1a: Snorlax|150/160 tox|[from] psn');same()
        h.public_turn(c,s,use,j.asarray([2],j.int32));parser._feed_line('|turn|2');same()
        # An arriving poisoned member starts at public stage0; first actual
        # residual source line makes the known first stage observable.
        s=s._replace(active=s.active.at[:,0].set(1),status=s.status.at[:,0,1].set(4));h.install(c,s,0,use)
        parser._feed_line('|switch|p1a: Gardevoir|Gardevoir, L80|160/160 tox')
        self.assertEqual(int(value(c,'toxic_stage',physical=1)[0]),parser.toxic_stage['p1'])
        h.public_turn(c,s,use,j.asarray([3],j.int32));parser._feed_line('|turn|3')
        self.assertEqual(int(value(c,'toxic_stage',physical=1)[0]),0)
        s=s._replace(hp=s.hp.at[:,0,1].set(150));h.hp_line(c,s,0,use,source_class=h.HP_POST_ITEM)
        parser._feed_line('|-damage|p1a: Gardevoir|150/160 tox|[from] psn')
        self.assertEqual(int(value(c,'toxic_stage',physical=1)[0]),parser.toxic_stage['p1'])
        h.public_turn(c,s,use,j.asarray([4],j.int32));parser._feed_line('|turn|4')
        self.assertEqual(int(value(c,'toxic_stage',physical=1)[0]),parser.toxic_stage['p1'])
        s=s._replace(hp=s.hp.at[:,0,1].set(0));h.hp_line(c,s,0,use,source_class=h.HP_POST_ITEM)
        parser._feed_line('|-damage|p1a: Gardevoir|0 fnt|[from] psn');parser._feed_line('|faint|p1a: Gardevoir')
        self.assertEqual(int(value(c,'toxic_stage',physical=1)[0]),parser.toxic_stage['p1'])

    def test_sleep_talk_cant_skips_are_refunded_only_on_public_reentry(self):
        s=raw_state();c=h.initialize(s,tables());use=j.ones(1,bool)
        engine=PublicBattleBeliefEngine(format_id='gen3randombattle')
        def line(text):engine.ingest_event(reference._public_event_from_line(text))
        for text in ('|switch|p1a: Snorlax|Snorlax, L80|160/160','|switch|p2a: Absol|Absol, L80|100/100','|turn|1'):
            line(text)
        def same():
            source=next(v for v in engine.snapshot().side('p1')if v.species=='Snorlax')
            self.assertEqual(int(value(c,'sleep_turns')[0]),source.sleep_turns)
            self.assertEqual(int(value(c,'sleep_skipped_turns')[0]),source.sleep_skipped_turns)
        s=s._replace(status=s.status.at[:,0,0].set(5));h.observe(c,s,use);line('|-status|p1a: Snorlax|slp')
        result=NS(status=j.full(1,5,j.int32),can_act=use,confusion_self_hit=j.zeros(1,bool))
        for turn in (1,2):
            h.before_move(c,s,0,result,use,sleep_usable=True);line('|cant|p1a: Snorlax|slp')
            h.move(c,s,0,j.full(1,2,j.int32),use,pp_use=True);line('|move|p1a: Snorlax|Sleep Talk|p2a: Absol')
            h.move(c,s,0,j.full(1,3,j.int32),use,original=False);line('|move|p1a: Snorlax|Body Slam|p2a: Absol|[from] sleeptalk')
            same()
            h.turn_end(c,s,s,j.zeros(1,bool),upkeep=use);line('|upkeep')
            h.public_turn(c,s,use,j.full(1,turn+1,j.int32));line('|turn|'+str(turn+1))
        away=s._replace(active=s.active.at[:,0].set(1));h.install(c,away,0,use)
        line('|switch|p1a: Gardevoir|Gardevoir, L80|160/160')
        h.install(c,s,0,use);line('|switch|p1a: Snorlax|Snorlax, L80|160/160 slp');same()
        self.assertEqual(int(value(c,'sleep_turns')[0]),0)
        # A plain sleep cant clears prior skippedTime only at actual upkeep.
        h.before_move(c,s,0,result,use,sleep_usable=False);line('|cant|p1a: Snorlax|slp')
        h.turn_end(c,s,s,j.zeros(1,bool),upkeep=use);line('|upkeep');same()

    def test_jit_mixed_rows_mask_and_both_readout_owners_keep_fresh_counters(self):
        s=raw_state(3);use=j.asarray([True,False,True]);s=s._replace(status=s.status.at[:,0,0].set(4))
        @jax.jit
        def run(s):
            c=h.initialize(s,tables())
            # Actual new STATUS after initialization is distinct from a switch
            # condition which already contains tox.
            c.trackers=c.trackers._replace(status=c.trackers.status.at[:,0,0].set(0))
            h.observe(c,s,use)
            h.public_turn(c,s,use,c.trackers.turn+1)
            return c.trackers
        t=run(s);np.testing.assert_array_equal(t.mon_features['toxic_stage'][:,0,0],[2,0,2])
        own=tr.readout(t,j.zeros(3,j.int32));foe=tr.readout(t,j.ones(3,j.int32))
        np.testing.assert_array_equal(own.own_mon_features['toxic_stage'][:,0],foe.mon_features['toxic_stage'][:,0])
        # No source change to true numerical state, namespace or chance path.
        self.assertEqual(int(reference.v4_numeric_index(reference.NUMERIC_TOXIC_STAGE)),17)
        self.assertEqual(int(reference.v4_numeric_index(reference.NUMERIC_SLEEP_TURNS)),24)

    def test_jit_sleep_refund_masks_called_children_and_nonusable_cants(self):
        s=raw_state(3);s=s._replace(status=s.status.at[:,0,0].set(5))
        @jax.jit
        def run(s):
            c=h.initialize(s,tables());use=j.asarray([True,True,False])
            h.before_move(c,s,0,NS(status=j.full(3,5,j.int32),can_act=j.asarray([True,False,False]),
                confusion_self_hit=j.zeros(3,bool)),use,sleep_usable=j.asarray([True,False,True]))
            h.move(c,s,0,j.full(3,2,j.int32),j.asarray([True,False,False]),pp_use=True)
            # A child cannot charge another skipped turn.
            h.move(c,s,0,j.full(3,3,j.int32),use,original=False)
            away=s._replace(active=s.active.at[:,0].set(1));h.install(c,away,0,use)
            h.install(c,s,0,use)
            return c.trackers
        t=run(s)
        np.testing.assert_array_equal(t.mon_features['sleep_turns'][:,0,0],[0,1,0])
        np.testing.assert_array_equal(t.mon_features['sleep_skipped_turns'][:,0,0],0)


if __name__=='__main__':unittest.main()
