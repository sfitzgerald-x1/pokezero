"""Original source writer and declared typed HP fixtures, not game replay."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from pokezero import showdown as original
from gpu_engine import engine_device_public_hooks as h, engine_device_trackers as tr
import test_gpu_engine_device_public_hooks as fixtures


def hp(state,side,value):
    return state._replace(hp=state.hp.at[:,side,0].set(value))


class DeviceDamageHistoryTests(unittest.TestCase):
    def parser(self,own):
        p=original._ReplayParser()
        p.feed(['|gen|3','|gametype|singles',
            '|switch|p1a: Lapras|Lapras, L80|'+('331/331' if own==0 else '100/100'),
            '|switch|p2a: Electrode|Electrode, L82|'+('100/100' if own==0 else '233/233'),
            '|turn|1'])
        return p

    def test_literal_residual_operands_match_original_both_observer_writers(self):
        self.assertEqual(original.v4_numeric_index(original.NUMERIC_LAST_DAMAGE_TAKEN),39)
        # Initial full HP here is DECLARED synthetic source state, not a claim
        # to reconstruct the old game's unrecorded prefix. Following HP writes
        # are the literal recorded first-turn operands in the compact packet.
        state=fixtures.state(1)
        state=state._replace(hp=state.hp.at[:,0,0].set(331).at[:,1,0].set(233),
            max_hp=state.max_hp.at[:,0,0].set(331).at[:,1,0].set(233))
        c=h.initialize(state);yes=j.ones(1,bool)
        hit=hp(state,1,151)
        h.damage(c,hit,0,j.asarray([82]),j.asarray([82]),yes);h.observe(c,hit,yes)
        healed=hp(hit,1,165);h.observe(c,healed,yes)
        residual=hp(healed,0,311)
        h.hp_line(c,residual,0,yes,source_class=h.HP_POST_ITEM)
        # A no-TURN boundary must retain last round, not expose current hits.
        self.assertEqual(float(tr.readout(c.trackers,j.asarray([0],j.int32)).own_mon_features['last_damage_taken'][0,0]),0.)
        h.public_turn(c,residual,yes,j.asarray([2],j.int32))
        views=[tr.readout(c.trackers,j.asarray([owner],j.int32)) for owner in (0,1)]
        for owner,view in enumerate(views):
            parser=self.parser(owner)
            parser.feed(['|move|p2a: Electrode|Toxic|p1a: Lapras','|-status|p1a: Lapras|tox',
                '|move|p1a: Lapras|Surf|p2a: Electrode',
                '|-damage|p2a: Electrode|'+('65/100' if owner==0 else '151/233'),
                '|','|-heal|p2a: Electrode|'+('71/100' if owner==0 else '165/233')+'|[from] item: Leftovers',
                '|-damage|p1a: Lapras|'+('311/331 tox' if owner==0 else '94/100 tox')+'|[from] psn',
                '|upkeep','|turn|2'])
            mine,foe=('p1','p2') if owner==0 else ('p2','p1')
            for key,source in (('last_damage_taken',parser.last_damage_taken),('last_damage_dealt',parser.last_damage_dealt)):
                np.testing.assert_allclose(view.own_mon_features[key][0,0],source[mine],rtol=0,atol=1e-7)
                np.testing.assert_allclose(view.mon_features[key][0,0],source[foe],rtol=0,atol=1e-7)
        np.testing.assert_allclose(views[0].own_mon_features['last_damage_taken'][0,0],20/331,rtol=0,atol=1e-7)
        np.testing.assert_allclose(views[1].mon_features['last_damage_taken'][0,0],.06,rtol=0,atol=1e-7)

    def test_no_turn_retains_previous_round_and_new_stay_settles_net_zero_residual(self):
        state=fixtures.state(1)._replace(hp=j.full((1,2,6),100,j.int32),max_hp=j.full((1,2,6),100,j.int32))
        c=h.initialize(state);yes=j.ones(1,bool)
        first=hp(state,1,80)
        h.damage(c,first,0,j.asarray([20]),j.asarray([20]),yes);h.observe(c,first,yes)
        h.public_turn(c,first,yes,j.asarray([2],j.int32))
        next_hit=hp(first,1,52)
        h.damage(c,next_hit,0,j.asarray([28]),j.asarray([28]),yes);h.observe(c,next_hit,yes)
        self.assertAlmostEqual(float(tr.readout(c.trackers,j.asarray([0],j.int32)).mon_features['last_damage_taken'][0,0]),.2,places=7)
        # Known-return installation has fresh counters even though the old
        # encounter's previous stay had direct damage .20 and current .28.
        away=next_hit._replace(active=next_hit.active.at[:,1].set(1));h.install(c,away,1,yes)
        incoming=away._replace(active=away.active.at[:,1].set(0),
            hp=away.hp.at[:,1,0].set(165),max_hp=away.max_hp.at[:,1,0].set(233))
        h.install(c,incoming,1,yes)
        sand=hp(incoming,1,151);h.hp_line(c,sand,1,yes,source_class=h.HP_PRE_ITEM)
        healed=hp(sand,1,165);h.observe(c,healed,yes)
        h.public_turn(c,healed,yes,j.asarray([7],j.int32))
        owner=tr.readout(c.trackers,j.asarray([1],j.int32));peer=tr.readout(c.trackers,j.asarray([0],j.int32))
        np.testing.assert_allclose(owner.own_mon_features['last_damage_taken'][0,0],14/233,rtol=0,atol=1e-7)
        np.testing.assert_allclose(peer.mon_features['last_damage_taken'][0,0],.06,rtol=0,atol=1e-7)
        self.assertEqual(float(owner.own_mon_features['last_damage_dealt'][0,0]),0.)
        self.assertEqual(float(peer.mon_features['last_damage_dealt'][0,0]),0.)
        # The actual original writer uses the same 165->151->165 conditions:
        # heal restores the HP baseline but never erases accumulated loss.
        parser=self.parser(1)
        parser.feed([f'|turn|{turn}' for turn in range(2,7)])
        parser.feed(['|switch|p2a: Electrode|Electrode, L82|165/233',
            '|-damage|p2a: Electrode|151/233|[from] Sandstorm',
            '|-heal|p2a: Electrode|165/233|[from] item: Leftovers','|turn|7'])
        np.testing.assert_allclose(owner.own_mon_features['last_damage_taken'][0,0],parser.last_damage_taken['p2'],rtol=0,atol=1e-7)

    def test_jit_mixed_rows_positive_sums_and_sethp_is_not_damage(self):
        state=fixtures.state(2)._replace(hp=j.full((2,2,6),100,j.int32),max_hp=j.full((2,2,6),100,j.int32))
        @jax.jit
        def fixture(state):
            c=h.initialize(state);use=j.asarray([True,False])
            a=hp(state,0,70);h.hp_line(c,a,0,use,source_class=h.HP_PRE_ITEM)
            b=hp(a,0,100);h.observe(c,b,use)
            d=hp(b,0,50);h.hp_line(c,d,0,use,source_class=h.HP_POST_ITEM)
            h.public_turn(c,d,use,j.asarray([2,2],j.int32))
            settled=c.trackers
            # A typed actual -sethp updates its baseline without counting as a
            # damage line; the following 30->20 damage adds .10, never .80.
            e=hp(d,0,30);h.hp_line(c,e,0,use,source_class=h.HP_ACTION,record_damage=False)
            f=hp(e,0,20);h.hp_line(c,f,0,use,source_class=h.HP_ACTION)
            h.public_turn(c,f,use,j.asarray([3,3],j.int32))
            return settled,c.trackers
        first,second=fixture(state)
        np.testing.assert_allclose(first.damage_exact_taken[:,0,0],[.8,0],rtol=0,atol=1e-7)
        np.testing.assert_allclose(second.damage_exact_taken[:,0,0],[.1,0],rtol=0,atol=1e-7)
        self.assertEqual(first.damage_history.hp.shape,(2,2))
        self.assertEqual(first.damage_history.taken.shape,(2,2,2,2))
        self.assertEqual(first.damage_history.taken.dtype,j.float32)


if __name__=='__main__':unittest.main()
