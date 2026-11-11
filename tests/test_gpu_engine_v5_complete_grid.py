"""Two original posterior worlds, ALL legal pairs and current public leaves.

Tiny original source/kernel components with registered untrained fake heads;
no scientific roots, actual models, Native, games, cluster or action-mask edits.
The production boundary is128; this CPU component uses genuine B8 chunks over
its entire original grid and makes no hardware throughput claim.
"""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import jax.numpy as j
import numpy as np

from gpu_engine import v5_depth1_provider as provider,v5_hypothetical_batch as batch
from gpu_engine import private_team,public_field,transition
from pokezero import v5_contract as contract
import test_gpu_engine_v5_depth1_provider as fixtures


def value_head(inputs):
    numbers=inputs['numeric_features'][:,0]
    hp=private_team.NUMERIC_COLUMNS[private_team.CORE.index('HP_FRACTION')]
    active=private_team.NUMERIC_COLUMNS[private_team.CORE.index('ACTIVE')]
    own_hp=j.sum(numbers[:,1:7,hp]*numbers[:,1:7,active],axis=1)
    turn=numbers[:,0,public_field.NUMERIC_COLUMNS[public_field.CORE.index('TURN_COUNT')]]
    return j.tanh(.75*own_hp+10*turn)


class CompleteGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):fixtures.ProviderTests.setUpClass.__func__(cls)
    setup=fixtures.ProviderTests.setup

    def test_switched_tied_positive_heal_stays_unknown_at_original_conditional_guard(self):
        with ExitStack() as stack:
            teacher,context,(learner,_),start=self.setup(stack,head_value=value_head)
            captured=provider.capture.public_root(context.history,config=self.f.config,contract=self.f.contract,
                battle_id='literal-tie-diagnostic',seed=31)
            materialized=provider.worlds.materialize(captured.root,start,prepared=self.f.f.prepared,
                history=context.history,own_static=context.own_static,seed=5)
            self.assertIsNotNone(materialized.branch,materialized.refusals)
            root=materialized.branch;executor=batch._executor(root.prepared,learner.adapter)
            row=executor.prepare_row(root)
            cell=provider.GridCell(context.address,0,0,4,0)
            row,_=batch._transport(row,root,cell,1,0)
            routes=executor.route(row.core,j.array([[4,0]],j.int32))
            self.assertTrue(bool((routes[0].admitted&routes[1].admitted).all()))
            actions=j.stack([x.engine_action for x in routes],axis=1)
            chance=transition.sample_chance_inputs(root.state)
            result=executor.run(row.core,row.publics,row.streams,row.record,actions,chance,j.ones(1,bool))
            state=result.frame.state;raw=result.frame.callbacks
            speeds=transition._speeds(state,self.f.runtime['request']['engine_table'])
            # These literal pure-damage/Leftovers fixtures cannot change Speed.
            # Thus final effective Speed equals the original cached pre-Wish
            # Speed; retain the raw writes/profile as direct evidence too.
            np.testing.assert_array_equal(state.stats,row.core.engine.stats)
            np.testing.assert_array_equal(state.ability,row.core.engine.ability)
            np.testing.assert_array_equal(state.item,row.core.engine.item)
            self.assertFalse(bool(state.boosts.any()));self.assertFalse(bool(state.status.any()))
            self.assertEqual(speeds[0,0],speeds[0,1])
            self.assertTrue(bool(result.frame.weather_profile.qualified_origin[0]))
            words=np.asarray(raw.words)[0,:int(raw.count[0])]
            heals=words[words[:,0]==17]
            self.assertEqual(len(heals),1);self.assertGreater(heals[0,6],heals[0,5])
            self.assertTrue(int(raw.refusal[0])&256)
            self.assertEqual(result.good.tolist(),[False])
            Path('/tmp/pokezero-v5-grid-tie-diagnostic.json').write_text(json.dumps(dict(
                cached_speed_equal=bool(speeds[0,0]==speeds[0,1]),effective_speed=np.asarray(speeds).tolist(),
                qualified_origin=np.asarray(result.frame.weather_profile.qualified_origin).tolist(),
                raw_words=words.tolist(),diagnostics=np.asarray(result.diagnostics).tolist()))+'\n')

    def test_two_original_worlds_every_legal_pair_current_leaf_completed_target(self):
        with ExitStack() as stack:
            # Separate coherent no-tie fixture: ALL original own slots L79,
            # source-sampled foe slots L80. Recompute exact original spreads/
            # stats/HP for all6. No legal mask, source axis or guard changes.
            teacher,context,(learner,opponent),_=self.setup(stack,head_value=value_head,all_own_levels=True)
            real=batch.iter_execute;chunks=[];genuine_values={};declarations=[]
            def bounded(declared,adapter,*,max_rows):
                self.assertEqual(max_rows,128) # unchanged production boundary
                declarations.append(declared)
                for chunk in real(declared,adapter,max_rows=8):
                    chunks.append(chunk)
                    if chunk.inputs is not None:
                        values=np.asarray(value_head(chunk.inputs))
                        for position,index in enumerate(chunk.model_indices):
                            cell=chunk.cells[index]
                            genuine_values[(cell.world,cell.root_action,cell.reply_action)]=float(values[position])
                    yield chunk
            stack.enter_context(patch.object(batch,'iter_execute',side_effect=bounded))
            sample=stack.enter_context(patch.object(provider.planner,'_gen3_randbat_belief_start_override_result',
                wraps=provider.planner._gen3_randbat_belief_start_override_result))
            label=teacher.label_batch((context,),learner=learner,opponents=(opponent,))[0]
            diagnostic=dict(status=label.status,refusal=label.refusal,report=label.report,
                chunks=[dict(cells=len(x.cells),model_rows=len(x.model_indices),admitted=np.asarray(x.admitted).tolist(),
                    refusals=x.refusals,diagnostics=x.diagnostics,device_rows=x.device_rows,
                    active_rows=x.active_rows,padding=x.padding) for x in chunks])
            Path('/tmp/pokezero-v5-complete-grid-diagnostic.json').write_text(json.dumps(diagnostic,sort_keys=True)+'\n')
            self.assertEqual(sample.call_count,2)
            self.assertEqual(len(declarations),1)
            declared=declarations[0]
            self.assertEqual(tuple(cell for chunk in chunks for cell in chunk.cells),declared.cells)
            self.assertEqual(len(declared.roots),2)
            self.assertIs(declared.roots[0].public,declared.roots[1].public)
            self.assertGreater(sum(x.active_rows>1 for x in chunks),0)
            self.assertTrue(all(x.device_rows==8 and x.active_rows+x.padding==8 for x in chunks))
            self.assertEqual(label.status,'LABELED',dict(status=label.status,refusal=label.refusal,
                completed=label.report['completed_pairs'],unknown=label.report['unknown_pairs'],
                reasons=label.report['branch_refusals']))
            self.assertIsNotNone(label.improvement)
            self.assertEqual(label.report['unknown_worlds'],0)
            self.assertEqual(label.report['unknown_pairs'],0)
            self.assertEqual(label.report['unknown_reply_mass'],0)
            self.assertEqual(label.report['completed_pairs'],label.report['required_pairs'])
            self.assertEqual(label.report['required_pairs'],len(declared.cells))
            legal=tuple(context.public_observation.legal_action_mask)
            expected=[[] for _ in range(9)]
            for world,slot in enumerate(label.report['world_slots']):
                replies=tuple(i for i,active in enumerate(slot['reply_legal']) if active)
                self.assertEqual(slot['required_pairs'],sum(legal)*len(replies))
                self.assertEqual(slot['completed_pairs'],slot['required_pairs'])
                self.assertTrue(any(cell.world==world and cell.reply_action==1 for cell in declared.cells))
                maximum=max(slot['reply_logits'][i] for i in replies)
                weights={i:math.exp(slot['reply_logits'][i]-maximum) for i in replies};total=math.fsum(weights.values())
                # The underflowed reply remains priced even though its floating
                # weight is0. No mask/denominator was narrowed to create success.
                self.assertEqual(weights[1],0)
                for action,use in enumerate(legal):
                    if not use:continue
                    for reply in replies:
                        actual=slot['joint_values'][action][reply]
                        self.assertAlmostEqual(actual,genuine_values[(world,action,reply)],places=6)
                        expected[action].append(weights[reply]/total*actual/2)
            q=tuple(math.fsum(x) if use else None for x,use in zip(expected,legal))
            np.testing.assert_allclose([v for v in label.improvement.completed_q if v is not None],
                [v for v in q if v is not None],rtol=0,atol=1e-7)
            self.assertGreater(len({round(v,6) for v in genuine_values.values()}),1)
            self.assertNotAlmostEqual(label.improvement.root_value,context.raw_public_value,places=6)
            label.improvement.validate(legal)
            labels=contract.V5Labels(context.address,terminal_status='COMPLETE',outcome=1.,
                root_improvement=label.improvement,search_status='LABELED')
            expected_mixed=(1+label.improvement.root_value)/2
            for _ in range(2):self.assertAlmostEqual(labels.columns(context.public_observation)['search_value_target'],expected_mixed)
