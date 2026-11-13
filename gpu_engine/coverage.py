"""Source-assessed selected-effect ledger, distinct from game qualification.

Weights come from actual nonempty accepted-pool move slots. Registry entries
name the bounded effect and its checks; they do not assert full Gen3 parity.
The immutable pool's historical implemented=False fields are not rewritten.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np


def registry():
    result={}
    def group(names,effect,module,tests,*,partial=None):
        for name in names.split():
            if name in result:raise ValueError('duplicate effect registration')
            result[name]=dict(status='PARTIAL' if partial else 'SELECTED_EFFECT_IMPLEMENTED',
                selected_effect=effect,runtime=[f'gpu_engine/{module}.py','gpu_engine/transition.py'],
                check_references=[f'tests/test_gpu_engine_{t}.py' for t in tests.split()],
                missing_intrinsic_effect=partial,
                evidence='Source implementation and authored check references; not full-game qualification')
    group('earthquake surf hydropump quickattack aerialace drillpeck megahorn dragonclaw extremespeed machpunch magicalleaf',
        'Singles ordinary damage, catalog accuracy/always-hit and priority',
        'mechanics','mechanics transition')
    group('icebeam icepunch thunderbolt thunderpunch thunder bodyslam sludgebomb fireblast flamethrower firepunch shadowball crunch psychic rockslide',
        'Ordinary damage plus selected target status, SpD drop or flinch; represented secondary abilities',
        'secondary','secondary')
    group('toxic poisonpowder thunderwave stunspore willowisp sleeppowder hypnosis spore',
        'Major-status application, represented type/ability admission and Gen3 counters; clauses excluded',
        'effects','effects secondary')
    group('swordsdance calmmind bulkup agility dragondance tailglow amnesia irondefense barrier meditate howl doubleteam',
        'Selected self boost stages with +/-6 clamps','effects','effects')
    group('substitute protect endure detect',
        'Substitute HP/cost and selected stall-counter defense scheduling','effects','defense')
    group('bellydrum','Gen3 live-HP threshold, direct half-max HP cost and Attack cap',
        'effects','belly_drum')
    group('recover softboiled milkdrink slackoff rest haze',
        'Half healing, Rest full heal/sleep, or Haze boost reset','effects','effects')
    group('painsplit','Live HP average, per-mon max-HP clamps and target Substitute/Protect gates',
        'effects','pain_split')
    group('return frustration','Per-Pokemon happiness power','variable_moves','variable_moves')
    group('reversal flail','Gen3 six-band integer live-HP base power','variable_moves','variable_moves')
    group('doubleedge','Floored actual-damage recoil, including Substitute; Rock Head','damage_moves','damage_moves')
    group('seismictoss nightshade sonicboom dragonrage','Fixed level/constant damage with represented target defenses',
        'damage_moves','damage_moves')
    group('explosion selfdestruct','Halved defense, Damp veto, PP and self-KO scheduling','self_destruct','self_destruct')
    group('sleeptalk','Selected caller eligibility, PP/Choice/Encore and deferred move scheduling','sleep_talk','sleep_talk')
    group('batonpass','Direct Baton transfers represented boosts/volatiles, suspends and resumes original turn',
        'baton','baton ko_residual')
    group('wish leechseed','Side-slot Wish and singles Leech Seed residuals; Liquid Ooze outside selected scope',
        'sustain','sustain')
    group('spikes','Layer placement, grounded entry damage and hazard-KO replacement scheduling','entry','entry ko_residual')
    group('synthesis morningsun moonlight','Gen3 fractional healing under live effective weather','effects','weather_healing')
    group('raindance sunnyday sandstorm','Five ordinary-turn weather clocks and represented weather interactions','weather','weather')
    group('encore','Gen3 last caller, duration/PP and ordinary-turn scheduling','encore','encore')
    group('focuspunch','Queued preparation, hit/PP/Choice veto and deferred-turn state','focus_punch','focus_punch')
    group('healbell','Living own-party major-status cure, including per-holder Soundproof','effects','heal_bell')
    group('roar whirlwind','Selected phazing, grounded entry hazards and represented blockers','phazing','phazing')
    group('meanlook spiderweb block','Stable party-index move-trap links and Baton/ordinary departure cleanup','linked_traps','linked_traps')
    group('pursuit','Gen3 pre-switch interception and ordinary-hit fallback','pursuit','pursuit')
    group('crosschop razorleaf leafblade','Ordinary damage with catalog high-crit stage','critical','critical')
    group('rapidspin','Gen3 damage and represented Leech Seed/Spikes/partial-trap cleanup (fidelity M2: Wrap released, a '
        'Substitute hit included)','sustain','rapid_spin fidelity_m2')
    group('brickbreak','Ordinary Fighting damage','mechanics','mechanics',
        partial='Reflect/Light Screen removal is not represented')
    group('sing','Primary sleep status application','effects','effects',
        partial='Soundproof admission guard is missing for this sound move')
    # Fidelity patch A1. The status group above also gained Sleep Clause Mod and
    # Insomnia/Vital Spirit admission (its text predates them and is unchanged).
    group('lovelykiss','Primary sleep at 75% accuracy, Sleep Clause Mod and Insomnia/Vital Spirit admission',
        'effects','fidelity_a1')
    group('curse','Non-Ghost user +1 Atk/+1 Def/-1 Spe with +/-6 clamps (no pool Ghost user)','effects','fidelity_a1')
    group('superpower','Ordinary damage plus user -1 Atk/-1 Def after a successful hit, Substitute and KO included; '
        'a White Herb holder (Deoxys forms always hold one) restores them once at the move\'s AfterMove (fidelity WH)',
        'effects','fidelity_a1 fidelity_wh')
    group('charm','Target Attack -2 with Protect/Substitute gates and Clear Body/White Smoke/Hyper Cutter blocks',
        'effects','fidelity_a1')
    group('aromatherapy','Living own-party major-status cure, Soundproof holders included','effects','fidelity_a1')
    group('refresh','User paralysis/burn/poison cure; sleep and freeze fail','effects','fidelity_a1')
    # Fidelity patch A2.
    group('knockoff','Ordinary Dark damage plus permanent target item removal after the hit (KO included), '
        'Sticky Hold block and Substitute exemption; end-of-turn Choice Band unlock','items','fidelity_a2')
    group('gigadrain','Ordinary damage plus drain max(1,floor(dealt/2)) (ceil through a Substitute) and Liquid Ooze reversal',
        'damage_moves','fidelity_a2')
    group('counter mirrorcoat','Priority -5, twice the last physical-or-Hidden-Power (Counter) or special (Mirror Coat) foe hit '
        'this turn; silent failure without one; Ghost/Dark immunity','damage_moves','fidelity_a2')
    group('destinybond','Volatile until the next move attempt; a foe move KO (Pursuit included) faints the attacker',
        'effects','fidelity_a2')
    # Fidelity patch R.
    group('bonemerang','Two hits after one accuracy roll, each with its own critical and damage roll (the second from the '
        'appended effect stream); a first-hit KO ends it; Substitute per hit (a broken one lets the second hit reach the '
        'Pokemon); Endure per hit; Counter/Mirror Coat read the last hit','damage_moves','fidelity_r')
    # Fidelity patch B (chance secondaries on the shared secondary roll; Serene Grace doubles each).
    group('irontail','Ordinary damage plus 30% target Def -1; Shield Dust, a Substitute hit or a fainted target blocks it, '
        'Clear Body/White Smoke delete it silently, -6 is a no-op','secondary','fidelity_b')
    group('steelwing meteormash','Ordinary damage plus 10% user Def +1 (Steel Wing) or 20% user Atk +1 (Meteor Mash), '
        'which survives Shield Dust, a Substitute and a KO; +6 is a no-op','secondary','fidelity_b')
    group('headbutt needlearm','Ordinary damage plus 30% flinch: Inner Focus, Focus Punch\'s focus, Shield Dust, a Substitute '
        'or a fainted target blocks it; it matters only before the target moves','secondary','fidelity_b')
    group('signalbeam','Ordinary damage plus 10% confusion for 2-5 moves (the attacker\'s draws[s][7]): Own Tempo, Shield '
        'Dust, a Substitute hit, a fainted or already confused target block it; 50% 40 BP typeless self-hit, snap-out, '
        'Lum Berry and Own Tempo cures, Baton Pass carry','secondary','fidelity_b')
    # Fidelity patch M1.
    group('hyperbeam','Ordinary damage plus mustrecharge after a hit (Substitute and KO included): the next action is '
        'the priority-0 recharge pseudo-move, ahead of sleep, freeze, flinch, confusion and Truant, with only that '
        'request lane legal and switching trapped; it expires after the next residual and ends on a switch or faint',
        'effects','fidelity_m1')
    group('trick','Item swap between the actives (an identical pair included); fails on a miss, Protect, a Substitute, '
        'Sticky Hold, a knocked-off item on either side or no item on both; the Choice Band lock follows the current '
        'item','items','fidelity_m1')
    # Fidelity patch M2 (the residual scheduler: order-10 list of 14 events, the order-12 Perish pass).
    group('perishsong','Every active without the volatile and without Soundproof (the user included) counts 4, 3, 2, 1 at '
        'the order-12 residual and faints at 0; Protect and a Substitute do not block it; both ends resolve before '
        'faintMessages; Baton Pass carries it, a switch clears it','effects','fidelity_m2')
    group('yawn','Drowsy for 2 residuals (subOrder 19), then trySetStatus(slp) for 2+effect[s][2]: silent if statused, '
        'Sleep Clause Mod applies; fails on a status, Protect, a Substitute, an existing yawn and Insomnia/Vital Spirit; a '
        'switch clears it, Baton Pass does not carry it','effects','fidelity_m2 residual_order')
    group('wrap','Ordinary damage plus a partial trap of 3+draws[s][7] (no restart, not on a Substitute hit): '
        'max(1,maxhp/16) per residual (subOrder 9, after Leftovers), d-1 ticks, a silent release once the trapper is not '
        'the active; switches illegal while the trapper is the living foe active; Rapid Spin and the holder\'s own '
        'Substitute end it; Baton Pass carries it with its source','effects','fidelity_m2 residual_order')
    # Fidelity WH (White Herb, generally, and C-lite: Sacred Fire's thaw and Facade).
    group('psychoboost overheat','Ordinary damage plus user SpA -2 after a successful hit; a White Herb holder restores every '
        'negative stage once (its item gone) at any move\'s AfterMove, at a SwitchIn event (after hazards, before the '
        'entrant\'s Intimidate in Gen 3) or at residual order 29; a holder at 0 HP cannot','effects','fidelity_wh')
    group('sacredfire','Ordinary Fire damage plus 50% burn; a frozen user (after the 20% thaw roll fails) still moves, '
        'thawed in useMove even when the move misses or fails','effects','fidelity_wh')
    group('facade','Ordinary damage at 140 base power when the user has par, brn, psn or tox (not slp; a frozen user '
        'cannot use it); Gen 3 burn halving still applies without Guts','variable_moves','fidelity_wh')
    return result


def ledger(metadata,*,root):
    rows=metadata['move_frequency'];total=metadata['move_slots'];known=registry()
    if not rows or len({r['id'] for r in rows})!=len(rows) or sum(r['count'] for r in rows)!=total:
        raise ValueError('unique actual nonempty move counts must reconcile')
    if any(type(r['count'])is not int or r['count']<=0 or r['id']=='_none' for r in rows):
        raise ValueError('positive nonpadding move counts required')
    result=[]
    for row in rows:
        name=row['id']
        if name.startswith('hiddenpower'):
            effect=dict(status='SELECTED_EFFECT_IMPLEMENTED',selected_effect='Per-Pokemon IV-derived Hidden Power type/power and Gen3 category',
                runtime=['gpu_engine/variable_moves.py','gpu_engine/transition.py'],
                check_references=['tests/test_gpu_engine_variable_moves.py'],missing_intrinsic_effect=None,
                evidence='Source implementation and authored check references; not full-game qualification')
        else:effect=known.get(name,dict(status='PLACEHOLDER',selected_effect=None,
            runtime=['gpu_engine/transition.py'],check_references=[],missing_intrinsic_effect='Intrinsic effect not assessed/implemented in selected registry',
            evidence='No selected-effect credit; static damage or no-op fallback'))
        if any(not (root/p).is_file() for p in effect['runtime']+effect['check_references']):
            raise ValueError(f'missing source/check reference for {name}')
        result.append(dict(id=name,count=row['count'],share=row['count']/total,**effect))
    shares={status:sum(r['count'] for r in result if r['status']==status)/total
        for status in ('SELECTED_EFFECT_IMPLEMENTED','PARTIAL','PLACEHOLDER')}
    return dict(schema='pokezero.gpu_selected_effect_ledger.v1',move_slots=total,moves=len(rows),rows=result,
        weighted_status_shares=shares,game_qualified_move_share=None,
        top49_selection_share=sum(r['count'] for r in rows[:49])/total,
        qualification='Selected-effect source assessment only; tests/references do not establish full-game correctness or the divergence gate. Partial moves receive no whole-move credit.',
        scope='Gen3 singles measurement subset. Sleep Clause Mod only (no Freeze Clause); no production encoder, full ability/contact/item/event coverage, or all volatile cross-products. Each move row states bounded selected support. Full-game qualification remains separate.')


def measure(pool_path,expected_sha):
    blob=Path(pool_path).read_bytes();digest=hashlib.sha256(blob).hexdigest()
    if digest!=expected_sha:raise ValueError('accepted pool digest mismatch')
    with np.load(io.BytesIO(blob),allow_pickle=False) as data:
        metadata=json.loads(data['metadata_utf8'].tobytes())
        values,counts=np.unique(data['moves'][data['moves']!=0],return_counts=True)
        observed={metadata['vocabulary']['moves'][int(v)]:int(c) for v,c in zip(values,counts)}
    if observed!={r['id']:r['count'] for r in metadata['move_frequency']}:
        raise ValueError('metadata frequencies differ from actual tensor slots')
    root=Path(__file__).resolve().parents[1]
    result=ledger(metadata,root=root)
    git=lambda *a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    if git('status','--porcelain','--','gpu_engine','tests'):raise RuntimeError('dirty ledger source or checks')
    result.update(source_commit=git('rev-parse','HEAD'),pool_sha256=digest,
        evidence_file_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
            for p in sorted({p for r in result['rows'] for p in r['runtime']+r['check_references']})})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--team-pool',required=True);p.add_argument('--expected-pool-sha',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('exclusive output required')
    result=measure(a.team_pool,a.expected_pool_sha)
    with out.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evidence_file_sha256')}))
