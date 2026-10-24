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
    group('recover softboiled milkdrink slackoff rest haze',
        'Half healing, Rest full heal/sleep, or Haze boost reset','effects','effects')
    group('painsplit','Live HP average, per-mon max-HP clamps and target Substitute/Protect gates',
        'effects','pain_split')
    group('return frustration','Per-Pokemon happiness power','variable_moves','variable_moves')
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
    group('rapidspin','Gen3 damage and represented Leech Seed/Spikes cleanup','sustain','rapid_spin',
        partial='Partial-trap damage/clocks are not represented or cleared')
    group('brickbreak','Ordinary Fighting damage','mechanics','mechanics',
        partial='Reflect/Light Screen removal is not represented')
    group('sing','Primary sleep status application','effects','effects',
        partial='Soundproof admission guard is missing for this sound move')
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
        scope='Gen3 singles measurement subset. No Sleep/Freeze Clause, production encoder, full ability/contact/item/event coverage, or all volatile cross-products. Each move row states bounded selected support. Full-game qualification remains separate.')


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
