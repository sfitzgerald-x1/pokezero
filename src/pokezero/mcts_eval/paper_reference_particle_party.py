"""Necessary public species conditioning; no later trait or feasibility forcing."""
import hashlib

from .paper_reference import ReferenceRefusal
from .paper_reference_sampling import HiddenTeamDraw
from ..randbat import canonical_gen3_randbat_species_id
from ..showdown_fixture import pack_team


def condition_necessary_party(prior, history, opponent, check, *, membership_first=False,
                              native_membership_batch_size=0, deadline_at=None):
    required = set()
    for line in history:
        fields = line.split('|')
        event = fields[1].lower().lstrip('-') if len(fields) > 1 else ''
        if (event in ('transform', 'formechange', 'detailschange', 'replace')
                or (event == 'move' and len(fields) > 3 and fields[3].lower() == 'transform')
                or 'illusion' in line.lower() or 'imposter' in line.lower()):
            return dict(status='ORIGINAL_PROPOSAL', reason='species-changing public history')
        if event in ('switch', 'drag') and len(fields) >= 5 and fields[2].startswith(opponent+'a: '):
            species = canonical_gen3_randbat_species_id(fields[3].split(',', 1)[0].strip())
            if species not in prior.set_source.universes:
                raise ReferenceRefusal('particle public species absent from pinned source')
            required.add(species)
    if len(required) > 6:
        raise ReferenceRefusal('particle public party exceeds six species')
    if not required or prior.pending_transition is not None:
        return dict(status='ORIGINAL_PROPOSAL', reason='empty public membership or nested original anchor')
    receipt = dict(status='NECESSARY_MEMBERSHIP_REJECTION', required_species=sorted(required),
        complete_proposals=0, matches=0, membership_rejections=0,
        conditioned_later_traits=False, exact_posterior=False)
    original = prior.sampler
    if getattr(prior, 'diagnostic_oracle_team', None) is not None:
        # Truth is a fixed complete party, not an unknown-party proposal.
        # Keep checking public membership without reverting to the sampler's
        # membership-first/native unknown-team generator.
        class OracleMembershipSampler:
            def draw(self, known, rng):
                check()
                draw = original.draw(known, rng)
                receipt['complete_proposals'] += 1
                species = {canonical_gen3_randbat_species_id(mon.species) for mon in draw.team}
                if not required <= species:
                    raise ReferenceRefusal('oracle original party contradicts public membership')
                receipt['matches'] += 1
                return draw
        prior.sampler = OracleMembershipSampler()
        receipt.update(status='DIAGNOSTIC_ORACLE_MEMBERSHIP_CHECK', membership_first=False,
            unknown_party_sampled=False)
        return receipt
    if membership_first:
        receipt.update(membership_first=True, known_completions_materialized=0,
            complete_proposals_scope='complete unknown-party completions before known-set sampling',
            joint_law='original known-set kernel times original unknown-party kernel conditioned on necessary species',
            deterministic_stream_coupling_changed=True)
        class MembershipFirstSampler:
            def draw(self, known, rng):
                if native_membership_batch_size:
                    from .paper_reference_native_membership import draw_native_membership
                    return draw_native_membership(original, known, rng, required=frozenset(required),
                        check=check, receipt=receipt, batch_size=native_membership_batch_size,
                        deadline_at=deadline_at)
                return original.draw_membership_first(known, rng, required=frozenset(required),
                    check=check, receipt=receipt)
        prior.sampler = MembershipFirstSampler()
        return receipt
    class Sampler:
        def draw(self, known, rng):
            for _ in range(2048):
                check()
                draw = original.draw(known, rng)
                receipt['complete_proposals'] += 1
                check()
                if (type(draw) is not HiddenTeamDraw or len(draw.team) != 6
                        or hashlib.sha256(pack_team(draw.team).encode()).hexdigest() != draw.packed_team_sha256):
                    raise ReferenceRefusal('particle membership proposal lost complete-team provenance')
                species = {canonical_gen3_randbat_species_id(mon.species) for mon in draw.team}
                if len(species) != 6:
                    raise ReferenceRefusal('particle complete proposal violates species clause')
                if required <= species:
                    receipt['matches'] += 1
                    return draw
                receipt['membership_rejections'] += 1
            raise ReferenceRefusal('particle necessary membership exhausted fixed original-proposal cap')
    prior.sampler = Sampler()
    return receipt
