"""Original membership rejection; transport batching, not a different proposal."""
import random
import time

from .paper_reference import ReferenceRefusal
from .paper_reference_sampling import HiddenTeamDraw, _fixture
from ..randbat import canonical_gen3_randbat_species_id
from ..showdown_fixture import pack_team
import hashlib


def validate_native_membership(value, *, membership_first):
    if type(value) is not int or not 0 <= value <= 16 or value and not membership_first:
        raise ReferenceRefusal("native membership batch requires membership-first and integer 0..16")
    return value


def draw_native_membership(sampler, known, rng, *, required, check, receipt, batch_size, deadline_at):
    sampler._validate_known(known)
    fixed = tuple(canonical_gen3_randbat_species_id(mon.species) for mon in known)
    if not isinstance(rng, random.Random):
        raise ReferenceRefusal("native membership requires an explicit Python random stream")
    receipt.update(native_membership_batch_size=batch_size, seed_coupling="same consumed original seed prefix",
                   unused_lookahead_seeds_consumed=False)
    receipt.setdefault("native_batches",0)
    receipt.setdefault("native_party_draws",0)
    for_offset = 0
    while for_offset < 2048:
        check()
        limit = min(batch_size, 2048-for_offset)
        # Look ahead on a CLONE. Advance the real stream by only the seeds
        # actually consumed, so accepted parties AND later known sets match.
        clone = random.Random()
        clone.setstate(rng.getstate())
        seeds = tuple(clone.getrandbits(32) for _ in range(limit*10))
        budget = None if deadline_at is None else max(0., (deadline_at-time.perf_counter())*1000)
        check()
        row = sampler.generator.generate_unknown_membership(seeds=seeds, known=fixed,
            required=sorted(required), max_proposals=limit, budget_ms=budget)
        consumed, complete, rejected = (row.get(k) for k in
            ("consumed", "completeProposals", "membershipRejections"))
        status = row.get("status")
        if (type(consumed) is not int or not 0 <= consumed <= len(seeds)
                or type(complete) is not int or not 0 <= complete <= limit
                or type(rejected) is not int or not 0 <= rejected <= complete
                or status not in ("ACCEPTED", "EXHAUSTED_BATCH", "DEADLINE_CANCELLED")):
            raise ReferenceRefusal("invalid native membership receipt")
        for index in range(consumed):
            if rng.getrandbits(32) != seeds[index]:
                raise ReferenceRefusal("native membership random-stream drift")
        receipt["complete_proposals"] += complete
        receipt["membership_rejections"] += rejected
        receipt["native_batches"] += 1
        receipt["native_party_draws"] += consumed
        for_offset += complete
        # Canonical absolute Python clock is authoritative: never accept
        # an expired native completion, or invent a deadline certificate.
        check()
        if status == "DEADLINE_CANCELLED":
            raise ReferenceRefusal("native membership cancelled before authoritative deadline")
        if status != "ACCEPTED":
            if complete == 0 or rejected != complete:
                raise ReferenceRefusal("native membership batch made no complete progress")
            continue
        unknown = tuple(_fixture(mon) for mon in row.get("unknown", ()))
        party_seeds = tuple(row.get("partySeeds", ()))
        species = fixed + tuple(canonical_gen3_randbat_species_id(mon.species) for mon in unknown)
        if (len(species) != 6 or len(set(species)) != 6 or not required <= set(species)
                or not 0 <= len(party_seeds) <= 10
                or party_seeds != seeds[consumed-len(party_seeds):consumed]
                or rejected != complete-1):
            raise ReferenceRefusal("native membership accepted completion/provenance drift")
        receipt["matches"] += 1
        team, known_receipts = sampler._draw_known(known, rng)
        check()
        team.extend(unknown)
        receipt["known_completions_materialized"] += 1
        return HiddenTeamDraw(tuple(team), tuple(known_receipts), party_seeds,
            hashlib.sha256(pack_team(tuple(team)).encode()).hexdigest())
    raise ReferenceRefusal("particle necessary membership exhausted fixed original-proposal cap")
