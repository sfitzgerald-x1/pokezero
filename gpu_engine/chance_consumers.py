"""Source-bound conservative chance consumers; not a fidelity qualification.

A plan partitions physical coordinates, not independently redrawn events.
Every positive successor must pass the full future-state mapper. One refusal
invalidates the WHOLE fixed slot; no prefix, redraw, or renormalization exists.
No sampled transition is used to decide inactivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from . import future_state as mapping
from . import mechanics
from .finite_law import Axis, COORDINATES, Observation, Plan as FinitePlan


# Literal consumer inventory of the reviewed transition source. A coordinate
# can have several uses: retain their JOINT observations on that one axis.
CONSUMERS = {
    **{f"draws.{s}.{c}": (use,) for s in range(2) for c, use in (
        (1, "mechanics.before_move: thaw<20"),
        (2, "mechanics.before_move: paralysis<25"),
        (3, "mechanics.before_move: confusion>=50"),
        (4, "transition/pursuit: critical<1 or<2"),
        (5, "transition/pursuit: damage85+draw; confusion damage reuses draw"),
        (6, "mechanics.accuracy_hit: pinned binary64/Q12 integer-roll predicates"),
        (7, "transition primary sleep; encore.apply duration"),
        (8, "effects.stall_success: draw<8//denominator"))},
    "draws.0.0": ("transition first move ordering",),
    "draws.1.0": ("transition residual SpeedBoost ordering",),
    "draws.0.9": ("transition residual item ordering",),
    "draws.1.9": ("transition residual major-status ordering",),
    "draws.0.10": ("transition residual LeechSeed ordering",),
    "draws.1.10": ("transition replacement runSwitch callback ordering",),
    "draws.0.11": ("transition switch installation ordering",),
    "draws.1.11": ("transition residual Encore ordering",),
    **{f"{name}.{s}": (use,) for s in range(2) for name, use in (
        ("phazing", "phazing.choose_bench modulo bench count OR secondary.apply_contact Static<20; dispatched Gen3 moves are disjoint"),
        ("secondary", "secondary.apply: shared draw<chance*SereneGrace"),
        ("sleep_talk", "sleep_talk.choose: draw modulo callable count before PP test"),
        ("entry_ties", "transition resumed switch/callback ordering"))},
    "initial_tie": ("transition.initialize unresolved initial entry ordering",),
}

# Changing any source that supplies the consumer/domain proof requires review.
SOURCE_PINS = {
    "transition.py": "981d26f9ccd8057e1072f15dacac8bb759da0f7dd26b6a544000790a50d39400",
    "mechanics.py": "e8476cc3e0029f3ed0bf82276312440e6dd8b340ffeb0c68cdd5611754eef661",
    "pursuit.py": "2b01102bb3819791fc1edea0ba0fe0a0b913e8dfa5a8f6501f48f3fe260a8309",
    "effects.py": "0138f0e90811c4829666960a106d80ed2d59f6cd06a8ff83324d189465f73104",
    "critical.py": "b66e6cf10c0d4df625b1c4f5e5db238af42052e556d7649a10f911745181401e",
    "secondary.py": "289f949b19371a7ebaffafbbfdf13d86adb387081b882ba5921381c9b84b6023",
    "sleep_talk.py": "968a78ddde87158b13f68dfcacc2158f07336d82c1be472d2283421e8583c154",
    "encore.py": "63da39bbd181a8c6371b3bca853322291a03674c60f535ad0eacdbe1a440e0dd",
    "phazing.py": "8f82d4894f03b39936de0e79b17f0b7715fdfe2f0edb6a7a42d44f1840671bee",
    "weather.py": "a5014e990f08bfb0cf04ac4aba5bc4cbc44717071601f80d3c340bfac79cade5",
    "state.py": "e657242e2d7191c4c7c1acb265d8f5751a6f00acaa08c08998af208d90124fe6",
    "entry.py": "b53080d3995b488477e28183d815aa8162fc91aef25e677c427e811c75714b2f",
    "action_history.py": "a47bdb2712f7eadaed12d91ede70be2ebe9f1c1ae3f70fe56495437997212407",
}
DOMAIN_MODULES = tuple("action_history absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather".split())
DOMAIN_SHA256 = "8009264a54997ef437a76d6dc3fbd43bb623bf9063355e6180d331cefe2c3cf9"


def _host(value):
    if hasattr(value, "_asdict"):
        return {name: _host(item) for name, item in value._asdict().items()}
    if isinstance(value, dict):
        return {name: _host(item) for name, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _at(table, family, field, index, default=0):
    group = table.get(family)
    return default if group is None else group[field][index]


@dataclass(frozen=True)
class ConsumerPlan:
    law: FinitePlan
    native_input: mapping.NativeInput
    native_actions: mapping.NativeActions
    reachable_moves: tuple[tuple[int, ...], tuple[int, ...]]
    reachable_participants: tuple[tuple[int, ...], tuple[int, ...]]
    source_certificate: tuple[tuple[str, str], ...]
    table_certificate: tuple
    metadata: dict
    cap: int

    @property
    def cardinality(self):
        return self.law.cardinality

    @property
    def axes(self):
        return self.law.axes

    def bounded_cases(self):
        return self.law.bounded_cases(self.cap)

    def canonical_successor(self, state, index):
        """Any returned REFUSED invalidates the whole slot, not only this case."""
        return mapping.canonical_jax_future(state, index, self.metadata,
            certificate=self.native_input.certificate)


def plan_for_state(state, index, actions, table, metadata, *, cap, native_input=None,
                   exact_move_slots=False):
    """Precount an exact conservative source partition; emit no sampled prefix.

    ``cap`` is mandatory and must be prospectively bound by the panel entry.
    Over-cap roots remain unknown fixed slots, never simplified or redrawn.
    Table arrays are the ACTUAL supplied kernel tables, not metadata guesses.
    """
    raw = ()
    original = None

    def refuse(code, detail):
        return mapping.Refused((mapping.Gap("chance_plan", code, detail),), raw,
            raw_actions=original)

    try:
        if type(cap) is not int or cap <= 0:
            return refuse("INVALID_EXPLICIT_CASE_CAP", "A positive explicit per-slot cap is required")
        original = mapping._freeze(actions)
        raw = mapping._freeze(mapping._sample(state, index))
        actual = tuple((name, sha256((Path(__file__).parent/name).read_bytes()).hexdigest())
                       for name in SOURCE_PINS)
        if actual != tuple(SOURCE_PINS.items()):
            return refuse("CONSUMER_SOURCE_DRIFT", repr(actual))
        domain = sha256(b"".join(name.encode()+b"\0"+(Path(__file__).parent/(name+".py")).read_bytes()+b"\0"
                                for name in DOMAIN_MODULES)).hexdigest()
        if domain != DOMAIN_SHA256:
            return refuse("CONSUMER_DOMAIN_DRIFT", domain)
        native = (mapping.native_input_from_jax(state, index, metadata)
                  if native_input is None else native_input)
        if isinstance(native, mapping.Refused):
            return native
        if native_input is not None:
            if not isinstance(native,mapping.NativeInput):
                return refuse("PROVIDED_NATIVE_INPUT_REQUIRED", "Explicit complete native correspondence required")
            expected=mapping.canonical_jax_future(state,index,metadata,certificate=native.certificate)
            physical=mapping.canonical_native_future(native.state,native.certificate)
            if isinstance(expected,mapping.Refused):return expected
            if isinstance(physical,mapping.Refused):return physical
            if expected.key!=physical.key:
                return refuse("PROVIDED_NATIVE_INPUT_MISMATCH", "Provided original-static correspondence differs from current physical state")
        bound_actions = mapping.native_actions_from_jax(state, index, actions, native,
            exact_move_slots=exact_move_slots)
        if isinstance(bound_actions, mapping.Refused):
            return bound_actions
        data = mapping._sample(state, index)
        t = _host(table._asdict() if hasattr(table, "_asdict") else table)
        if not isinstance(t, dict):
            return refuse("UNRECOGNIZED_KERNEL_TABLE", "MoveTable or an exact host dictionary required")
        moves_count = len(metadata["vocabulary"]["moves"])
        if any(_at(t, 'secondary', 'contact', move) and _at(t, 'phazing', 'move', move)
               for move in range(moves_count)):
            return refuse('CONTACT_PHAZING_OVERLAP',
                'The shared auxiliary stream requires mutually exclusive dispatched moves')
        if any(len(t[name]) != moves_count for name in ("power", "accuracy", "always_hit", "priority")):
            return refuse("TABLE_VOCABULARY_SHAPE", "Move table/vocabulary lengths must agree")
        active = data["active"]
        forced = [data["forced_switch"][s] or data["hp"][s][active[s]] <= 0 for s in range(2)]
        replacement = any(forced)
        live_move = [not replacement and original[s] < 4 for s in range(2)]
        selected = [set() for _ in range(2)]
        participants = [{active[s]} for s in range(2)]
        for s in range(2):
            if original[s] >= 4 and (not replacement or forced[s]):
                bench = [p for p in range(6) if p != active[s]]
                participants[s].add(bench[original[s]-4])
            if live_move[s]:
                move = data["charge_move"][s][active[s]] or data["moves"][s][active[s]][original[s]]
                if bound_actions.choices[s] == "struggle":
                    move = t["struggle"]
                selected[s].add(move)

        # Two actions, but either may dispatch a caller/Encore. Monotone closure
        # safely includes all callable slots (zero PP selection still consumes
        # the rank draw) and last-move overrides activated by the earlier foe.
        changed = True
        while changed:
            before = tuple(tuple(sorted(x)) for x in selected)
            for s in range(2):
                p = active[s]
                if any(_at(t, "sleep_talk", "move", move) for move in selected[s]):
                    selected[s].update(move for move in data["moves"][s][p]
                        if _at(t, "sleep_talk", "callable", move))
                if live_move[s] and any(_at(t, "encore", "move", move) for move in selected[1-s]):
                    kind, slot = data['last_action_kind'][s], data['last_action_index'][s]
                    if kind == 2 and 0 <= slot < 4:
                        last = data['moves'][s][p][slot]
                        if (data['pp'][s][p][slot] > 0 and _at(t, 'encore', 'allowed_last', last)):
                            selected[s].add(last)
                if any(_at(t, "phazing", "move", move) for move in selected[1-s]):
                    participants[s].update(p for p in range(6) if data["hp"][s][p] > 0)
            changed = before != tuple(tuple(sorted(x)) for x in selected)

        # A reachable Trace entrant may gain the opposing current ability.
        # Close both seats together before declaring an existing chance axis
        # inactive. This is a conservative source closure, not a sampled path.
        abilities=[{data["ability"][s][p] for p in participants[s]} for s in range(2)]
        trace_table=(t.get('entry') or {}).get('trace')
        changed=True
        while changed:
            before=tuple(frozenset(values) for values in abilities)
            for s in range(2):
                if trace_table is not None and any(trace_table[ability] for ability in abilities[s]):
                    abilities[s].update(abilities[1-s])
            changed=before!=tuple(frozenset(values) for values in abilities)
        statuses = [{data["status"][s][p] for p in participants[s]} for s in range(2)]
        for s in range(2):
            if (any(_at(t, 'secondary', 'contact', move) for move in selected[s])
                    and any(_at(t, 'secondary', 'static', ability)
                            for ability in abilities[1-s])):
                statuses[s].add(1)  # conservative own-contact paralysis closure
            for move in selected[1-s]:
                statuses[s].add(_at(t, "effects", "status", move))
                statuses[s].add(_at(t, "secondary", "status", move))
        # Fresh status from either a move or a contact ability may reflect onto
        # the other seat before its queued action or residual ordering. Use the
        # actual table and Trace-closed current abilities, never base ability.
        changed=True
        while changed:
            before=tuple(frozenset(values) for values in statuses)
            for s in range(2):
                if any(_at(t,'secondary','synchronize',a) for a in abilities[s]):
                    statuses[1-s].update(3 if value==4 else value
                        for value in tuple(statuses[s]) if value in (1,2,3,4))
            changed=before!=tuple(frozenset(values) for values in statuses)
        weather = {data["weather"], 0}  # suppression may disappear after a KO
        for s in range(2):
            weather.update(_at(t, "weather", "move_weather", move) for move in selected[s])
            weather.update(_at(t, "entry", "weather_ability", ability) for ability in abilities[s])
        haze = any(_at(t, "effects", "haze", move) for choices in selected for move in choices)
        stages = [[{data["boosts"][s][p][field] for p in participants[s]} for field in (5, 6)] for s in range(2)]
        for s in range(2):
            for position, field in enumerate((5, 6)):
                if haze or len(participants[s]) > 1:
                    stages[s][position].add(0)
                delta = {_at(t, "effects", "self_boosts", move, [0]*7)[field] for move in selected[s]}
                # Include both orderings of Haze then self boost and vice versa.
                stages[s][position].update(max(-6, min(6, value+change))
                    for value in tuple(stages[s][position]) for change in delta)

        observations = {name: [] for name, _ in COORDINATES}
        reasons = {}

        def use(name, *obs):
            observations[name].extend(obs)

        def inactive(name, reason):
            reasons[name] = reason

        for s in range(2):
            prefix = f"draws.{s}."
            # Earlier opposing status/secondary may activate the later gate.
            if live_move[s] and 6 in statuses[s]:
                use(prefix+"1", Observation("less_than", 20))
            if live_move[s] and 1 in statuses[s]:
                use(prefix+"2", Observation("less_than", 25))
            if live_move[s] and any(data["confusion_counter"][s][p] for p in participants[s]):
                use(prefix+"3", Observation("less_than", 50))
            random_damage = any(t["power"][move] > 0 or _at(t, "variable_moves", "hidden_power", move)
                or _at(t, "variable_moves", "returns", move) or _at(t, "variable_moves", "frustration", move)
                or _at(t, "variable_moves", "hp_dependent", move)
                for move in selected[s])
            random_damage |= bool(observations[prefix+"3"])
            if random_damage:
                use(prefix+"4", Observation("less_than", 1), Observation("less_than", 2))
                use(prefix+"5", Observation("identity"))
            accuracy = set()
            reachable_hustle = any(_at(t, "combat", "hustle", ability) for ability in abilities[s])
            compound=(t.get('combat') or {}).get('compound_eyes')
            sand=(t.get('weather') or {}).get('sand_veil')
            for move in selected[s]:
                if (not t['always_hit'][move]
                        and (type(t['accuracy'][move]) is not int or not 0 <= t['accuracy'][move] <= 100)):
                    return refuse('ACCURACY_DOMAIN',
                        'Actual numeric kernel accuracy must be an integer in0..100 before partitioning')
                for climate in weather:
                    always = t["always_hit"][move]
                    value = t["accuracy"][move]
                    if _at(t, "weather", "thunder", move):
                        always |= climate == 1
                        value = 50 if climate == 2 else value
                    if always:
                        continue
                    for a in stages[s][0]:
                        for e in stages[1-s][1]:
                            types = {None}
                            if reachable_hustle:
                                types = {t["type"][move]}
                                if _at(t, "variable_moves", "hidden_power", move):
                                    types = {data["hidden_power_type"][s][p]
                                        or t["variable_moves"]["dark_type"] for p in participants[s]}
                            # Union only reachable ability/type/climate handlers.
                            # Trace entrants and weather suppression are already
                            # conservatively closed above, without a rollout.
                            for ability in abilities[s]:
                                for target in abilities[1-s]:
                                    for typ in types:
                                        hustle=bool(typ is not None and _at(t,'combat','hustle',ability)
                                            and _at(t,'combat','physical_type',typ))
                                        accuracy.add(Fraction(mechanics.accuracy_count(value,a,e,
                                            hustle=hustle,compound_eyes=bool(compound is not None and compound[ability]),
                                            sand_veil=bool(climate==3 and sand is not None and sand[target]))))
            use(prefix+"6", *(Observation("less_than", value) for value in sorted(accuracy)))
            if any(_at(t, "effects", "status", move) == 5 or _at(t, "encore", "move", move) for move in selected[s]):
                use(prefix+"7", Observation("identity"))
            if any(_at(t, "effects", "protect", move) or _at(t, "effects", "endure", move) for move in selected[s]):
                # Input mapper admits only denominators0/2/4/8. Haze does not
                # change stall; no within-turn actor can increment it twice.
                denominator = data["protect_counter"][s][active[s]]
                use(prefix+"8", Observation("less_than", 8//max(1, denominator)))
            if any(_at(t, "phazing", "move", move) for move in selected[s]):
                use(f"phazing.{s}", *(Observation("modulo", count) for count in range(1, 6)))
            if (any(_at(t, 'secondary', 'contact', move) for move in selected[s])
                    and any(_at(t, 'secondary', 'static', ability)
                            for ability in abilities[1-s])):
                use(f'phazing.{s}', Observation('less_than', 20))
            for move in selected[s]:
                chance = _at(t, "secondary", "chance", move)
                if chance:
                    # A reachable entrant/copy cannot produce a multiplier
                    # other than1 or2 in this bound source; union both safely.
                    use(f"secondary.{s}", Observation("less_than", chance), Observation("less_than", 2*chance))
            if any(_at(t, "sleep_talk", "move", move) for move in selected[s]):
                for p in participants[s]:
                    count = sum(bool(_at(t, "sleep_talk", "callable", move)) for move in data["moves"][s][p])
                    use(f"sleep_talk.{s}", Observation("modulo", max(1, count)))

        # Binary physical coins stay binary unless all uses are immaterial by
        # source/domain proof. Do not infer inactivity from one rollout/order.
        if all(live_move):
            use("draws.0.0", Observation("identity"))
        both_switch = all(original[s] >= 4 and (not replacement or forced[s]) for s in range(2))
        if both_switch:
            use("draws.0.11", Observation("identity"))
            if replacement and t.get("entry") is not None:
                use("draws.1.10", Observation("identity"))
        seed = [any(data["volatile"][s][p] & 4 for p in participants[s]) or
                any(_at(t, "sustain", "seed", move) for move in selected[1-s]) for s in range(2)]
        if not replacement and all(seed):
            use("draws.0.10", Observation("identity"))
        if not replacement and all(statuses[s] & {2, 3, 4} for s in range(2)):
            use("draws.1.9", Observation("identity"))
        inactive("draws.1.0", "SpeedBoost paired callbacks change only own stage; neither can KO or alter the frozen residual ordering")
        inactive("draws.0.9", "Selected item callbacks heal or boost only their own live holder; neither can KO, resurrect, or alter the frozen residual ordering or the other's callback")
        inactive("draws.1.11", "Encore residual callbacks change only own counter/lock and cannot KO or alter the other's callback")
        inactive("initial_tie", "Full input mapper refuses unresolved entry_pending; initialize has no eligible initial callback")
        for s in range(2):
            inactive(f"entry_ties.{s}", "Full input mapper refuses pending_residual; resuming is false so saved/replacement tie stream is not read")

        axes = []
        for name, _ in COORDINATES:
            obs = tuple(dict.fromkeys(observations[name]))
            if obs:
                axes.append(Axis(name, obs, source_uses=CONSUMERS[name]))
            else:
                reason = reasons.get(name, "Conservative selected/caller/participant closure contains no live consumer for this coordinate; ordinary source gates exclude it")
                axes.append(Axis(name, (), inactive_reason=reason))
        law = FinitePlan(tuple(axes))
        if law.cardinality > cap:
            return refuse("CHANCE_CARDINALITY_CAP", f"required={law.cardinality}; cap={cap}; zero cases emitted")
        return ConsumerPlan(law, native, bound_actions,
            tuple(tuple(sorted(x)) for x in selected), tuple(tuple(sorted(x)) for x in participants),
            (*actual, ("transitive_kernel_domain", domain)), mapping._freeze(t), metadata, cap)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError) as error:
        return refuse("INVALID_CONSUMER_DOMAIN", str(error))
