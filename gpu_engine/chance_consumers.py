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
        (6, "mechanics.accuracy_hit: joint rational predicates"),
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
    "transition.py": "91c7fef87cec911e5d7acd6b5d370a6b5c44093b3480ef1cfb51045a3978db87",
    "mechanics.py": "704b951ef2df15004e49601a90b934def7bbf63d2f884d73702316eea7e0ea31",
    "pursuit.py": "a7287b041b219ad790de874684bf0578100084a95dd8f6aff57f2b53916c442e",
    "effects.py": "4df3dcc232bbe27003f6808a8d788591e005fe5f7f149248594cfaf4dc91eb9a",
    "critical.py": "b66e6cf10c0d4df625b1c4f5e5db238af42052e556d7649a10f911745181401e",
    "secondary.py": "1cdb43d30bd157e176a6cdb6ee5f108e884ae8b3c0ac70dca4c6af3ddfa413ea",
    "sleep_talk.py": "5af8fc87ec092ff662ebb0958390b60cc711795e85a3f4b28188be8c004faa91",
    "encore.py": "c682e851f1ea7f10afb0b7502e10a98a3806bd72a2dacf5dc908133492458793",
    "phazing.py": "53ebef4c2202c0dda8227befc959c609e1d8c4ab7a9ede9e8ed8abeadbd0a402",
    "weather.py": "a5014e990f08bfb0cf04ac4aba5bc4cbc44717071601f80d3c340bfac79cade5",
    "state.py": "c14656f1f5c121ae78c8b57f98797f0da560a02ffe66105edb14ad3f3f7bb86e",
    "entry.py": "c717563781e9d4e665071be544dfe921414f0445c7412b58a3d30e58a7c76d03",
}
DOMAIN_MODULES = tuple("absorption baton combat critical damage_moves effects encore entry focus_punch items linked_traps mechanics phazing pressure pursuit secondary self_destruct sleep_talk solar_beam state sustain transition variable_moves weather".split())
DOMAIN_SHA256 = "4104b2a577e7a22423457f476beaf2f9952171ca5db292a2c20cf31992004611"


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


def plan_for_state(state, index, actions, table, metadata, *, cap):
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
        native = mapping.native_input_from_jax(state, index, metadata)
        if isinstance(native, mapping.Refused):
            return native
        bound_actions = mapping.native_actions_from_jax(state, index, actions, native)
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
                    last = data["last_move"][s][p]
                    if last in data["moves"][s][p] and _at(t, "encore", "allowed_last", last):
                        selected[s].add(last)
                if any(_at(t, "phazing", "move", move) for move in selected[1-s]):
                    participants[s].update(p for p in range(6) if data["hp"][s][p] > 0)
            changed = before != tuple(tuple(sorted(x)) for x in selected)

        statuses = [{data["status"][s][p] for p in participants[s]} for s in range(2)]
        for s in range(2):
            if (any(_at(t, 'secondary', 'contact', move) for move in selected[s])
                    and any(_at(t, 'secondary', 'static', data['ability'][1-s][p])
                            for p in participants[1-s])):
                statuses[s].add(1)  # conservative own-contact paralysis closure
            for move in selected[1-s]:
                statuses[s].add(_at(t, "effects", "status", move))
                statuses[s].add(_at(t, "secondary", "status", move))
        weather = {data["weather"], 0}  # suppression may disappear after a KO
        for s in range(2):
            weather.update(_at(t, "weather", "move_weather", move) for move in selected[s])
            weather.update(_at(t, "entry", "weather_ability", data["ability"][s][p]) for p in participants[s])
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
            for move in selected[s]:
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
                            numerator = (3+a if a >= 0 else 3) * (3 if e >= 0 else 3-e)
                            denominator = (3 if a >= 0 else 3-a) * (3+e if e >= 0 else 3)
                            accuracy.add(Fraction(value*numerator, denominator))
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
                    and any(_at(t, 'secondary', 'static', data['ability'][1-s][p])
                            for p in participants[1-s])):
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
