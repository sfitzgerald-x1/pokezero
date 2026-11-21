"""Sampled owning Showdown H2H for the finite transfer assay."""

from .v5_python_collection import Game, OwningEnv, _terminal_winner


def evaluate_game(config, cell, *, factories, packed_start, check_start, max_requests):
    from gpu_engine.v5_evaluation_sampling import EvaluationDraw
    from gpu_engine.v5_live_protocol import check_factory

    if len(factories) != 2 or factories[0] is factories[1]:
        raise ValueError("independently loaded physical-seat factories required")
    for factory in factories:
        check_factory(factory)
        if factory.deterministic or factory.compute_dtype != "bfloat16":
            raise ValueError("matched H2H uses stochastic bf16 v5 forwards")
    if factories[0].config != factories[1].config:
        raise ValueError("same explicit v5 model configuration required")
    game = Game(cell["cluster"], f'{cell["role"]}-{cell["cluster"]}-{cell["seat"]}',
                cell["chance"], cell["action"])
    actors = []
    frames = 0
    try:
        env = OwningEnv(config)
        try:
            env.reset_with_start_override(seed=game.seed, start_override=packed_start)
            birth_receipt = check_start(env.snapshot().bridge_snapshot)
            for side, factory in enumerate(factories):
                actor = factory.create_actor()
                actor.begin_game(battle_id=game.battle_id, owner=f"p{side + 1}",
                                 seed=game.seed, stream_id=game.stream)
                actors.append(actor)
            offsets = [0, 0]
            while frames < max_requests and env.terminal() is None:
                for side, actor in enumerate(actors):
                    lines = env.owner_lines[actor.owner]
                    if len(lines) != offsets[side]:
                        actor.accept_owner_stream(
                            battle_id=game.battle_id, owner=actor.owner, generation=actor.generation,
                            prefix_sha256=actor.prefix_sha256, lines=tuple(lines[offsets[side]:]),
                        )
                        offsets[side] = len(lines)
                actions = {}
                choices = {}
                for player in env.requested_players():
                    side = int(player[1]) - 1
                    actor = actors[side]
                    boundary = actor.snapshot_boundary()
                    draw = EvaluationDraw(game.sampling_seed, game.stream, frames, side)
                    actions[player] = actor.decide(boundary, rng=draw).action_index
                    choices[player] = actor.choice(boundary)
                if not actions or env._choices_for_actions(actions) != choices:
                    raise ValueError("Showdown canonical action routing changed")
                env.step(actions)
                frames += 1
            terminal = env.terminal()
            if terminal is None or terminal.capped:
                return "CAPPED", dict(frames=frames, reason="request-cap", birth=birth_receipt)
            lines = [tuple(line for line in env.owner_lines[seat]
                           if line.startswith(("|win|", "|tie|")) or line == "|tie")
                     for seat in ("p1", "p2")]
            if len(lines[0]) != 1 or lines[0] != lines[1]:
                raise ValueError("owning terminal witnesses disagree")
            marker = lines[0][0]
            _terminal_winner(env.owner_lines, marker, terminal)
            players = [next(line.split("|")[3] for line in env.owner_lines["p1"]
                            if line.startswith(f"|player|p{i}|")) for i in (1, 2)]
            for actor in actors:
                if actor.close_game()["status"] == "FAILED_CLOSED":
                    raise ValueError("owning actor failed to close cleanly")
            actors.clear()
            return "COMPLETE", dict(marker=marker, players=players, frames=frames, birth=birth_receipt)
        finally:
            try:
                for actor in actors:
                    actor.close_game()
            finally:
                env.close()
    except PermissionError:
        raise
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        status = "REFUSED" if isinstance(error, ValueError) else "FAILED"
        return status, dict(frames=frames, reason=f"{type(error).__name__}: {error}")
