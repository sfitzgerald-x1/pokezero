"""Play a trained PokeZero checkpoint on a live Pokemon Showdown server.

The websocket protocol a live battle streams is the same line format our offline pipeline
already consumes, so the agent reuses it end to end:

    accumulated room protocol lines
        -> parse_showdown_replay        (transport state + the |request| payloads)
        -> normalize_for_player         (our player-relative battle state)
        -> observation_from_player_state(observation tensor)
        -> policy.select_action         (the checkpoint)
        -> showdown_choice_for_action   ("move N" / "switch N")

The only genuinely new piece is the network client: connecting, logging in (challstr ->
assertion -> ``/trn``), routing global vs. room messages, accepting/searching games, and
replying ``ROOMID|/choose ...``. Built to the standard protocol so it works against a local
``pokemon-showdown`` server (for development) and the official server (for play in the wild).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .local_showdown import belief_set_source_env_enabled
from .observation import OBSERVATION_SCHEMA_VERSION_V2_2
from .randbat import load_gen3_randbat_source_cached
from .showdown import (
    DEFAULT_REPLAY_OBSERVATION_SPEC,
    normalize_for_player,
    observation_from_player_state,
    parse_showdown_replay,
    showdown_choice_for_action,
)

logger = logging.getLogger("pokezero.online")

DEFAULT_LOCAL_WS = "ws://localhost:8000/showdown/websocket"
DEFAULT_LOGIN_URL = "https://play.pokemonshowdown.com/action.php"
DEFAULT_FORMAT = "gen3randombattle"


def to_id(value: str) -> str:
    """Showdown user/format id: lowercase alphanumerics only."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


class LoginError(RuntimeError):
    """Raised when the login server refuses to issue an assertion."""


def request_assertion(
    challstr: str,
    username: str,
    password: str | None,
    *,
    login_url: str = DEFAULT_LOGIN_URL,
    timeout: float = 20.0,
) -> str:
    """Exchange a server challstr for a login assertion via the Showdown login server.

    Unregistered names use ``act=getassertion`` (no password); registered names use
    ``act=login`` and the assertion is pulled from the (``]``-prefixed) JSON response.
    """
    if password:
        payload = {"act": "login", "name": username, "pass": password, "challstr": challstr}
    else:
        payload = {"act": "getassertion", "userid": to_id(username), "challstr": challstr}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(login_url, data=data, headers={"User-Agent": "pokezero-online"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (trusted login URL)
        body = response.read().decode("utf-8").strip()
    if password:
        if not body.startswith("]"):
            raise LoginError(f"Unexpected login response: {body[:80]!r}")
        parsed = json.loads(body[1:])
        assertion = parsed.get("assertion")
        if not assertion or str(assertion).startswith(";"):
            raise LoginError(f"Login refused for {username!r}: {str(assertion)[:80]!r}")
        return str(assertion)
    if not body or body.startswith(";"):
        raise LoginError(f"Assertion refused for {username!r}: {body[:80]!r}")
    return body


@dataclass
class OnlineBattleAgent:
    """Wraps a checkpoint policy and turns a battle room's protocol log into a ``/choose`` body."""

    policy: Any
    vocab: Any
    dex: Any
    our_name: str
    spec: Any = DEFAULT_REPLAY_OBSERVATION_SPEC
    # Encode-time feature masks, derived from the checkpoint's stamped provenance in
    # build_agent (never left at defaults for a masked ablation checkpoint — the mask-axis
    # twin of the #492 belief-source train/eval mismatch).
    feature_masks: Any = None
    rng: random.Random = None  # type: ignore[assignment]
    # Candidate-set source for belief views. None defers to the POKEZERO_BELIEF_SET_SOURCE env
    # gate at build time (build_agent) — the online client is the cluster foul-play probes'
    # bot path, so a mismatch here recreates the train/eval observation gap fixed in the
    # controlled bridge (see foulplay_bridge._resolved_belief_set_source).
    set_source: Any = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    def choose(self, room_lines: list[str], room_id: str) -> Optional[str]:
        """Return the choice body (e.g. ``"move 1"``) for the latest request, or None to wait."""
        try:
            replay = parse_showdown_replay(room_lines, battle_id=room_id)
            state = normalize_for_player(
                replay, player_id="bot", player_name=self.our_name, set_source=self.set_source
            )
        except ValueError:
            return None  # our seat / request not resolvable yet
        if state.request is None or state.request_kind in {"wait", "none"}:
            return None
        if not any(state.legal_action_mask):
            return None
        encode_kwargs = {"feature_masks": self.feature_masks} if self.feature_masks is not None else {}
        observation = observation_from_player_state(
            state, category_vocab=self.vocab, spec=self.spec, dex=self.dex, **encode_kwargs
        )
        decision = self.policy.select_action(observation, rng=self.rng)
        return showdown_choice_for_action(state, decision.action_index)


def build_agent(
    checkpoint: str | Path,
    showdown_root: str | Path,
    our_name: str,
    *,
    deterministic: bool = True,
    seed: int | None = None,
) -> OnlineBattleAgent:
    """Load the checkpoint and pair it with the observation spec/vocab/dex it expects."""
    from .dex import load_showdown_dex_cached
    from .neural_policy import load_transformer_policy
    from .randbat_vocab import gen3_category_vocabulary

    from .neural_policy import feature_masks_from_model_config, observation_spec_from_model_config

    policy = load_transformer_policy(checkpoint, deterministic=deterministic)
    config = policy.result.model_config
    # Feed the model the observation schema + shape it was trained on (dual-schema resolution;
    # widths may predate later feature slots within the checkpoint's own schema).
    spec = observation_spec_from_model_config(config)
    return OnlineBattleAgent(
        policy=policy,
        # Vocabulary latches with the schema (review MED-2): v2.2 needs the
        # turn-merged families.
        vocab=gen3_category_vocabulary(
            showdown_root,
            include_turn_merged=(
                spec.schema_version == OBSERVATION_SCHEMA_VERSION_V2_2
            ),
        ),
        dex=load_showdown_dex_cached(showdown_root),
        feature_masks=feature_masks_from_model_config(config),
        set_source=(
            load_gen3_randbat_source_cached(showdown_root)
            if belief_set_source_env_enabled()
            else None
        ),
        our_name=our_name,
        spec=spec,
        rng=random.Random(seed),
    )


def split_server_message(raw: str) -> tuple[str, list[str]]:
    """Split a websocket frame into (room_id, protocol lines). Global frames have room_id ''."""
    lines = raw.split("\n")
    if lines and lines[0].startswith(">"):
        return lines[0][1:].strip(), [line for line in lines[1:] if line]
    return "", [line for line in lines if line]


class ShowdownClient:
    """Minimal async Showdown protocol client that plays battles with an :class:`OnlineBattleAgent`."""

    def __init__(
        self,
        agent: OnlineBattleAgent,
        *,
        username: str,
        password: str | None = None,
        websocket_url: str = DEFAULT_LOCAL_WS,
        login_url: str = DEFAULT_LOGIN_URL,
        battle_format: str = DEFAULT_FORMAT,
        accept_challenges: bool = True,
        search_ladder: bool = False,
        challenge_user: str | None = None,
        max_games: int | None = 1,
        skip_login: bool = False,
    ) -> None:
        self.agent = agent
        self.username = username
        self.password = password
        self.websocket_url = websocket_url
        self.login_url = login_url
        self.skip_login = skip_login
        self.battle_format = battle_format
        self.accept_challenges = accept_challenges
        self.search_ladder = search_ladder
        self.challenge_user = challenge_user
        self.max_games = max_games
        self._room_lines: dict[str, list[str]] = {}
        self._started_matchmaking = False
        self._games_finished = 0
        self._ws: Any = None

    async def _send(self, text: str) -> None:
        logger.debug(">> %s", text)
        await self._ws.send(text)

    async def run(self) -> None:
        import websockets

        async with websockets.connect(self.websocket_url, max_size=None) as ws:
            self._ws = ws
            async for raw in ws:
                room_id, lines = split_server_message(raw)
                for line in lines:
                    await self._handle_line(room_id, line)
                if self.max_games is not None and self._games_finished >= self.max_games:
                    logger.info("Played %d game(s); disconnecting.", self._games_finished)
                    return

    async def _handle_line(self, room_id: str, line: str) -> None:
        logger.debug("<< [%s] %s", room_id or "global", line)
        if not line.startswith("|"):
            return
        parts = line.split("|")
        message_type = parts[1] if len(parts) > 1 else ""

        if message_type == "challstr":
            if self.skip_login:  # local servers started with --no-security accept names unsigned
                await self._send(f"|/trn {self.username},0,")
                return
            challstr = "|".join(parts[2:])
            assertion = await asyncio.to_thread(
                request_assertion, challstr, self.username, self.password, login_url=self.login_url
            )
            await self._send(f"|/trn {self.username},0,{assertion}")
            return

        if message_type == "updateuser":
            named = len(parts) > 3 and parts[3] == "1"
            if named and to_id(parts[2]) == to_id(self.username):
                await self._start_matchmaking()
            return

        if message_type == "updatechallenges":
            await self._handle_challenges(parts[2] if len(parts) > 2 else "{}")
            return

        if message_type == "pm":
            await self._handle_pm_challenge(parts)
            return

        if room_id.startswith("battle-"):
            await self._handle_battle_line(room_id, line, message_type, parts)

    async def _start_matchmaking(self) -> None:
        if self._started_matchmaking:
            return
        self._started_matchmaking = True
        if self.challenge_user:
            logger.info("Challenging %s to %s", self.challenge_user, self.battle_format)
            await self._send(f"|/utm null")
            await self._send(f"|/challenge {self.challenge_user}, {self.battle_format}")
        elif self.search_ladder:
            logger.info("Searching the %s ladder", self.battle_format)
            await self._send(f"|/search {self.battle_format}")
        elif self.accept_challenges:
            logger.info("Waiting for a %s challenge to %s …", self.battle_format, self.username)

    async def _handle_pm_challenge(self, parts: list[str]) -> None:
        # |pm| SENDER| RECEIVER|/challenge FORMAT|... — the local server's challenge notification.
        if not self.accept_challenges or len(parts) < 5:
            return
        sender, receiver, message = parts[2], parts[3], parts[4]
        if not message.startswith("/challenge"):
            return
        fmt = message[len("/challenge "):].strip() if " " in message else ""
        if to_id(receiver) != to_id(self.username) or to_id(fmt) != to_id(self.battle_format):
            return
        logger.info("Accepting challenge from %s", sender.strip())
        await self._send("|/utm null")
        await self._send(f"|/accept {to_id(sender)}")

    async def _handle_challenges(self, payload: str) -> None:
        if not self.accept_challenges:
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        for challenger, fmt in (data.get("challengesFrom") or {}).items():
            if to_id(fmt) == to_id(self.battle_format):
                logger.info("Accepting challenge from %s", challenger)
                await self._send(f"|/utm null")
                await self._send(f"|/accept {challenger}")

    async def _handle_battle_line(self, room_id: str, line: str, message_type: str, parts: list[str]) -> None:
        buffer = self._room_lines.setdefault(room_id, [])
        buffer.append(line)

        if message_type == "request":
            choice = self.agent.choose(buffer, room_id)
            if choice is not None:
                await self._send(f"{room_id}|/choose {choice}")
        elif message_type in {"win", "tie"}:
            winner = parts[2] if message_type == "win" and len(parts) > 2 else None
            if message_type == "tie":
                logger.info("Battle %s ended in a tie.", room_id)
            elif to_id(winner or "") == to_id(self.username):
                logger.info("Battle %s — we won!", room_id)
            else:
                logger.info("Battle %s — %s won.", room_id, winner)
            self._games_finished += 1
            self._room_lines.pop(room_id, None)
            reset = getattr(self.agent.policy, "reset", None)
            if callable(reset):
                reset()  # clear the policy's per-battle observation history before the next game
            await self._send(f"{room_id}|/leave")


async def run_online(
    *,
    checkpoint: str | Path,
    showdown_root: str | Path,
    username: str,
    password: str | None = None,
    websocket_url: str = DEFAULT_LOCAL_WS,
    login_url: str = DEFAULT_LOGIN_URL,
    battle_format: str = DEFAULT_FORMAT,
    accept_challenges: bool = True,
    search_ladder: bool = False,
    challenge_user: str | None = None,
    max_games: int | None = 1,
    deterministic: bool = True,
    seed: int | None = None,
    skip_login: bool = False,
) -> None:
    agent = build_agent(checkpoint, showdown_root, username, deterministic=deterministic, seed=seed)
    client = ShowdownClient(
        agent,
        username=username,
        password=password,
        websocket_url=websocket_url,
        login_url=login_url,
        battle_format=battle_format,
        accept_challenges=accept_challenges,
        search_ladder=search_ladder,
        challenge_user=challenge_user,
        max_games=max_games,
        skip_login=skip_login,
    )
    await client.run()
