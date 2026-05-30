"""Read-only Gen 3 randbat belief sidecar webview."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlparse

from .belief import PublicBattleBeliefEngine
from .randbat import Gen3RandbatSource
from .showdown import parse_showdown_replay


DEFAULT_SHOWDOWN_WEBSOCKET_URL = "ws://localhost:8000/showdown/websocket"
DEFAULT_SIDECAR_PORT = 8010


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class BeliefSidecarState:
    def __init__(
        self,
        *,
        room_id: str,
        set_source: Gen3RandbatSource,
        format_id: str = "gen3randombattle",
        perspective: str = "both",
    ) -> None:
        self.room_id = room_id
        self.format_id = format_id
        self.set_source = set_source
        self.perspective = perspective
        self.connection_state = "starting"
        self.connection_detail = ""
        self._lines: list[str] = []
        self._lock = threading.RLock()

    def set_connection_state(self, state: str, detail: str = "") -> None:
        with self._lock:
            self.connection_state = state
            self.connection_detail = detail

    def ingest_lines(self, lines: Iterable[str]) -> bool:
        public_lines = [line.strip() for line in lines if _is_public_battle_line(line)]
        if not public_lines:
            return False
        with self._lock:
            self._lines.extend(public_lines)
        return True

    def payload(self) -> dict[str, Any]:
        with self._lock:
            lines = list(self._lines)
            connection_state = self.connection_state
            connection_detail = self.connection_detail
            perspective = self.perspective
        replay = parse_showdown_replay(lines, battle_id=self.room_id)
        engine = PublicBattleBeliefEngine.from_events(
            replay.public_events,
            format_id=self.format_id,
            set_source=self.set_source,
        )
        engine.resolve_pending_switches_at_boundary()
        snapshot = engine.snapshot()
        payload: dict[str, Any] = {
            "room_id": self.room_id,
            "format_id": self.format_id,
            "connection": {
                "state": connection_state,
                "detail": connection_detail,
            },
            "perspective": perspective,
            "players": dict(replay.players),
            "winner": replay.winner,
            "source": self.set_source.metadata.to_payload(),
            "battle": snapshot.to_overlay_payload(),
            "recent_public_lines": lines[-80:],
        }
        if perspective in {"p1", "p2"}:
            payload["player_view"] = snapshot.for_player(perspective).to_overlay_payload()
        else:
            payload["player_views"] = {
                "p1": snapshot.for_player("p1").to_overlay_payload(),
                "p2": snapshot.for_player("p2").to_overlay_payload(),
            }
        return payload


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item is not subscriber]

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(payload)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(payload)
                except queue.Full:
                    pass


def serve_sidecar(
    *,
    state: BeliefSidecarState,
    host: str,
    port: int,
    showdown_url: Optional[str],
    username: str,
) -> None:
    broker = EventBroker()
    if showdown_url:
        thread = threading.Thread(
            target=_run_showdown_listener,
            kwargs={
                "state": state,
                "broker": broker,
                "showdown_url": showdown_url,
                "username": username,
            },
            daemon=True,
        )
        thread.start()
    handler = _handler_factory(state, broker)
    server = ReusableThreadingHTTPServer((host, port), handler)
    print(f"PokeZero belief sidecar: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PokeZero belief sidecar.")


def _run_showdown_listener(
    *,
    state: BeliefSidecarState,
    broker: EventBroker,
    showdown_url: str,
    username: str,
) -> None:
    try:
        from websockets.sync.client import connect as websocket_connect
    except ImportError:
        state.set_connection_state("error", "The `websockets` package is required for live Showdown attachment.")
        broker.publish(state.payload())
        return

    while True:
        try:
            state.set_connection_state("connecting", showdown_url)
            broker.publish(state.payload())
            with websocket_connect(showdown_url, open_timeout=10, close_timeout=5) as connection:
                state.set_connection_state("connected", showdown_url)
                broker.publish(state.payload())
                connection.send(f"|/trn {username},0,")
                connection.send(f"|/join {state.room_id}")
                while True:
                    payload = connection.recv(timeout=30)
                    if not isinstance(payload, str):
                        continue
                    for room_id, lines in _split_protocol_message(payload):
                        _handle_global_auth(connection, lines, username)
                        if room_id == state.room_id and state.ingest_lines(lines):
                            broker.publish(state.payload())
        except Exception as error:  # pragma: no cover - exercised manually against local Showdown.
            state.set_connection_state("reconnecting", str(error))
            broker.publish(state.payload())
            time.sleep(2)


def _handle_global_auth(connection: Any, lines: Iterable[str], username: str) -> None:
    for line in lines:
        if line.startswith("|challstr|"):
            connection.send(f"|/trn {username},0,")


def _handler_factory(state: BeliefSidecarState, broker: EventBroker) -> type[BaseHTTPRequestHandler]:
    class SidecarHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(render_index_html(), content_type="text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                perspective = query.get("perspective", [None])[0]
                if perspective in {"p1", "p2", "both"}:
                    state.perspective = perspective
                self._send_json(state.payload())
                return
            if parsed.path == "/api/events":
                self._send_sse()
                return
            if parsed.path == "/healthz":
                self._send_text("ok\n", content_type="text/plain; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, *, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self) -> None:
            subscriber = broker.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(_sse_event("state", state.payload()))
                self.wfile.flush()
                while True:
                    try:
                        payload = subscriber.get(timeout=15)
                        self.wfile.write(_sse_event("state", payload))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                broker.unsubscribe(subscriber)

    return SidecarHandler


def render_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PokeZero Belief Sidecar</title>
  <style>
    :root {
      --bg: #101510;
      --panel: #172017;
      --panel-2: #1f2a1f;
      --text: #eff6e9;
      --muted: #aeb9a8;
      --accent: #d7ff5f;
      --line: #344033;
      --danger: #ff9b73;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, #24351e, var(--bg) 48rem);
      color: var(--text);
      font: 15px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--line);
      background: rgba(16, 21, 16, 0.92);
      backdrop-filter: blur(10px);
    }
    h1, h2, h3 { margin: 0; }
    h1 { font-size: 1.05rem; letter-spacing: 0.04em; text-transform: uppercase; }
    main { display: grid; grid-template-columns: 1fr 22rem; gap: 1rem; padding: 1rem; }
    section, aside, .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: color-mix(in srgb, var(--panel) 88%, transparent);
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.2);
    }
    section, aside { padding: 1rem; }
    .toolbar { display: flex; align-items: center; gap: 0.75rem; color: var(--muted); }
    select {
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 0.35rem 0.5rem;
    }
    .status { color: var(--accent); }
    .status.error, .status.reconnecting { color: var(--danger); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 1rem; }
    .card { padding: 0.9rem; margin-top: 0.8rem; }
    .card.active { border-color: var(--accent); }
    .muted { color: var(--muted); }
    .pill {
      display: inline-flex;
      margin: 0.15rem;
      padding: 0.15rem 0.45rem;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #101810;
      color: var(--muted);
      font-size: 0.78rem;
    }
    details { margin-top: 0.7rem; }
    summary { cursor: pointer; color: var(--accent); }
    pre {
      overflow: auto;
      max-height: 34rem;
      padding: 0.75rem;
      border-radius: 10px;
      background: #080d08;
      color: #dce8d2;
      white-space: pre-wrap;
    }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>PokeZero Belief Sidecar</h1>
      <div id="subtitle" class="muted">Waiting for battle state...</div>
    </div>
    <div class="toolbar">
      <span id="status" class="status">starting</span>
      <label>Perspective
        <select id="perspective">
          <option value="both">Both</option>
          <option value="p1">p1</option>
          <option value="p2">p2</option>
        </select>
      </label>
    </div>
  </header>
  <main>
    <section>
      <h2>Belief State</h2>
      <div id="belief" class="grid"></div>
    </section>
    <aside>
      <h2>Source</h2>
      <div id="source" class="muted"></div>
      <details open>
        <summary>Recent protocol</summary>
        <pre id="raw"></pre>
      </details>
    </aside>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const subtitleEl = document.getElementById('subtitle');
    const beliefEl = document.getElementById('belief');
    const sourceEl = document.getElementById('source');
    const rawEl = document.getElementById('raw');
    const perspectiveEl = document.getElementById('perspective');

    perspectiveEl.addEventListener('change', () => refresh());

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    function pills(values) {
      if (!values || !values.length) return '<span class="muted">unknown</span>';
      return values.map(value => `<span class="pill">${esc(value)}</span>`).join('');
    }

    function renderPokemon(pokemon) {
      const variants = pokemon.candidate_variants || [];
      const evidence = pokemon.evidence || [];
      return `<article class="card ${pokemon.active ? 'active' : ''}">
        <h3>${esc(pokemon.showdown_slot)} · ${esc(pokemon.species)} ${pokemon.active ? '<span class="pill">active</span>' : ''}</h3>
        <p class="muted">HP/status: ${esc(pokemon.condition || 'unknown')} · candidates: ${esc(pokemon.candidate_set_count ?? 'unknown')} · uncertainty: ${Number(pokemon.uncertainty ?? 1).toFixed(2)}</p>
        <div><strong>Revealed moves</strong><br>${pills(pokemon.revealed_moves)}</div>
        <div><strong>Possible abilities</strong><br>${pills(pokemon.possible_abilities)}</div>
        <div><strong>Possible items</strong><br>${pills(pokemon.possible_items)}</div>
        <div><strong>Possible moves</strong><br>${pills(pokemon.possible_moves)}</div>
        <details>
          <summary>Surviving variants (${variants.length})</summary>
          <pre>${esc(JSON.stringify(variants, null, 2))}</pre>
        </details>
        <details ${evidence.length ? 'open' : ''}>
          <summary>Evidence (${evidence.length})</summary>
          <pre>${esc(JSON.stringify(evidence, null, 2))}</pre>
        </details>
      </article>`;
    }

    function render(payload) {
      statusEl.textContent = payload.connection?.state || 'unknown';
      statusEl.className = `status ${payload.connection?.state || ''}`;
      subtitleEl.textContent = `${payload.room_id} · ${payload.format_id} · ${payload.connection?.detail || ''}`;
      perspectiveEl.value = payload.perspective || 'both';
      const sides = payload.battle?.sides || {};
      beliefEl.innerHTML = Object.entries(sides)
        .map(([slot, mons]) => `<div><h3>${esc(slot)} ${esc(payload.players?.[slot] || '')}</h3>${mons.map(renderPokemon).join('')}</div>`)
        .join('');
      sourceEl.innerHTML = `<p>Hash: <strong>${esc(payload.source?.source_hash || 'unknown')}</strong></p><p>${esc(payload.source?.showdown_root || '')}</p>`;
      rawEl.textContent = (payload.recent_public_lines || []).join('\\n');
    }

    async function refresh() {
      const response = await fetch(`/api/state?perspective=${encodeURIComponent(perspectiveEl.value)}`);
      render(await response.json());
    }

    const events = new EventSource('/api/events');
    events.addEventListener('state', event => render(JSON.parse(event.data)));
    events.onerror = () => { statusEl.textContent = 'reconnecting'; statusEl.className = 'status reconnecting'; };
    refresh();
  </script>
</body>
</html>
"""


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n".encode("utf-8")


def _split_protocol_message(payload: str) -> list[tuple[Optional[str], list[str]]]:
    chunks: list[tuple[Optional[str], list[str]]] = []
    current_room: Optional[str] = None
    current_lines: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith(">"):
            if current_lines or current_room is not None:
                chunks.append((current_room, current_lines))
            current_room = line[1:]
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines or current_room is not None:
        chunks.append((current_room, current_lines))
    return chunks


def _is_public_battle_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    if stripped.startswith("|request|"):
        return False
    return True


def _available_host(host: str, port: int) -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return sock.getsockname()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pokezero.sidecar")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="Serve a read-only belief sidecar for a local Showdown battle.")
    serve.add_argument("--room", required=True, help="Showdown battle room id, e.g. battle-gen3randombattle-123.")
    serve.add_argument("--showdown-root", required=True, type=Path, help="Local built Pokemon Showdown checkout root.")
    serve.add_argument("--showdown-url", default=DEFAULT_SHOWDOWN_WEBSOCKET_URL, help="Showdown websocket URL.")
    serve.add_argument("--host", default="127.0.0.1", help="HTTP bind host for the sidecar webview.")
    serve.add_argument("--port", type=int, default=DEFAULT_SIDECAR_PORT, help="HTTP bind port for the sidecar webview.")
    serve.add_argument("--username", default="PokeZeroSidecar", help="Spectator username for local Showdown attachment.")
    serve.add_argument("--perspective", choices=("both", "p1", "p2"), default="both", help="Initial view perspective.")
    serve.add_argument("--no-cache", action="store_true", help="Disable source-universe cache reads/writes.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "serve":
        _available_host(args.host, args.port)
        source = Gen3RandbatSource.from_showdown_root(args.showdown_root, use_cache=not args.no_cache)
        state = BeliefSidecarState(
            room_id=args.room,
            set_source=source,
            perspective=args.perspective,
        )
        serve_sidecar(
            state=state,
            host=args.host,
            port=args.port,
            showdown_url=args.showdown_url,
            username=args.username,
        )


if __name__ == "__main__":
    main()
