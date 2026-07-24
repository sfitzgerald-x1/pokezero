"""Loopback-only HTTP API and static-file host for the scenario studio."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from typing import Any
from urllib.parse import unquote, urlparse

from .domain import ScenarioValidationError
from .service import ScenarioStudioService


_STATIC_ROOT = Path(__file__).with_name("static")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class ScenarioStudioHTTPServer(ThreadingHTTPServer):
    """Expose one studio service only on an explicitly selected local address."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: ScenarioStudioService) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("Scenario studio must bind to a loopback address.")
        super().__init__(address, ScenarioStudioRequestHandler)
        self.service = service


class ScenarioStudioRequestHandler(BaseHTTPRequestHandler):
    server: ScenarioStudioHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in _STATIC_FILES:
            filename, content_type = _STATIC_FILES[path]
            self._write_static(filename, content_type)
            return
        if path == "/api/catalog":
            self._json(HTTPStatus.OK, self.server.service.catalog_payload())
            return
        if path == "/api/scenarios":
            self._json(HTTPStatus.OK, {"scenarios": self.server.service.list()})
            return
        slug = _scenario_slug_from_path(path)
        if slug is not None:
            self._run(lambda: self.server.service.load(slug))
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint.")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload = self._request_json()
        if payload is None:
            return
        if path == "/api/teams/generate":
            seed = payload.get("seed") if isinstance(payload, dict) else None
            if seed is None:
                seed = secrets.randbelow(2**31)
            if isinstance(seed, bool) or not isinstance(seed, int):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "seed must be an integer.")
                return
            self._run(lambda: {"seed": seed, "side": self.server.service.generate_team(seed=seed)})
            return
        if path == "/api/validate":
            self._run(lambda: self.server.service.validate_payload(_scenario_payload(payload)))
            return
        if path == "/api/evaluate-root":
            if not isinstance(payload, dict) or not isinstance(payload.get("checkpoint_path"), str):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "checkpoint_path must be a string.")
                return
            self._run(
                lambda: self.server.service.evaluate_root(
                    _scenario_payload(payload),
                    checkpoint_path=payload["checkpoint_path"],
                    device=payload.get("device") if isinstance(payload.get("device"), str) else "cpu",
                )
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint.")

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        slug = _scenario_slug_from_path(path)
        if slug is None:
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Unknown endpoint.")
            return
        payload = self._request_json()
        if payload is None:
            return
        self._run(lambda: self.server.service.save(slug, _scenario_payload(payload)))

    def _request_json(self) -> Any | None:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "Content-Length must be an integer.")
            return None
        if length <= 0 or length > 2_000_000:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "JSON body must be between 1 and 2,000,000 bytes.")
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "Request body must be valid UTF-8 JSON.")
            return None

    def _run(self, operation) -> None:
        try:
            result = operation()
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, "not_found", str(exc))
        except ScenarioValidationError as exc:
            self._error(HTTPStatus.BAD_REQUEST, "validation_error", str(exc), path=exc.path)
        except (OSError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        except Exception:
            # The browser receives a stable error shape without Python paths or tracebacks. The
            # local process keeps its normal concise server log rather than reflecting internals.
            self.log_error("Unhandled scenario studio request failure")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "Scenario processing failed.")
        else:
            self._json(HTTPStatus.OK, result)

    def _write_static(self, filename: str, content_type: str) -> None:
        path = (_STATIC_ROOT / filename).resolve()
        if path.parent != _STATIC_ROOT or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not_found", "Static file not found.")
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _error(self, status: HTTPStatus, code: str, message: str, *, path: str | None = None) -> None:
        payload: dict[str, Any] = {"error": {"code": code, "message": message}}
        if path:
            payload["error"]["path"] = path
        self._json(status, payload)

    def log_message(self, format: str, *args: object) -> None:
        # The tool is local and chatty validation calls should not drown out CLI output.
        return


def _scenario_slug_from_path(path: str) -> str | None:
    prefix = "/api/scenarios/"
    if not path.startswith(prefix):
        return None
    suffix = unquote(path[len(prefix) :])
    return suffix if suffix and "/" not in suffix else None


def _scenario_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "scenario" in payload:
        return payload["scenario"]
    return payload
