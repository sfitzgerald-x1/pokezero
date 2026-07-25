"""Command-line entry point for the local endgame scenario studio."""

from __future__ import annotations

import argparse
from pathlib import Path
import webbrowser

from ..local_showdown import DEFAULT_SHOWDOWN_ROOT
from .server import ScenarioStudioHTTPServer
from .service import ScenarioStudioService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local PokeZero endgame scenario studio.")
    parser.add_argument("--showdown-root", type=Path, default=DEFAULT_SHOWDOWN_ROOT)
    parser.add_argument("--scenario-dir", type=Path, default=Path("scenarios/endgame"))
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 chooses a free port.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening it.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be in 0..65535.")
    service = ScenarioStudioService(
        showdown_root=args.showdown_root,
        scenario_dir=args.scenario_dir,
    )
    with ScenarioStudioHTTPServer(("127.0.0.1", args.port), service) as server:
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/"
        print(f"PokeZero scenario studio: {url}")
        print(f"Pinned Gen 3 randbats source: {service.catalog.source_hash}")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nScenario studio stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
