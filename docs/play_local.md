# Playing a checkpoint locally (browser)

Play any PokeZero checkpoint against yourself in a browser, **fully local** — nothing touches
`play.pokemonshowdown.com`.

## One command
```sh
POKEZERO_SHOWDOWN_ROOT=/path/to/pokemon-showdown scripts/play_local.sh
# -> open http://localhost:8080 , challenge PokeZeroBot-500k or PokeZeroBot-1M to [Gen 3] Random Battle
```
Custom variants (one `PokeZeroBot-<name>` per `name=checkpoint`):
```sh
scripts/play_local.sh 1m=checkpoints/pokezero-gen3-1m.pt mine=runs/<...>/transformer-policy.pt
```

## What it runs (and the gotcha it solves)
Three local pieces:
1. **Sim server** (`pokemon-showdown`, `--no-security`) on `:8000` — the websocket + battle engine.
2. **One bot per checkpoint** (`scripts/play_online.py`, `--accept`) that auto-accepts gen3randombattle challenges.
3. **Web client** (`pokemon-showdown-client`) served on `:8080`, pointed at the local sim.

**Why a separate client:** the sim checkout is *server-only*. Bare `http://localhost:8000` just hangs
at "Loading client…" because it tries to fetch the client from the production CDN. The hosted
`testclient.html` is also gone (404). So we serve the real client locally and connect it to the local
sim via `testclient-new.html?~~localhost:8000`. The script writes a redirect `index.html` so the clean
URL `http://localhost:8080` drops you straight in (`http://pokezero.localhost:8080` works too).

## Requirements
- `node`, `python3`
- A local **pokemon-showdown** (sim) checkout → `POKEZERO_SHOWDOWN_ROOT`
- A local **pokemon-showdown-client** checkout → `POKEZERO_SHOWDOWN_CLIENT`
  (defaults to `<sim>/../pokemon-showdown-client`)
- The repo `.venv` with `pip install -e '.[neural]'` (or set `POKEZERO_PYTHON`)

These Showdown checkouts live outside this repo; point the env vars at yours. `Ctrl-C` stops
everything (server, bots, client).
