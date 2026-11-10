"""The flag-off golden of the chance-seed change: live search responses and re-priced rows on the CPU fakes, written once from
the tree before the change (c109669b) and compared bit for bit by tests/test_depthn_chance_seed.py.

    PYTHONPATH=src:. python -m tests.depthn_chance_golden tests/data/depthn_chance_off_golden.json

Every stage clock is pinned (perf_counter 0, the search's arrival clock 0), so whole responses compare; in the re-priced rows the
fields that carry a wall clock or the process id (request ids) are blanked by `normalize`."""
from contextlib import ExitStack
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

GOLDEN = Path(__file__).with_name('data') / 'depthn_chance_off_golden.json'


def still(stack):
    """Every stage clock of the live search and constructor_worlds at 0."""
    from gpu_engine import constructor_worlds as cw, depthn_live_search as live
    for module in (live, cw):stack.enter_context(patch.object(module, 'time', NS(perf_counter=lambda: 0.)))


def live_responses(**search_keywords):
    """Responses of the fake-engine live search (tests/test_gpu_engine_depthn_live_search's fixtures) on three batches: three
    complete roots at bucket 4 (W=8), a constructor refusal, a refused world and a complete root (bucket 1), and one W=16 root.
    `search_keywords` go to every DepthNLiveSearch (the flag-off spellings of the change)."""
    from gpu_engine.phase_d_live_worlds import Refusal
    from tests import test_gpu_engine_depthn_live_search as L
    L.LiveSearchTests.setUpClass()
    case = L.LiveSearchTests('test_constructor_refusals');case.setUp();out = {}
    try:
        with ExitStack() as stack:
            still(stack)
            search = lambda W=8: case.search(W, clock=lambda: 0., **search_keywords)
            out['complete_bucket4'] = case.run_batch(search(), [L.request(f'b{i}', 0, stream=10 + i) for i in range(3)],
                {(f'b{i}', 0): case.outcome(i) for i in range(3)})[0]
            outcomes = {('p', 0): Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH'), ('w', 0): case.outcome(1, refused=(3,)), ('ok', 0): case.outcome(2)}
            out['refusals'] = case.run_batch(search(), [L.request('p', 0), L.request('w', 0), L.request('ok', 0)], outcomes)[0]
            out['w16_bucket1'] = case.run_batch(search(16), [L.request('w', 0, rung='DN2W16')], {('w', 0): case.outcome(0, 16)})[0]
    finally:case.doCleanups()
    return json.loads(json.dumps(out, sort_keys=True))


VOLATILE = re.compile(r'(^|_)seconds$|^wall$|^capture$')


def normalize(value, key=None):
    """A re-priced row with its wall clocks (any *_seconds / seconds key) and process-dependent request ids blanked."""
    if isinstance(value, dict):return {k: normalize(v, k) for k, v in value.items()}
    if isinstance(value, list):return [normalize(v) for v in value]
    if key is not None and VOLATILE.search(key) and (value is None or type(value) in (int, float)):return None
    if key == 'request_id' and isinstance(value, str):return re.sub(r'-\d+$', '-PID', value)
    return value


def reprice_rows(*extra):
    """The re-pricer's rows on its CPU fixture (tests/test_depthn_rr_reprice MainTests.replay: decisions 0-4 of one cell, 1
    single-legal, 4 constructor-refused), normalized; `extra` re-pricer flags."""
    from tests import test_depthn_rr_reprice as RT
    case = RT.MainTests('test_main_replays_records_and_joins');case.setUp()
    try:
        with tempfile.TemporaryDirectory() as tmp:rows = case.replay(tmp, *extra)[1]
    finally:case.doCleanups()
    return [normalize(r) for r in rows]


def golden():
    return dict(schema='pokezero.depthn-chance-off-golden.v1', live=live_responses(), rows=reprice_rows())


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    with path.open('x') as stream:stream.write(json.dumps(golden(), sort_keys=True, indent=0) + '\n')
    print(path)
