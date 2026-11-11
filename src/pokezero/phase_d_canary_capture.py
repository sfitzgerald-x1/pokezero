"""Private opt-in capture bootstrap for the original canary endpoint workers.

This module may be packaged beside the endpoint owner, outside its clean,
pinned source checkout. It wraps the SAME original factory, worker and process
owner. No cell, seed, model, scorer, RNG, precision or scheduling code is copied.
Importing this module installs nothing and starts no process or game.
"""
import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

_INSTALLED = None


def _capture(path, expected_sha256):
    name = 'pokezero.phase_d_owning_capture'
    if path is None:
        if expected_sha256 is not None:
            raise ValueError('explicit private capture source path required')
        return importlib.import_module(name)
    path = Path(path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('reviewed private owning capture source differs')
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise ValueError('another owning capture module is already loaded')
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def install_endpoint_capture(*, worker_script=None, capture_source=None, expected_capture_sha256=None):
    """Install explicitly in the owner child AND each actual persistent worker.

    The private path mode preserves the original clean endpoint checkout. The
    original process launcher retains all pipe/session/cleanup ownership; only
    its exact canary worker entry point is replaced with this bootstrap.
    """
    global _INSTALLED
    worker_script = Path(worker_script or __file__).resolve(strict=True)
    source = None if capture_source is None else Path(capture_source).resolve(strict=True)
    identity = (str(worker_script), None if source is None else str(source), expected_capture_sha256)
    if _INSTALLED is not None:
        if _INSTALLED != identity:
            raise ValueError('endpoint capture installation binding differs')
        return
    capture = _capture(source, expected_capture_sha256)
    from pokezero import b4_canary_endpoint as endpoint, b4_device_canary_runtime as runtime
    original_actor = runtime.Factory.actor
    original_worker = endpoint.worker
    original_spawn = endpoint._spawn_owned

    def actor(self, value):
        return capture.capture_actor(original_actor(self, value))

    async def worker(request_path, expected_sha256, **options):
        blob = Path(request_path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected_sha256:
            raise ValueError('original endpoint capture request digest differs')
        request = json.loads(blob)
        # Scope in THIS worker process, not just in its parent. The untouched
        # worker still checks original declaration/start/claim/checkpoint before
        # calling runtime.run. Capture creates files only when its actor opens.
        with capture.capture_to(Path(request['run_directory']) / 'phase-d-owning-wire',
                cell=request['cell'], declaration_sha256=request['declaration_sha256']):
            return await original_worker(request_path, expected_sha256, **options)

    async def spawn(*args, **options):
        if len(args) >= 4 and args[1:4] == ('-B', '-m', 'pokezero.b4_device_canary_runtime'):
            replacement = [args[0], '-B', str(worker_script)]
            if source is not None:
                replacement += ['--capture-source', str(source), '--expected-capture-sha256', expected_capture_sha256]
            args = (*replacement, *args[4:])
        return await original_spawn(*args, **options)

    runtime.Factory.actor = actor
    endpoint.worker = worker
    endpoint._spawn_owned = spawn
    _INSTALLED = identity


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-pair', nargs=2, type=int, required=True)
    p.add_argument('--capture-source')
    p.add_argument('--expected-capture-sha256')
    args = p.parse_args(argv)
    install_endpoint_capture(capture_source=args.capture_source,
        expected_capture_sha256=args.expected_capture_sha256)
    from pokezero.b4_device_canary_runtime import serve
    return asyncio.run(serve(tuple(args.cpu_pair)))


if __name__ == '__main__':
    raise SystemExit(main())
