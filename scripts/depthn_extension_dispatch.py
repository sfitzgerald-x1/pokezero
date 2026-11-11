#!/usr/bin/env python3
"""The 2,000-cell extension of a panel run under the unknown-cells amendment and its addenda
(evals/depthn-foulplay-dn2w16-amendment-extension-20261108.json): the frozen panel's pending(limit=2000) gates the extension on the
official screen readout only, and the run's own tree (its readout's code identity) cannot change, so this script dispatches it with the
panel's own cell, request, packet, interleave and dispatch functions, unchanged, behind the amended gate:
- the official screen readout (sha256-bound) is genuine: a 500-cell readout of this run's declaration whose recomputation now reproduces
  its bytes, as pending() requires;
- the amended verdict (sha256-bound) is the one depthn_screen_amended_verdict.verdict gives for that readout, this run's result files and
  the committed amendment and its first two addenda (each by sha256; the extension addendum names all three), and its decision is exactly
  CONTINUE_TO_2000 with no amended breakage (every official breakage item except unknown_cells still applies);
- the final fairness gate (evals/depthn-foulplay-dn2w16-amendment-final-fairness-20261108.json, an addendum to all four earlier files),
  recomputed here over the whole screen by depthn_fairness_final.final_gate, is within 5%; the amended verdict must carry that result;
- anything else is refused: STOP_HARM, STOP_BREAKAGE, VOID_*, FLAG_ARM_FAIRNESS, INCOMPLETE, another tree or a sha256 that does not match.
Another run adopts the extension through its run binding (--run-binding PATH SHA256): the adoption record that
depthn_screen_amended_verdict.py pins in RUN_BINDINGS (the four rule files, unchanged, bound to the run's directory, declaration and
tree), carrying an `extension` entry {path, sha256} that adopts this extension file unchanged, and pinned again here in
EXTENSION_BINDINGS (a binding not pinned here cannot extend). The gate then takes the run fields from the binding, recomputes the
binding's declared final fairness gate (RULE_R3: depthn_fairness_final.final_gate_r3) instead of the coarse one, and holds the amended
verdict to depthn_screen_amended_verdict.verdict under that binding; every other check is unchanged.
After the gate, extension_pending is pending()'s body past its own gate, verbatim (tests pin equal packets, order and attempted counts
with pending(limit=2000) on an official CONTINUE_TO_2000), and the dispatch steps are depthn_foulplay_panel's dispatch mode's.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

from pokezero import depthn_live_panel as panel
from scripts import depthn_fairness_final as FF
from scripts import depthn_screen_amended_verdict as V

TREE = Path(__file__).resolve().parents[1]
FILES = dict(amendment='evals/depthn-foulplay-dn2w16-amendment-unknown-cells-20261108.json',
             addendum='evals/depthn-foulplay-dn2w16-amendment-harm-mirror-20261108.json',
             both='evals/depthn-foulplay-dn2w16-amendment-both-bounds-20261108.json',
             extension='evals/depthn-foulplay-dn2w16-amendment-extension-20261108.json',
             final='evals/depthn-foulplay-dn2w16-amendment-final-fairness-20261108.json')
# The committed chain, pinned: main() refuses any other bytes (a self-consistent but different chain would otherwise pass the gate).
PINNED = dict(amendment='c4ff30a25a573f90fdd02957b40fb8307d8b87b56dfbfe60d85ea662b18e8b3b',
              addendum='dbce9a54223969a213a3e584fd2e01bbe259ebad7530678fbba7cd38ecd9539f',
              both='98e6120f30565c0b40d423504aa0d07a2929b2207551a5c901adb7579d361e13',
              extension='8a708ebbc6a4206dafc38b2d729656e25da6cf914931b5424353f441f6e9ed00',
              final='b2fccc253136b2bce3125cc773973c7186a0dff6a5ff8e6825a5618d6268050e')
EXTENSION_BINDINGS = dict(screen_r4a='8a7ef53684f732f86cc7522abebb9a40a416d51f86786b77b0a9840fed9ccb7d',
                          screen_r4a2='b8cc7ee97a795b7eb6c7df069b4c0652acc77267a88bf50fb925003d8317a167')   # {run name: committed adoption-record sha256} of runs that pre-registered the extension (also in V.RUN_BINDINGS)


def committed(pinned=PINNED):
    """The five committed files of this tree as {name: (value, sha256)}, each refused unless its bytes are the pinned ones."""
    out = {}
    for name, path in FILES.items():
        raw = (TREE/path).read_bytes();digest = hashlib.sha256(raw).hexdigest()
        if pinned is not None and digest != pinned[name]:raise ValueError(f'{path}: not the committed amendment file ({digest})')
        out[name] = (json.loads(raw), digest)
    return out


def bound_run(binding, files):
    """(declaration_sha256, run_directory, final fairness rule) of a run binding that adopts the four rule files unchanged
    (V.run_fields) and this extension file unchanged (its `extension` entry); ValueError otherwise."""
    value, digest = binding;final_sha = files['final'][1]
    declaration_sha256, directory, _ = V.run_fields(value, digest, [files[k][1] for k in ('amendment', 'addendum', 'both')] + [final_sha])
    entry = value.get('extension') if type(value.get('extension')) is dict else {}
    if entry != dict(path=FILES['extension'], sha256=files['extension'][1], change='none'):
        raise ValueError(f'a run binding adopting the extension file unchanged required ({digest})')
    rule = V.final_rule(value, digest)
    if rule != 'RULE_R3' or V.RUN_AMENDMENTS.get(digest):   # the bound route's only final gate; run amendments (screen-r2's) are not carried
        raise ValueError(f'a run binding declaring final_fairness_rule RULE_R3 and needing no run amendment required ({digest})')
    return declaration_sha256, directory, rule


def gate(run_directory, *, screen, verdict, files, tree_commit=None, binding=None):
    """The amended extension gate (module docstring). screen and verdict are (path, sha256); files is committed()'s shape; binding
    None (screen-r1) or (the run binding, its sha256). Returns the amended verdict; raises ValueError on anything else."""
    root, run, d, lad = panel.load(run_directory);panel.require_tree(run, tree_commit)
    (amendment, amendment_sha), (addendum, addendum_sha), (both, both_sha), (extension, _) = (files[k] for k in ('amendment', 'addendum', 'both', 'extension'))
    gate_rule = None
    if binding is None:
        if run['run_directory'] != amendment['run']['run_directory']:raise ValueError('the amended run\'s directory required')
        declaration_sha256 = extension.get('declaration_sha256')
    else:
        declaration_sha256, directory, gate_rule = bound_run(binding, files)
        if str(Path(run['run_directory']).resolve()) != directory:raise ValueError('the bound run\'s directory required')
    if (extension.get('schema') != V.AMENDMENT_SCHEMA or declaration_sha256 != run['declaration_sha256']   # the binding's, or the extension's own
            or sorted(x.get('sha256') for x in extension.get('addendum_to') or ()) != sorted([amendment_sha, addendum_sha, both_sha])):
        raise ValueError('the extension addendum to this declaration\'s amendment and both addenda required')
    value = panel.read_bound(*screen)
    if (type(value) is not dict or value.get('schema') != panel.READOUT_SCHEMA or value.get('declaration_sha256') != run['declaration_sha256']
            or value.get('paired_cells') != d['screen_paired_cells'] or value.get('arm') != d['arms'][1]):
        raise ValueError('the pinned 500-cell screen readout of this run required')
    again = panel.readout(root, arm=value['arm'], limit=d['screen_paired_cells'], codex_raw_directory=value['codex_raw_directory'],
        server_telemetry=value['server_telemetry'], tree_commit=tree_commit, nodes=value.get('nodes'))
    if panel.sha(again) != screen[1]:raise ValueError('screen readout differs from its recomputation')
    final, final_sha = files['final']
    if not {amendment_sha, addendum_sha, both_sha, files['extension'][1]} <= {x.get('sha256') for x in final.get('addendum_to') or ()}:
        raise ValueError('the final-fairness addendum to this amendment chain required')
    amended = panel.read_bound(*verdict)
    if gate_rule == 'RULE_R3':final_fairness = FF.final_gate_r3(root, value['arm'], d['screen_paired_cells'])   # the binding's declared gate
    else:final_fairness = FF.final_gate(root, value['arm'], d['screen_paired_cells'])   # the whole run, recomputed here
    if final_fairness['within'] is not True:raise ValueError(f"the final fairness gate fails: {final_fairness.get('finer', final_fairness)}")
    expected = V.verdict(value, amendment, root, readout_sha256=screen[1], amendment_sha256=amendment_sha, addendum=addendum,
        addendum_sha256=addendum_sha, both=both, both_sha256=both_sha, final=final, final_sha256=final_sha, final_fairness=final_fairness,
        **({} if binding is None else dict(binding=binding[0], binding_sha256=binding[1])))
    if amended != json.loads(json.dumps(expected)):raise ValueError('amended verdict differs from its recomputation')
    if amended['decision'] != 'CONTINUE_TO_2000' or amended['breakage'] != []:
        raise ValueError(f"an amended verdict continuing to 2000 with no breakage required (found {amended['decision']}, {amended['breakage']})")
    return amended


def extension_pending(run_directory, *, arm, resume=False, tree_commit=None):
    """pending(limit=2000)'s body past its screen gate, verbatim (the gate is gate())."""
    root, run, d, lad = panel.load(run_directory);panel.require_tree(run, tree_commit);limit = d['max_paired_cells']
    if arm not in d['arms']:raise ValueError('undeclared arm')
    if d['expected_searched_share'] is None:raise ValueError('expected searched share from the constructor-world gate must be set before dispatch')
    panel.refuse_by_state(root)
    todo = [];attempted = 0
    for cell in panel.cells(lad, arm, limit=limit):
        if (root/'claims'/f"{cell['ordinal']}.json").exists():attempted += 1;continue
        path = root/'requests'/f"{cell['ordinal']}.json";value = panel.request(run, cell)
        if path.exists():
            if path.read_bytes() != panel.encoded(value):raise ValueError('existing cell request differs')
        else:panel.write_new(path, value)
        todo.append(dict(path=str(path), sha256=panel.sha(value), arm=arm, pair_ordinal=cell['pair_ordinal']))
    if attempted and not resume:raise ValueError('attempted cells exist; resume explicitly')
    return todo


def extension_interleaved(run_directory, *, search_arm, resume=False, tree_commit=None):
    """interleaved(limit=2000)'s body with extension_pending, verbatim."""
    root, run, d, lad = panel.load(run_directory);limit = d['max_paired_cells']
    if search_arm not in d['arms'][1:]:raise ValueError('declared search arm required')
    packets = []
    for arm in ('RAW', search_arm):packets += extension_pending(root, arm=arm, resume=resume, tree_commit=tree_commit)
    packets.sort(key=lambda p: (p['pair_ordinal'], p['arm'] != 'RAW'))
    attempted = {arm: sum((root/'claims'/f"{c['ordinal']}.json").exists() for c in panel.cells(lad, arm, limit=limit)) for arm in ('RAW', search_arm)}
    return packets, attempted


def server(value):
    host, _, port = value.rpartition(':')
    if not host or not port.isdigit():raise argparse.ArgumentTypeError('HOST:PORT required')
    return dict(host=host, port=int(port))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-directory', required=True);p.add_argument('--search-arm', choices=panel.SEARCH_ARMS, required=True)
    for name in ('screen-readout', 'screen-readout-sha256', 'amended-verdict', 'amended-verdict-sha256', 'cache-directory'):p.add_argument('--' + name, required=True)
    p.add_argument('--cpu-pair', nargs=2, type=int, action='append', required=True);p.add_argument('--server', type=server, required=True)
    p.add_argument('--probe-capture');p.add_argument('--cells-per-worker', type=int, default=1);p.add_argument('--resume', action='store_true')
    p.add_argument('--accept-uncalibrated', action='store_true');p.add_argument('--tree-commit')
    p.add_argument('--run-binding', nargs=2, metavar=('PATH', 'SHA256'), help='the run binding (adoption record) of a run that '
        'pre-registered the extension: pinned in EXTENSION_BINDINGS, in the verdict\'s RUN_BINDINGS and, with its final rule, RUN_FINAL_RULES')
    args = p.parse_args(argv);binding = None
    if args.run_binding is not None:
        path, digest = args.run_binding
        if digest not in EXTENSION_BINDINGS.values() or digest not in V.RUN_BINDINGS.values():
            raise SystemExit(f'--run-binding must be a committed binding pinned in EXTENSION_BINDINGS and RUN_BINDINGS: {sorted(EXTENSION_BINDINGS.values())}')
        binding = (V.read_bound(path, digest), digest)
        if V.RUN_FINAL_RULES.get(digest) != V.final_rule(binding[0], digest):raise SystemExit('--run-binding: its final rule must be pinned in RUN_FINAL_RULES')
    scripts = {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (__file__, V.__file__, FF.__file__)}   # outside the panel's digest
    print(json.dumps(dict(event='EXTENSION_START', scripts=scripts)), flush=True)   # first, so a crash mid-dispatch keeps the audit trail
    amended = gate(args.run_directory, screen=(args.screen_readout, args.screen_readout_sha256),
        verdict=(args.amended_verdict, args.amended_verdict_sha256), files=committed(), tree_commit=args.tree_commit, binding=binding)
    packets, attempted = extension_interleaved(args.run_directory, search_arm=args.search_arm, resume=args.resume, tree_commit=args.tree_commit)
    # depthn_foulplay_panel's dispatch mode from here, step for step
    root, _, declaration, _ = panel.load(args.run_directory)
    panel.require_calibration(root, declaration, pairs=args.cpu_pair, accept_uncalibrated=args.accept_uncalibrated)
    if args.cells_per_worker != declaration['actor']['cells_per_worker']:
        raise SystemExit(f"--cells-per-worker must be the declared {declaration['actor']['cells_per_worker']}")
    monitor = panel.Monitor(args.run_directory, search_arm=args.search_arm)
    capture = args.probe_capture or panel.probe_capture(args.run_directory)
    probe = panel.probe(args.run_directory, arm=args.search_arm, server=args.server, capture=capture, tree_commit=args.tree_commit)
    cache = Path(args.cache_directory).resolve()
    result = asyncio.run(panel.dispatch(packets, pairs=[tuple(p) for p in args.cpu_pair], servers={'RAW': None, args.search_arm: args.server},
        cells_per_worker=args.cells_per_worker, cache_directory=cache, attempted=attempted, after=monitor, warmup=capture))
    print(json.dumps(dict(result, probe=probe['status'], amended_decision=amended['decision'], scripts=scripts)))
    return 1 if result['failed'] or result['stopped'] else 0


if __name__ == '__main__':sys.exit(main())
