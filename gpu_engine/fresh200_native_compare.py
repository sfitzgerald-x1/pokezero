"""Seal all typed Native leaves before decoding the paired device boundary."""

import json
import re

import numpy as np

from .device_corpus_attestation import load_snapshot, save_snapshot, seal_manifest, verify_artifact
from .fresh200_boundary_gate import BoundaryAddress, BoundaryGate
from .fresh200_native_initial import _layout


def freeze_pre(root, stream, frame, group_arrays, *, recording_sha256, registry, source):
    """Seal the exact 73 PRE leaves and return their comparison gate."""
    layout = _layout()
    values = {}
    for group in group_arrays:
        if values.keys() & group.keys():
            raise ValueError('overlapping Native projection field groups')
        values.update(group)
    if values.keys() != layout.keys():
        raise ValueError(f'complete Native projection required; missing={sorted(layout.keys() - values.keys())}, '
                         f'extra={sorted(values.keys() - layout.keys())}')
    for name, value in values.items():
        spec = layout[name]
        shape = (1, *spec['shape'][1:])
        if (type(value) is not np.ndarray or value.shape != shape
                or value.dtype.str != np.dtype(spec['dtype']).str):
            raise ValueError('Native projection typed shape differs: ' + name)
    for digest in (recording_sha256, registry):
        if type(digest) is not str or re.fullmatch('[a-f0-9]{64}', digest) is None:
            raise ValueError('sealed Native recording and source registry digests required')
    gate = BoundaryGate(BoundaryAddress(stream=stream, step=frame, role='PRE'))
    # Copy before sealing. The comparison reads back these sealed bytes.
    sealed = {name: values[name].copy(order='C') for name in layout}
    for array in sealed.values():
        array.flags.writeable = False

    def seal(address, projection):
        name = f'{address.stream}-pre-{address.step}'
        snapshot = save_snapshot(root, name + '-state', projection, source=source)
        reference = seal_manifest(root, name, dict(
            stream=address.stream, frame=address.step, role=address.role, field_count=len(layout),
            source_registry_hash=registry, native_recording_sha256=recording_sha256, qualification=False,
        ), parents=[snapshot], source=source)
        return dict(address=address, reference=reference, field_count=len(layout), source_registry_hash=registry)

    gate.freeze(sealed, seal)
    return gate


def compare_pre(root, gate, expected_reader, *, row, source):
    """Compare the sealed row and report literal dtype, shape, or byte differences."""
    if not isinstance(gate, BoundaryGate) or gate.receipt is None or gate.address.role != 'PRE':
        raise ValueError('complete frozen Native PRE receipt required')
    manifest = json.loads(verify_artifact(root, gate.receipt['reference']))
    payload = manifest['payload']
    if (manifest['source'] != source or len(manifest['parents']) != 1
            or payload['stream'] != gate.address.stream or payload['frame'] != gate.address.step
            or payload['role'] != 'PRE' or payload['field_count'] != 73
            or payload['source_registry_hash'] != gate.receipt['source_registry_hash']):
        raise ValueError('comparison must use the source-bound frozen Native PRE')
    layout = _layout()
    template = {name: np.empty((1, *spec['shape'][1:]), dtype=spec['dtype']) for name, spec in layout.items()}
    frozen = load_snapshot(root, manifest['parents'][0], template, expected_source=source)
    expected = expected_reader.read(gate, row=row)
    if expected.keys() != frozen.keys():
        raise ValueError('frozen and expected physical field rosters differ')
    differences = {}
    for name, data in frozen.items():
        native, device = data[0], expected[name]
        if native.shape != device.shape or native.dtype.str != device.dtype.str:
            differences[name] = 'dtype_or_shape'
        elif native.tobytes() != device.tobytes():
            differences[name] = 'bytes'
    return differences
