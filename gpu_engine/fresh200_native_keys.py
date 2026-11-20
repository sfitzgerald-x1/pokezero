"""Bind the external device key chain to the frozen Native prefix phases."""

import re

import numpy as np

from .fresh200_fixed_prefix import FrozenPrefix
from .fresh200_native_archive import RecordedBoundary
from .fresh200_witness_audit import document


def project_key_chain(prefix, boundaries, raw_key):
    """Use the constructor operand and the same split rule as the pinned engine."""
    import jax

    if (not isinstance(prefix, FrozenPrefix) or not isinstance(raw_key, np.ndarray)
            or raw_key.shape != (2,) or raw_key.dtype != np.uint32):
        raise ValueError('sealed original constructor uint32 key pair required')
    start = document(prefix.start_wire.decode('utf8'))
    seed = start['seed']
    if type(seed) is not str or not re.fullmatch(r'\d{1,5}(,\d{1,5}){3}', seed, re.ASCII):
        raise ValueError('literal Native constructor seed coordinates required')
    words = tuple(map(int, seed.split(',')))
    if any(word > 65535 for word in words):
        raise ValueError('Native constructor seed word exceeds its uint16 coordinate')
    joined = [(words[0] << 16) | words[1], (words[2] << 16) | words[3]]
    if joined != raw_key.tolist():
        raise ValueError('Native start seed does not match the sealed constructor key')
    if not jax.config.jax_threefry_partitionable:
        raise ValueError('pinned engine requires the partitionable Threefry split')
    if len(boundaries) != len(prefix.steps) + 1:
        raise ValueError('key chain must span the complete frozen prefix')

    result = []
    key = jax.random.wrap_key_data(raw_key, impl='threefry2x32')
    for frame, boundary in enumerate(boundaries):
        if (not isinstance(boundary, RecordedBoundary) or boundary.frame != frame
                or boundary.phase.kind == 'terminal'):
            raise ValueError('key chain requires contiguous continuing Native phases')
        data = np.asarray(jax.random.key_data(key))
        result.append(dict(keys=data[None]))
        if frame < len(prefix.steps):
            step = prefix.steps[frame]
            if step.frame != frame or step.requested != boundary.phase.requested:
                raise ValueError('key update is outside the admitted choice and phase binding')
            # The engine keeps the key on suspended-turn resumes. Every
            # admitted live, non-resuming step splits once, even a replacement.
            if not boundary.phase.pending_residual:
                key = jax.random.split(key, 2)[0]
    return tuple(result)
