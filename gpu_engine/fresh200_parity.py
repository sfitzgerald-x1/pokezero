"""Strict array reduction for the fixed fresh200 corpus, never admission.

The caller must establish Native physical agreement, literal owner/request
identity, source/runtime provenance and the exact frozen roster separately.
This reducer cannot qualify those facts. It reuses the existing approved
cumulative-sum policy, after preserving a byte-bound Python float64 conversion.
"""
import hashlib

import numpy as np

ROOTS = 200
SHAPES = dict(numeric_features=(23, 132), categorical_ids=(23, 41), token_type_ids=(23,),
    attention_mask=(23,), legal_action_mask=(9,))
CUMULATIVE = frozenset((38, 39, 115, 116))


def array_identity(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
        bytes=value.nbytes, sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def python_float32(original):
    """Convert from one frozen copy of the exact original IEEE binary64 bytes."""
    return _python_float32(original, ROOTS)


def _python_float32(original, roots):
    if not isinstance(original, np.ndarray) or original.dtype != np.dtype('float64') or original.shape != (roots, 23, 132):
        raise ValueError(f'original Python float64[{roots},23,132] array required')
    payload = original.tobytes(order='C')
    frozen = np.frombuffer(payload, dtype=original.dtype).reshape(original.shape)
    if not np.isfinite(frozen).all():raise ValueError('nonfinite original Python numeric value')
    with np.errstate(over='ignore', invalid='ignore'):converted = frozen.astype(np.float32)
    if not np.isfinite(converted).all():raise ValueError('nonfinite Python float32 conversion')
    converted = np.frombuffer(converted.tobytes(order='C'), dtype=np.float32).reshape(converted.shape)
    return converted, dict(schema='pokezero.fresh200.python_float32_conversion.v1',
        input=array_identity(frozen), output=array_identity(converted),
        operation='numpy.ndarray.astype(numpy.float32) from frozen original float64 C-order bytes',
        numpy=np.__version__, original_modified=False)


def _arrays(value, *, kind, namespace, roots):
    expected = set(SHAPES) | ({'error'} if kind == 'device' else set())
    if not isinstance(value, dict) or set(value) != expected:raise ValueError('exact '+kind+' observation tensors required')
    result = {}
    for name, shape in SHAPES.items():
        if kind == 'device' and name != 'legal_action_mask':shape = (29, *shape[1:])
        array = value[name]
        if not isinstance(array, np.ndarray) or array.shape != (roots, *shape):raise ValueError(kind+' shape/array differs: '+name)
        if name == 'numeric_features':
            dtype = np.dtype('float64' if kind == 'python' else 'float32')
            if array.dtype != dtype or not np.isfinite(array).all():raise ValueError(kind+' numeric dtype/nonfinite value')
        elif name.endswith('mask'):
            if array.dtype != np.dtype('bool'):raise ValueError(kind+' Boolean mask required: '+name)
        else:
            if array.dtype.kind not in 'iu' or kind != 'python' and array.dtype != np.dtype('int32'):
                raise ValueError(kind+' integer dtype differs: '+name)
            size = namespace['categorical_vocab_size'] if name == 'categorical_ids' else namespace[
                'device_token_type_vocab_size' if kind == 'device' else 'token_type_vocab_size']
            if np.any(array < 0) or np.any(array >= size):raise ValueError(kind+' categorical/type namespace differs: '+name)
        result[name] = array
    if not result['attention_mask'][:, :23].any(axis=1).all() or not result['legal_action_mask'].any(axis=1).all():
        raise ValueError(kind+' requested root has empty attention/action mask')
    if kind == 'device':
        error = value['error']
        if not isinstance(error, np.ndarray) or error.shape != (roots,) or error.dtype.kind not in 'iu' or np.any(error != 0):
            raise ValueError('selected actual-device encoder errors are fatal')
    return result


def compare(device, b2, python, *, namespace, policy):
    """Compare all 200 rows/all cells/padding; no failed-row filtering.

    ``policy`` is the source-bound existing encoder_parity_policy module.
    Explicit namespace sizes must come from the independently bound contracts.
    The full29 device tensors stay intact, including six unpaired tokens.
    """
    return _compare_rows(device, b2, python, namespace=namespace, policy=policy, roots=ROOTS)


def compare_one(device, b2, python, *, namespace, policy):
    """Compare one sealed owning row, with a leading axis of length one."""
    return _compare_rows(device, b2, python, namespace=namespace, policy=policy, roots=1)


def _compare_rows(device, b2, python, *, namespace, policy, roots):
    if (set(namespace) != {'categorical_vocab_size', 'token_type_vocab_size', 'device_token_type_vocab_size'}
            or any(type(value) is not int or not 0 < value <= 2**31 for value in namespace.values())):
        raise ValueError('complete bound categorical/token namespaces required')
    columns = policy.cumulative_columns()
    if set(columns) != CUMULATIVE:raise ValueError('approved cumulative column policy changed')
    actual = _arrays(device, kind='device', namespace=namespace, roots=roots)
    canonical = _arrays(b2, kind='b2', namespace=namespace, roots=roots)
    original = _arrays(python, kind='python', namespace=namespace, roots=roots)
    converted, conversion = _python_float32(original['numeric_features'], roots)
    reference = dict(original, numeric_features=converted)
    shared = {name: value if name == 'legal_action_mask' else value[:, :23] for name, value in actual.items()}
    pairs = {};all_good = True
    for name, left, right in (('device_python', shared, reference), ('device_b2', shared, canonical), ('b2_python', canonical, reference)):
        bad, ulps = policy.compare_numeric(left['numeric_features'], right['numeric_features'])
        other = {field: int(np.count_nonzero(left[field] != right[field])) for field in SHAPES if field != 'numeric_features'}
        item = dict(numeric_bad_cells=int(bad.sum()), other_bad_cells=other,
            cumulative_one_ulp_cells=int(np.count_nonzero((ulps[..., sorted(CUMULATIVE)] == 1) & ~bad[..., sorted(CUMULATIVE)])),
            max_ulp=int(ulps.max()), first_numeric_difference=None)
        if bad.any():
            index = tuple(int(value) for value in np.argwhere(bad)[0]);a = left['numeric_features'];b = right['numeric_features']
            item['first_numeric_difference'] = dict(index=list(index), column=index[-1],
                left=float(a[index]), right=float(b[index]), left_bits='0x%08x' % int(a.view(np.uint32)[index]),
                right_bits='0x%08x' % int(b.view(np.uint32)[index]), ulps=int(ulps[index]))
        item['array_rules_passed'] = not bad.any() and not any(other.values())
        all_good &= item['array_rules_passed'];pairs[name] = item
    report = dict(schema='pokezero.fresh200.strict_array_parity.v1', roots=roots,
        array_rules_passed=bool(all_good), qualification=False, namespace=namespace,
        numeric_rule='exact binary32 bits except existing approved four cumulative columns <=1 ULP, same signs, finite',
        approved_cumulative_columns={str(index): columns[index] for index in sorted(columns)}, pairs=pairs,
        python_conversion=conversion, original_python={name:array_identity(value) for name, value in original.items()},
        canonical_b2={name:array_identity(value) for name, value in canonical.items()},
        full_device={name:array_identity(value) for name, value in device.items()},
        extra_device_tokens=dict(tokens=6, canonical_counterpart=False,
            retained={name:array_identity(value[:, 23:]) for name, value in actual.items() if name != 'legal_action_mask'}),
        open_admission_gates=['complete Native physical agreement', 'literal same-prefix owning references',
            'exact sealed selected addresses/owner counts and full source/runtime DAG'])
    return report, converted
