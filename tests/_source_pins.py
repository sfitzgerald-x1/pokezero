"""The engine's source-pin tables, read from the files' text (stdlib only; no engine import), shared by the tests that
bind them.

PIN_COUNTS is the one place a pinned file is added or dropped: each table's size is a deliberate review act, asserted
by tests/test_gpu_engine_source_pin_refresh.py against the same tables scripts/refresh_source_pins.py rewrites.
Every other test reads tables through `table()` and the counts through PIN_COUNTS, so a new pin edits one number here
instead of every test that once repeated it."""
import ast
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'gpu_engine'
# (file, table) -> entries. SOURCE_PINS / MAPPING_PINS are sorted by file; DOMAIN_PINS / KERNEL_PINS hold one entry per
# transition-domain module, in their module list's order.
PIN_COUNTS = {
    ('engine_source_frame_witness.py', 'SOURCE_PINS'): 92,   # includes residual_order, transform, infatuation and forecast
    ('chance_consumers_pins.py', 'SOURCE_PINS'): 14,
    ('chance_consumers_pins.py', 'DOMAIN_PINS'): 29,
    ('phase_d_chance.py', 'SOURCE_PINS'): 14,
    ('phase_d_chance.py', 'DOMAIN_PINS'): 29,
    ('ordinary_law.py', 'MAPPING_PINS'): 6,
    ('ordinary_law.py', 'DOMAIN_PINS'): 29,
    ('conditional_law.py', 'MAPPING_PINS'): 4,
    ('conditional_law.py', 'KERNEL_PINS'): 29,
}
# per-module tables -> (file holding the module list, list name); the gates require the table to name it in order
DOMAIN_TABLES = {
    ('chance_consumers_pins.py', 'DOMAIN_PINS'): ('chance_consumers.py', 'DOMAIN_MODULES'),
    ('phase_d_chance.py', 'DOMAIN_PINS'): ('phase_d_chance.py', 'DOMAIN_MODULES'),
    ('ordinary_law.py', 'DOMAIN_PINS'): ('chance_consumers.py', 'DOMAIN_MODULES'),
    ('conditional_law.py', 'KERNEL_PINS'): ('conditional_law.py', 'KERNEL_MODULES'),
}


def assignment(file, name, text=None):
    """The value node of the module-level `name = ...` in gpu_engine/`file` (or in `text`)."""
    tree = ast.parse((ENGINE / file).read_text() if text is None else text)
    return next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def table(file, name, text=None):
    """A pin table's literal, as the module that holds it would load it."""
    return ast.literal_eval(assignment(file, name, text))


def modules(file, name, text=None):
    """A module list (`NAME = tuple("a b c".split())`) as its names, in order."""
    source = (ENGINE / file).read_text() if text is None else text
    return re.search(rf'^{name} = tuple\("([^"]*)"\.split\(\)\)$', source, re.M).group(1).split()


def sha(name, root=ENGINE):
    return hashlib.sha256((root / name).read_bytes()).hexdigest()


def domain_digest(names, read):
    """The single transition-domain digest the per-module tables replaced: sha256 over name, NUL, bytes, NUL in order."""
    return hashlib.sha256(b''.join(n.encode() + b'\0' + read(n + '.py') + b'\0' for n in names)).hexdigest()
