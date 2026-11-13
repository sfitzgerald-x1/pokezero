"""engine_since checks shared by the effect-guard and carry-audit tests (git only; no repo imports).

An engine_since names the integration merge from which the engine models an effect. A guard or audit change may land only on a tree
that contains that merge, and never with the placeholder used while the merge SHA is unknown. depthn-light checks out with full
history and fails on any skip, so these checks run there."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# The placeholder a guard PR carries until its engine merge exists. Split so that filling it (a repo-wide substitution) never
# rewrites this check.
PLACEHOLDER = 'f00d' + 'feed'


def git(*args):
    """git in this checkout: the CompletedProcess (returncode 127 when git itself is missing, 124 on a timeout)."""
    try:return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, timeout=60)
    except FileNotFoundError:return subprocess.CompletedProcess(args, 127, '', 'git not found')
    except subprocess.TimeoutExpired:return subprocess.CompletedProcess(args, 124, '', 'git timed out')


def _toplevel():
    top = git('rev-parse', '--show-toplevel')
    return Path(top.stdout.strip()).resolve() if top.returncode == 0 and top.stdout.strip() else None


# This tree is its own git checkout (not a copy nested inside some other repository, not a tree git refuses to read).
IN_GIT = _toplevel() == ROOT
SKIP_REASON = 'needs this tree as a git checkout (depthn-light checks out with full history and fails on any skip)'


def shallow():
    return git('rev-parse', '--is-shallow-repository').stdout.strip() == 'true'


def problem(sha):
    """'' when `sha` is an integration merge (two or more parents, GitHub's 'Merge pull request #N' subject) that HEAD contains; else why
    not. A squash of integration onto main would rewrite these commits and fail every entry: main takes integration by merge commit."""
    if not re.fullmatch(r'[0-9a-f]{8,40}', sha or ''):return f'{sha!r} is not a hex commit id'
    if sha == PLACEHOLDER:return f'{sha} is the placeholder, not a merge'
    full = git('rev-parse', '--verify', '--quiet', f'{sha}^{{commit}}')
    if full.returncode != 0 or not full.stdout.strip():return f'{sha} is not a commit here (or is ambiguous)'
    full = full.stdout.strip()
    listing = git('rev-list', '--parents', '-n', '1', full)
    if listing.returncode != 0:return f'{sha}: git rev-list failed ({listing.returncode})'
    if len(listing.stdout.split()[1:]) < 2:return f'{sha} is not a merge commit (engine_since names the integration merge)'
    # a PR's own merge of integration is also a merge HEAD contains (and may predate the engine merge): engine_since must be the
    # integration merge of the engine PR itself, GitHub's 'Merge pull request #N' (integration lands PRs by merge commit, never squash)
    subject = git('log', '-1', '--format=%s', full)
    if subject.returncode != 0 or not subject.stdout.startswith('Merge pull request #'):
        return f'{sha} is not a pull request merge (its subject: {subject.stdout.strip()[:60]!r})'
    ancestor = git('merge-base', '--is-ancestor', full, 'HEAD')
    if ancestor.returncode == 1:return f'{sha} is not an ancestor of HEAD'
    if ancestor.returncode != 0:return f'{sha}: git merge-base failed ({ancestor.returncode}): {ancestor.stderr.strip()[:200]}'
    return ''
