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
    """'' when `sha` is a GitHub pull request merge (two or more parents, committed by GitHub, 'Merge pull request #N' / 'Merge #N:')
    that changed gpu_engine/ and that HEAD contains; else why not. A squash of integration onto main would rewrite these commits and
    fail every entry: main takes integration by merge commit."""
    if not re.fullmatch(r'[0-9a-f]{8,40}', sha or ''):return f'{sha!r} is not a hex commit id'
    if sha == PLACEHOLDER:return f'{sha} is the placeholder, not a merge'
    full = git('rev-parse', '--verify', '--quiet', f'{sha}^{{commit}}')
    if full.returncode != 0 or not full.stdout.strip():return f'{sha} is not a commit here (or is ambiguous)'
    full = full.stdout.strip()
    listing = git('rev-list', '--parents', '-n', '1', full)
    if listing.returncode != 0:return f'{sha}: git rev-list failed ({listing.returncode})'
    if len(listing.stdout.split()[1:]) < 2:return f'{sha} is not a merge commit (engine_since names the integration merge)'
    # A PR's own (local) merge of integration is also a merge HEAD contains, and may predate the engine merge: engine_since must be a
    # pull request merge GitHub made (committer noreply@github.com; subject 'Merge pull request #N' or the older 'Merge #N: ...';
    # not the 'Update branch' button's 'Merge branch ...') that changed gpu_engine/ against its first parent (an engine landing, not
    # an unrelated PR). It cannot prove it is THIS effect's engine PR; the guard's evidence names it. --no-show-signature: GitHub
    # signs its merges, and log.showSignature would otherwise put gpg lines on stdout.
    meta = git('log', '-1', '--no-show-signature', '--format=%ce%n%s', full)
    committer, _, subject = meta.stdout.partition('\n')
    if meta.returncode != 0 or committer.strip() != 'noreply@github.com' or not re.match(r'Merge (pull request )?#\d+', subject):
        return f'{sha} is not a GitHub pull request merge (committer {committer.strip()!r}, subject {subject.strip()[:50]!r})'
    changed = git('diff', '--quiet', f'{full}^1', full, '--', 'gpu_engine/').returncode
    if changed == 0:return f'{sha} did not change gpu_engine/ (not an engine merge)'
    if changed != 1:return f'{sha}: git diff failed ({changed})'
    ancestor = git('merge-base', '--is-ancestor', full, 'HEAD')
    if ancestor.returncode == 1:return f'{sha} is not an ancestor of HEAD'
    if ancestor.returncode != 0:return f'{sha}: git merge-base failed ({ancestor.returncode}): {ancestor.stderr.strip()[:200]}'
    return ''
