#!/usr/bin/env python3
"""Install the pinned Phase0 runtime into an owned worker without overwrites.

Preflight the entire archive and every existing destination before creating
anything. Existing identical files/links are retained, not copied or chmodded.
This installs runtime files only; it cannot launch a game or create a pod.
Use only a fresh, inactive, exclusively prepared worker: destination parent
checks are pathname-based, not protection against concurrent hostile writers.
Absolute links require the original root layout and external /usr/bin/python3.12;
a scratch-root roundtrip is an archive test, not a runnable relocated environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import stat
import sys
import tarfile

RUNTIME = "nvme/pokezero-phase0-reference-20261019.1Pq3fK"
PYTHON_ALIAS = "root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu"
PYTHON_BASE = "root/.local/share/uv/python/cpython-3.13.15-linux-aarch64-gnu"
TREES = (PYTHON_BASE, *(RUNTIME + "/" + name for name in (
    "repo", ".venv", "showdown", "foul-play", "foulplay-venv", "node-v22.22.2-linux-arm64")))
FILES = {PYTHON_ALIAS, *(RUNTIME + "/" + name for name in (
    "transformer-policy.pt", "foulplay-affinity-python.sh", "authority-plan.md",
    "primary-roster-entry.json", "exclusion-basis.json"))}


def digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def allowed(name):
    return name in FILES or any(name == prefix or name.startswith(prefix + "/") for prefix in TREES)


def member_name(raw):
    name = raw.rstrip("/")
    path = PurePosixPath(name)
    if (path.is_absolute() or str(path) != name or ".." in path.parts
            or not allowed(name)):
        raise ValueError(f"archive path outside the explicit runtime: {raw}")
    return name


def parents_are_directories(root, path):
    for parent in reversed(path.parents):
        if parent == root or root in parent.parents:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f"destination parent is not a plain directory: {parent}")


def check_link(member, name, members):
    if member.issym():
        target = member.linkname
        if target.startswith("/"):
            expected = {
                PYTHON_ALIAS: "/" + PYTHON_BASE,
                RUNTIME + "/.venv/bin/python": "/" + PYTHON_ALIAS + "/bin/python3.13",
                RUNTIME + "/foulplay-venv/bin/python": "/usr/bin/python3.12",
            }
            if expected.get(name) != target:
                raise ValueError(f"unregistered absolute symlink: {name}")
        elif not allowed(posixpath.normpath(posixpath.join(posixpath.dirname(name), target))):
            raise ValueError(f"symlink escapes runtime: {name}")
    elif member.islnk():
        target = member_name(member.linkname)
        if target not in members or not members[target].isfile():
            raise ValueError(f"hardlink target is not an archived regular file: {name}")


def preflight(archive, root):
    members = {}
    for member in archive.getmembers():
        name = member_name(member.name)
        if name in members:
            raise ValueError(f"duplicate archive member: {name}")
        if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise ValueError(f"unsupported archive member type: {name}")
        members[name] = member
    for name, member in members.items():
        check_link(member, name, members)
        path = root / name
        parents_are_directories(root, path)
        # Refuse descendants through a symlink planned by this same archive,
        # even if the link has not been installed yet.
        for parent in PurePosixPath(name).parents:
            prior = members.get(str(parent))
            if prior is not None and not prior.isdir():
                raise ValueError(f"archive parent is not a directory: {name}")
        if not path.exists() and not path.is_symlink():
            continue
        if member.issym():
            identical = path.is_symlink() and os.readlink(path) == member.linkname
        elif member.isdir():
            identical = path.is_dir() and not path.is_symlink()
        else:
            identical = path.is_file() and not path.is_symlink()
            if identical:
                source_member = members[member.linkname] if member.islnk() else member
                with path.open("rb") as current, archive.extractfile(source_member) as expected:
                    identical = digest(current) == digest(expected)
                identical = identical and stat.S_IMODE(path.stat().st_mode) == member.mode & 0o777
        if not identical:
            raise ValueError(f"existing destination differs; nothing installed: {path}")
    return members


def install(bundle, expected_sha256, destination, pod_uid=None):
    root = Path(destination).resolve()
    if root == Path("/") and (not sys.platform.startswith("linux") or not pod_uid
            or os.environ.get("POKEZERO_WORKER_POD_UID") != pod_uid):
        raise ValueError("system-root installation requires the owned Linux worker's downward-API UID")
    created, retained = 0, 0
    # Hash and extract the same open inode, not two independently opened paths.
    with Path(bundle).open("rb") as stream:
        if digest(stream) != expected_sha256:
            raise ValueError("bundle SHA256 mismatch")
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r:gz") as archive:
            members = preflight(archive, root)
            # Hardlinks follow regular files regardless of archive ordering.
            ordered = sorted(members.items(), key=lambda item: item[1].islnk())
            for name, member in ordered:
                path = root / name
                parents_are_directories(root, path)
                if path.exists() or path.is_symlink():
                    retained += 1
                    continue  # Never replace or chmod an existing destination.
                path.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    path.mkdir(mode=member.mode & 0o777)
                elif member.issym():
                    os.symlink(member.linkname, path)
                elif member.islnk():
                    os.link(root / member.linkname, path, follow_symlinks=False)
                else:
                    with archive.extractfile(member) as source, path.open("xb") as target:
                        shutil.copyfileobj(source, target, 1024 * 1024)
                    path.chmod(member.mode & 0o777)
                created += 1
    return dict(bundle_sha256=expected_sha256, members=len(members), created=created,
                retained_identical=retained, scientific_work_launched=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--pod-uid")
    args = parser.parse_args()
    print(json.dumps(install(args.bundle, args.sha256, args.root, args.pod_uid)), flush=True)
