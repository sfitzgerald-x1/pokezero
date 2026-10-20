"""Small authored archives only; no scientific runtime or cluster operations."""
import importlib.util
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "installer", Path(__file__).resolve().parents[1] / "scripts" / "install_phase0_reference_bundle.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def bundle(self, root, entries):
        path = root / "bundle.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, kind, value in entries:
                info = tarfile.TarInfo(name)
                info.mode = 0o755 if kind == "dir" else 0o644
                if kind == "file":
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
                else:
                    info.type = {"dir": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE,
                                 "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                    info.linkname = value
                    archive.addfile(info)
        with path.open("rb") as stream:
            checksum = installer.digest(stream)
        return path, checksum

    def test_install_and_repeated_preparation_keep_identical_files_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = installer.RUNTIME + "/repo/module.py"
            bundle, checksum = self.bundle(root, [(name, "file", b"frozen")])
            destination = root / "worker"
            result = installer.install(bundle, checksum, destination)
            path = destination / name
            before = path.stat()
            self.assertEqual(path.read_bytes(), b"frozen")
            self.assertEqual(result["created"], 1)
            second = installer.install(bundle, checksum, destination)
            self.assertEqual(second["created"], 0)
            self.assertEqual(second["retained_identical"], 1)
            self.assertEqual(path.stat().st_ino, before.st_ino)
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)

    def test_conflict_anywhere_prevents_all_writes_and_preserves_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, last = (installer.RUNTIME + "/repo/" + name for name in ("new.py", "owned.py"))
            bundle, checksum = self.bundle(root, [(first, "file", b"new"), (last, "file", b"expected")])
            destination = root / "worker"
            owned = destination / last
            owned.parent.mkdir(parents=True)
            owned.write_bytes(b"user-owned")
            with self.assertRaisesRegex(ValueError, "nothing installed"):
                installer.install(bundle, checksum, destination)
            self.assertFalse((destination / first).exists())
            self.assertEqual(owned.read_bytes(), b"user-owned")

    def test_digest_mismatch_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, _checksum = self.bundle(root, [(installer.RUNTIME + "/repo/a", "file", b"a")])
            with self.assertRaisesRegex(ValueError, "SHA256"):
                installer.install(bundle, "wrong", root / "worker")
            self.assertFalse((root / "worker").exists())

    def test_traversal_unregistered_paths_duplicate_and_special_files_refuse(self):
        for entries in (
            [("../escape", "file", b"bad")], [("/root/escape", "file", b"bad")],
            [("etc/passwd", "file", b"bad")],
            [(installer.RUNTIME + "/repo/a", "file", b"a")] * 2,
            [(installer.RUNTIME + "/repo/fifo", "fifo", "")],
        ):
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle, checksum = self.bundle(root, entries)
                with self.assertRaises(ValueError):
                    installer.install(bundle, checksum, root / "worker")
                self.assertFalse((root / "worker").exists())

    def test_existing_or_planned_symlink_parent_cannot_redirect_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "worker"
            prefix = installer.RUNTIME + "/repo"
            redirected = root / "unrelated"
            redirected.mkdir()
            (destination / installer.RUNTIME).mkdir(parents=True)
            (destination / prefix).symlink_to(redirected, target_is_directory=True)
            bundle, checksum = self.bundle(root, [(prefix + "/a", "file", b"bad")])
            with self.assertRaisesRegex(ValueError, "parent"):
                installer.install(bundle, checksum, destination)
            self.assertEqual(list(redirected.iterdir()), [])
            second_root = root / "second"
            bundle, checksum = self.bundle(root, [
                (prefix, "symlink", "showdown"), (prefix + "/a", "file", b"bad")])
            with self.assertRaisesRegex(ValueError, "parent"):
                installer.install(bundle, checksum, second_root)
            self.assertFalse(second_root.exists())

    def test_only_registered_absolute_symlinks_and_internal_links_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                (installer.PYTHON_ALIAS, "symlink", "/" + installer.PYTHON_BASE),
                (installer.RUNTIME + "/.venv/bin/python", "symlink", "/" + installer.PYTHON_ALIAS + "/bin/python3.13"),
                (installer.RUNTIME + "/foulplay-venv/bin/python", "symlink", "/usr/bin/python3.12"),
                (installer.RUNTIME + "/repo/link", "symlink", "target"),
            ]
            bundle, checksum = self.bundle(root, entries)
            destination = root / "worker"
            installer.install(bundle, checksum, destination)
            for name, _kind, value in entries:
                self.assertEqual(os.readlink(destination / name), value)
            for target in ("/etc/passwd", "../../../../../../etc/passwd"):
                bundle, checksum = self.bundle(root, [(installer.RUNTIME + "/repo/link", "symlink", target)])
                with self.assertRaises(ValueError):
                    installer.install(bundle, checksum, root / "bad")

    def test_hardlinks_must_target_an_archived_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = (installer.RUNTIME + "/repo/" + name for name in ("a", "b"))
            bundle, checksum = self.bundle(root, [(second, "hardlink", first), (first, "file", b"same")])
            destination = root / "worker"
            installer.install(bundle, checksum, destination)
            self.assertEqual((destination / first).stat().st_ino, (destination / second).stat().st_ino)
            self.assertEqual(installer.install(bundle, checksum, destination)["created"], 0)
            bundle, checksum = self.bundle(root, [(second, "hardlink", "etc/passwd")])
            with self.assertRaises(ValueError):
                installer.install(bundle, checksum, root / "bad")

    def test_system_root_requires_owned_linux_downward_api_identity(self):
        for platform, uid, environment in (("darwin", "uid", "uid"), ("linux", None, "uid"),
                                            ("linux", "uid", "different")):
            with self.subTest(platform=platform, uid=uid), \
                    patch.object(installer.sys, "platform", platform), \
                    patch.dict(installer.os.environ, POKEZERO_WORKER_POD_UID=environment):
                with self.assertRaisesRegex(ValueError, "owned Linux worker"):
                    installer.install("does-not-exist", "sha", "/", uid)


if __name__ == "__main__":
    unittest.main()
