from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pokezero.audit_provenance import PUBLIC_REPO_COMMIT_ENV, public_repo_commit


class PublicRepoCommitTests(unittest.TestCase):
    def test_injected_full_revision_is_used_without_git_metadata(self) -> None:
        revision = "a" * 40
        with patch.dict(os.environ, {PUBLIC_REPO_COMMIT_ENV: revision}, clear=False):
            self.assertEqual(public_repo_commit(Path("/not-a-git-checkout")), revision)

    def test_invalid_injected_revision_is_not_recorded(self) -> None:
        with patch.dict(os.environ, {PUBLIC_REPO_COMMIT_ENV: "not-a-commit"}, clear=False):
            self.assertIsNone(public_repo_commit(Path("/not-a-git-checkout")))
