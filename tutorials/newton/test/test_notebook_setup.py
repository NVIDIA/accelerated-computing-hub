"""Hosted checkout pins source while preserving a reader's existing work."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

PART3 = Path(__file__).resolve().parents[1] / "notebooks/newton/part3"
spec = importlib.util.spec_from_file_location("newton_notebook_setup", PART3 / "notebook_setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "origin"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    target = root / setup.TUTORIAL_RELATIVE / "part3"
    target.mkdir(parents=True)
    (target / "migration_benchmark.py").write_text("# first revision\n")
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "Initial fixture")
    return root, git(root, "rev-parse", "HEAD")


def test_fresh_checkout_resolves_exact_revision_and_reuses_it(source, tmp_path):
    origin, revision = source
    checkout = tmp_path / "checkout"
    assert setup.resolve_checkout(origin.as_uri(), revision, checkout) == checkout
    assert git(checkout, "rev-parse", "HEAD") == revision
    (checkout / "user-notes.txt").write_text("keep this\n")
    setup.resolve_checkout(origin.as_uri(), revision, checkout)
    assert (checkout / "user-notes.txt").read_text() == "keep this\n"


def test_checkout_change_requires_fresh_directory(source, tmp_path):
    origin, revision = source
    checkout = tmp_path / "checkout"
    setup.resolve_checkout(origin.as_uri(), revision, checkout)
    target = origin / setup.TUTORIAL_RELATIVE / "part3/migration_benchmark.py"
    target.write_text("# second revision\n")
    git(origin, "add", ".")
    git(origin, "-c", "commit.gpgsign=false", "commit", "-m", "Second fixture")
    with pytest.raises(RuntimeError, match="REF changed"):
        setup.resolve_checkout(origin.as_uri(), git(origin, "rev-parse", "HEAD"), checkout)
    assert git(checkout, "rev-parse", "HEAD") == revision
    assert (checkout / setup.TUTORIAL_RELATIVE / "part3/migration_benchmark.py").read_text() == "# first revision\n"


def test_edited_tracked_source_is_preserved(source, tmp_path):
    origin, revision = source
    checkout = tmp_path / "checkout"
    setup.resolve_checkout(origin.as_uri(), revision, checkout)
    target = checkout / setup.TUTORIAL_RELATIVE / "part3/migration_benchmark.py"
    target.write_text("# reader's exercise\n")
    with pytest.raises(RuntimeError, match="edited tracked files"):
        setup.resolve_checkout(origin.as_uri(), revision, checkout)
    assert target.read_text() == "# reader's exercise\n"


def test_another_remote_is_rejected_before_fetch(source, tmp_path):
    origin, revision = source
    checkout = tmp_path / "checkout"
    setup.resolve_checkout(origin.as_uri(), revision, checkout)
    with pytest.raises(RuntimeError, match="another repository"):
        setup.resolve_checkout((tmp_path / "unavailable").as_uri(), revision, checkout)
    assert git(checkout, "rev-parse", "HEAD") == revision
