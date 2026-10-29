"""Checkout preservation and locked-kernel setup, without external commands."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "notebooks/mujoco"
SPEC = importlib.util.spec_from_file_location("article2_setup", ROOT / "notebook_setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)
SHA = "a" * 40


class HostedSetupTests(unittest.TestCase):
    def checkout(self, *, remote=setup.REPO_URL, head=SHA):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=head + "\n", stderr="")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(subprocess, "check_output", return_value=remote + "\n"), \
                patch.object(subprocess, "run", side_effect=run):
            setup.source_checkout(Path(directory), SHA)
        return calls

    def test_existing_other_remote_refused(self):
        with self.assertRaisesRegex(RuntimeError, "another repository"):
            self.checkout(remote="https://example.invalid/another.git")

    def test_existing_other_revision_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "existing files were preserved"):
            self.checkout(head="b" * 40)

    def test_real_checkout_with_other_revision_preserves_dirty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
            git('init', '-q')
            git('remote', 'add', 'origin', setup.REPO_URL)
            (checkout / 'lesson.py').write_text('original\n')
            git('add', 'lesson.py')
            git('-c', 'user.name=Setup test', '-c', 'user.email=setup@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
            head = git('rev-parse', 'HEAD')
            (checkout / 'lesson.py').write_text('user changes\n')
            with self.assertRaisesRegex(RuntimeError, 'existing files were preserved'):
                setup.source_checkout(checkout, SHA)
            self.assertEqual(git('rev-parse', 'HEAD'), head)
            self.assertEqual((checkout / 'lesson.py').read_text(), 'user changes\n')
            # The second Colab cell invocation discovers the local helper after
            # chdir. An explicit ref switch must still be checked against HEAD.
            helper = checkout / setup.TUTORIAL_RELATIVE / 'notebook_setup.py'
            with patch.object(setup, '__file__', str(helper)), \
                    patch.object(setup.importlib.util, 'find_spec', return_value=object()), \
                    patch.object(setup, 'prepare_lesson', return_value=helper.parent / 'part1'):
                with patch.dict(os.environ, {'HUB_SOURCE_REF': head}):
                    setup.open_notebook('part1')
                with patch.dict(os.environ, {'HUB_SOURCE_REF': SHA}):
                    with self.assertRaisesRegex(RuntimeError, 'existing files were preserved'):
                        setup.open_notebook('part1')
            self.assertEqual(git('rev-parse', 'HEAD'), head)
            self.assertEqual((checkout / 'lesson.py').read_text(), 'user changes\n')

    def test_matching_checkout_never_resets_or_checks_out(self):
        calls = self.checkout()
        self.assertFalse(any("reset" in c or "checkout" in c or "fetch" in c for c in calls))

    def test_unresolved_ref_refused(self):
        with self.assertRaises(ValueError):
            setup.source_checkout(Path("/not-created"), "main")

    def prepare(self, *, files=True, version=(3, 12), loaded=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lesson = root / setup.TUTORIAL_RELATIVE / "part1"
            lesson.mkdir(parents=True)
            if files:
                (lesson / "viewer_window.py").touch()
            previous = Path.cwd()
            physics = {k: sys.modules.pop(k) for k in ("mujoco", "mujoco_warp", "warp") if k in sys.modules}
            try:
                if loaded:
                    sys.modules["mujoco"] = object()
                with patch.dict(os.environ), patch.object(sys, "version_info", version), patch.object(subprocess, "run") as run:
                    setup.prepare_lesson(root, "part1", hosted=True)
                    return run.call_args.args[0]
            finally:
                os.chdir(previous)
                for path in (lesson, lesson.parent):
                    if str(path) in sys.path:
                        sys.path.remove(str(path))
                for k in ("mujoco", "mujoco_warp", "warp"):
                    sys.modules.pop(k, None)
                sys.modules.update(physics)

    def test_old_source_refused(self):
        with self.assertRaisesRegex(RuntimeError, "current box lessons"):
            self.prepare(files=False)

    def test_wrong_kernel_and_loaded_physics_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.12 kernel"):
            self.prepare(version=(3, 11))
        with self.assertRaisesRegex(RuntimeError, "Restart the runtime"):
            self.prepare(loaded=True)

    def test_installs_hash_lock(self):
        command = self.prepare()
        self.assertIn("--require-hashes", command)
        self.assertTrue(command[-1].endswith("requirements.lock.txt"))

    def test_notebook_bootstrap_uses_one_commit_for_helper_and_checkout(self):
        notebook = json.loads((ROOT / "part1/02__pick_and_place.ipynb").read_text())
        code = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'hub-source-setup')
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            google = types.ModuleType("google")
            google.colab = types.ModuleType("google.colab")
            def download(url, target):
                self.assertIn('/' + SHA + '/', url)
                Path(target).write_text('def open_notebook(part, *, source_commit):\n'
                                        '    return {"selected_commit": source_commit}\n')
            try:
                os.chdir(directory)
                with patch.dict(sys.modules, {"google": google, "google.colab": google.colab}), \
                        patch.dict(os.environ, {"HUB_SOURCE_REF": "refs/pull/286/head"}), \
                        patch.object(subprocess, 'check_output', return_value=SHA + '\trefs/pull/286/head\n'), \
                        patch('urllib.request.urlretrieve', side_effect=download):
                    scope = {}
                    exec(code, scope)
                    self.assertEqual(scope['selected_commit'], SHA)
                with patch.dict(sys.modules, {"google": google}), \
                        patch.object(subprocess, 'check_output', return_value=''), \
                        patch('urllib.request.urlretrieve') as download:
                    with self.assertRaisesRegex(RuntimeError, 'one upstream commit'):
                        exec(code, {})
                    download.assert_not_called()
                with patch.dict(sys.modules, {"google": google}), \
                        patch.dict(os.environ, {"HUB_SOURCE_REF": 'z' * 40}), \
                        patch('urllib.request.urlretrieve') as download:
                    with self.assertRaisesRegex(RuntimeError, 'full commit SHA'):
                        exec(code, {})
                    download.assert_not_called()
            finally:
                os.chdir(previous)

    def test_hosted_helper_passes_exact_commit_and_locked_kernel(self):
        with patch.object(setup.importlib.util, 'find_spec', return_value=object()), \
                patch.object(setup, 'source_checkout', return_value=ROOT.parents[3]) as checkout, \
                patch.object(setup, 'prepare_lesson', return_value=ROOT / 'part1') as prepare, \
                patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=SHA)):
            setup.open_notebook('part1', source_commit=SHA)
            checkout.assert_called_once_with(Path('/content/ach-mujoco-source'), SHA)
            self.assertTrue(prepare.call_args.kwargs['hosted'])

    def test_benchmark_installs_into_separate_python_with_strict_hashes(self):
        with patch.object(Path, 'is_file', return_value=False), patch.object(subprocess, 'run') as run:
            python = setup.benchmark_python(ROOT, hosted=True)
        self.assertEqual(python, Path('/content/mjwarp-benchmark-runtime/python312/bin/python'))
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(any('3.12' in command and 'venv' in command for command in commands))
        self.assertIn('--require-hashes', commands[-1])
        self.assertEqual(commands[-1][-1], str(ROOT / 'requirements.lock.txt'))
