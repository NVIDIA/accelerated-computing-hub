"""Notebook routing and capacity contracts; these tests perform no physics."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def cells(path):
    return [''.join(cell['source']) for cell in json.loads(path.read_text())['cells'] if cell['cell_type'] == 'code']


class BoxTeachingNotebookRoutes(unittest.TestCase):
    def test_article2_box_is_default_and_failed_task_propagates(self):
        for part, name, reference_name, student_name in (
                ('part1', '02__pick_and_place.ipynb', 'so101_pick_place_solution.py', 'so101_pick_place.py'),
                ('part2', '03__mujoco_warp.ipynb', 'so101_mjwarp_solution.py', 'so101_mjwarp.py')):
            sources = cells(ROOT / 'notebooks' / 'mujoco' / part / name)
            setup = next(s for s in sources if 'SCRIPT =' in s)
            self.assertIn('REFERENCE = True', setup)
            assignment = next(n for n in ast.parse(setup).body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'SCRIPT' for t in n.targets))
            body = next(s for s in sources if 'for robot in ROBOTS:' in s and 'json.loads' in s)
            for reference in (True, False):
                for failure in (False, True):
                    with self.subTest(part=part, reference=reference, failure=failure), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        scope = {'PART1': root, 'PART2': root, 'REFERENCE': reference,
                                 'ROBOTS': ('so101', 'rebot'), 'OUTPUT': root, 'DEVICE': 'cuda:0',
                                 'sys': sys, 'subprocess': subprocess, 'json': json}
                        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'script-selector', 'exec'), scope)
                        expected = root / ('solutions/' + reference_name if reference else student_name)
                        self.assertEqual(scope['SCRIPT'], expected)
                        calls = []
                        def run(command, **kwargs):
                            self.assertEqual(command[:2], [sys.executable, str(expected)])
                            for flag, value in (('--task', 'box'), ('--headless-steps', '2000'), ('--sim-substeps', '20')):
                                self.assertEqual(command[command.index(flag)+1], value)
                            self.assertIs(kwargs['check'], True)
                            if part == 'part2':
                                self.assertEqual(command[command.index('--device')+1], 'cuda:0')
                            calls.append(command[command.index('--robot')+1])
                            Path(command[command.index('--report')+1]).write_text(json.dumps({'success': not failure}))
                            return subprocess.CompletedProcess(command, 0)
                        with mock.patch.object(subprocess, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                            if failure:
                                with self.assertRaises(AssertionError): exec(compile(body, 'box-run', 'exec'), scope)
                            else:
                                exec(compile(body, 'box-run', 'exec'), scope)
                                self.assertEqual(calls, ['so101', 'rebot'])

    def test_box_capacity_flags_and_pinned_source_preserve_legacy_defaults(self):
        for article, part, name, selector, primary, legacy in (
                (2, 'part2', '04__cpu_gpu_benchmark.ipynb', 'TASK', 'box', 'stack'),
                ):
            sources = cells(ROOT / 'notebooks' / 'mujoco' / part / name)
            setup = next(s for s in sources if 'REPO_URL =' in s)
            constants = {node.targets[0].id: ast.literal_eval(node.value)
                         for node in ast.parse(setup).body
                         if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id in {'REPO_URL', 'REF'}}
            self.assertEqual(constants['REPO_URL'], 'https://github.com/johnnynunez/accelerated-computing-hub.git')
            self.assertTrue(constants['REF'], 'Hosted source ref must be explicit')
            control = next(s for s in sources if 'RUN_BENCHMARK = False' in s)
            for value in (primary, legacy):
                changed = control.replace(f'{selector} = "{primary}"', f'{selector} = "{value}"')
                scope = {'os': os, 'PART2': ROOT, 'PART3': ROOT, 'BENCHMARK_PYTHON': Path(sys.executable),
                         'SCRIPT': ROOT / 'migration_benchmark.py', 'BENCHMARK_ENV': {}, 'subprocess': subprocess}
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(subprocess, 'run') as run:
                    exec(compile(changed, 'benchmark-settings', 'exec'), scope)
                    run.assert_not_called()
                command = scope['command']
                if value == primary:
                    self.assertEqual(command[command.index('--nconmax')+1], '64')
                    self.assertEqual(command[command.index('--njmax')+1], '128')
                    self.assertEqual(command[command.index('--timeout')+1], '7200')
                else:
                    self.assertNotIn('--nconmax', command)
                    self.assertNotIn('--njmax', command)
                    self.assertNotIn('--timeout', command)
