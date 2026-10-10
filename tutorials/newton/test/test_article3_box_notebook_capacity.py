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

ROOT = (Path(__file__).resolve().parents[1] / "notebooks")


def cells(path):
    return [''.join(cell['source']) for cell in json.loads(path.read_text())['cells'] if cell['cell_type'] == 'code']


class NewtonBoxNotebookCapacityTests(unittest.TestCase):
    def test_box_capacity_flags_and_pinned_source_preserve_legacy_defaults(self):
        for article, part, name, selector, primary, legacy in (
                (3, 'part3', '05__migration_benchmark.ipynb', 'SCOPE', 'newton-batch', 'replay'),):
            sources = cells(ROOT / 'newton' / part / name)
            setup = next(s for s in sources if 'REPO_URL =' in s)
            constants = {node.targets[0].id: ast.literal_eval(node.value)
                         for node in ast.parse(setup).body
                         if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id in {'REPO_URL', 'REF'}}
            self.assertEqual(constants['REPO_URL'], 'https://github.com/johnnynunez/accelerated-computing-hub.git')
            self.assertTrue(constants['REF'] == 'feature/blog3-validated-box-benchmark' or __import__('re').fullmatch(r'[0-9a-f]{40}', constants['REF']))
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
