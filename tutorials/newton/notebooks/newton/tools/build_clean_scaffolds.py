#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Regenerate clean-table teaching scaffolds/snippets from verified solutions.

Maintenance only: --write overwrites the STARTER exercise files, never use it on
learner edits. --check is read-only and verifies the shipped exercises/snippets.
"""
from __future__ import annotations

import argparse
import ast
import textwrap
from pathlib import Path

PART = Path(__file__).resolve().parents[1] / "part3"
SOLUTIONS = PART / "solutions"


def function(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = source.splitlines(keepends=True)
            return node, lines[node.lineno - 1:node.end_lineno]
    raise ValueError(f"No function {name!r}")


def replace_function_body(source, name, body):
    node, _ = function(source, name)
    lines = source.splitlines(keepends=True)
    # Keep the signature, replace the complete body including its docstring.
    indent = " " * (node.col_offset + 4)
    replacement = [indent + line + "\n" if line else "\n" for line in body.splitlines()]
    lines[node.body[0].lineno - 1:node.end_lineno] = replacement
    return "".join(lines)


def outputs():
    scene = (SOLUTIONS / "clean_table_scene_solution.py").read_text()
    runner = (SOLUTIONS / "clean_the_table_solution.py").read_text()
    generated = {}
    exercises = (
        (0, "add_bin", "Build the five static box colliders and return Bin plus shape IDs."),
        (1, "add_garment", "Build a free shirt-shaped triangle mesh with add_cloth_mesh."),
        (2, "add_cable", "Build the rod and return its body, joint and shape IDs."),
        (3, "build_coupled_solver", "Assign robot/cubes to MuJoCo and cloth/cable to VBD; proxy both jaws and cubes."),
    )
    scaffold = scene
    for step, name, instruction in exercises:
        _, lines = function(scene, name)
        generated[SOLUTIONS / f"clean_step_{step:02d}_{name}.py"] = (
            "# SPDX-License-Identifier: MIT\n"
            f"# Step {step}: paste this function into clean_table_scene.py.\n"
            "# Imports and scene helpers are supplied by the exercise file.\n\n" + "".join(lines)
        )
        scaffold = replace_function_body(scaffold, name,
            f'"""{instruction}"""\n# TODO Step {step}: {instruction}\n'
            f'raise NotImplementedError("Complete TODO Step {step}: {name}")')
    # A starter lives in part3/, not solutions/. No sys.path mutation is needed.
    scaffold = scaffold.replace('sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n', '')
    scaffold = scaffold.replace('"""Gripper pick-and-place scene: MuJoCo robot/cubes, VBD free cloth/rod."""',
                                '"""Gripper pick-and-place exercise scene. Complete TODO Steps 0-3."""')
    generated[PART / "clean_table_scene.py"] = scaffold
    _, lines = function(runner, "simulate")
    generated[SOLUTIONS / "clean_step_04_simulate.py"] = (
        "# SPDX-License-Identifier: MIT\n# Step 4: paste this method into Example.\n\n" + textwrap.dedent("".join(lines))
    )
    runner_scaffold = runner.replace('from clean_table_scene_solution import ', 'from clean_table_scene import ')
    runner_scaffold = runner_scaffold.replace('sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n', '')
    runner_scaffold = replace_function_body(runner_scaffold, "simulate",
        '"""Advance one control frame through the coupled solver."""\n'
        '# TODO Step 4: clear external forces, detect contacts, step, synchronize joint coordinates, swap.\n'
        'raise NotImplementedError("Complete TODO Step 4: coupled substep loop")')
    generated[PART / "clean_the_table.py"] = runner_scaffold
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    different = []
    generated = outputs()
    for path, content in generated.items():
        if args.write:
            path.write_text(content)
        elif not path.exists() or path.read_text() != content:
            different.append(str(path.relative_to(PART)))
    if different:
        raise SystemExit("Stale teaching files: " + ", ".join(different))
    print(f"{'Generated' if args.write else 'Verified'} {len(generated)} clean-table teaching files")


if __name__ == "__main__":
    main()
