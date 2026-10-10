"""Fake-model contracts for legacy solver options; no physics or devices."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parents[1]


def load_function(path, model):
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_pick_place_model')
    function = copy.deepcopy(function)
    function.returns = None
    for arg in function.args.args: arg.annotation = None
    fake = SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda path: model))
    scope = {'Path': Path, 'mujoco': fake, 'active_robot': lambda: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), scope)
    return scope['load_pick_place_model']


class Tests(unittest.TestCase):
    def exercise(self, robot, filename, ls, iterations):
        for part in ('part1', 'part2'):
            model = SimpleNamespace(nu=0, opt=SimpleNamespace(ls_iterations=ls, iterations=iterations, tolerance=1e-8))
            fn = load_function(HERE/'notebooks'/'mujoco'/part/'pick_place_common.py', model)
            result = fn(Path('/no-access-needed')/filename, SimpleNamespace(key=robot, arm_force_limit=None))
            self.assertIs(result, model)
            self.assertEqual(model.opt.iterations, iterations)
            self.assertEqual(model.opt.tolerance, 1e-8)
            yield model.opt.ls_iterations

    def test_so101_legacy_only_raises_20_to_50(self):
        self.assertEqual(list(self.exercise('so101', 'scene_pick_place.xml', 20, 10)), [50, 50])

    def test_rebot_existing_100_50_unchanged(self):
        self.assertEqual(list(self.exercise('rebot', 'scene_pick_place.xml', 50, 100)), [50, 50])

    def test_box_initial_options_unchanged_even_before_explicit_box_configuration(self):
        self.assertEqual(list(self.exercise('so101', 'scene_pick_place_box.xml', 20, 10)), [20, 20])

    def test_explicit_box_100_50_unchanged(self):
        self.assertEqual(list(self.exercise('so101', 'scene_pick_place_box.xml', 50, 100)), [50, 50])

    def test_larger_user_budget_not_reduced(self):
        self.assertEqual(list(self.exercise('so101', 'scene_pick_place.xml', 70, 10)), [70, 70])

    def test_unrelated_scenes_unchanged(self):
        self.assertEqual(list(self.exercise('so101', 'custom.xml', 20, 10)), [20, 20])

    def test_two_shared_copies_identical(self):
        self.assertEqual((HERE/'notebooks/mujoco/part1/pick_place_common.py').read_bytes(),
                         (HERE/'notebooks/mujoco/part2/pick_place_common.py').read_bytes())



if __name__ == '__main__': unittest.main()
