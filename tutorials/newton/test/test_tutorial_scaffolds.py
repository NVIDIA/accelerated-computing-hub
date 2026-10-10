from pathlib import Path
import unittest


ROOT = (Path(__file__).resolve().parents[1] / "notebooks")


class TutorialScaffoldTests(unittest.TestCase):
    def test_cpu_scaffold_applies_control_timestep(self) -> None:
        source = (ROOT / "mujoco/part1/so101_pick_place.py").read_text()

        calculation = source.index("sim_dt = frame_dt / sim_substeps")
        self.assertIn("model.opt.timestep = sim_dt", source)
        assignment = source.index("model.opt.timestep = sim_dt")
        simulation = source.index("def simulate_frame()")

        self.assertLess(calculation, assignment)
        self.assertLess(assignment, simulation)

    def test_mjwarp_scaffold_sets_timestep_before_model_upload(self) -> None:
        source = (ROOT / "mujoco/part2/so101_mjwarp.py").read_text()

        load = source.index("mjm = load_pick_place_model(xml_path, spec)")
        self.assertIn("mjm.opt.timestep = (1.0 / fps) / sim_substeps", source)
        assignment = source.index("mjm.opt.timestep = (1.0 / fps) / sim_substeps")
        upload_step = source.index("# TODO Step 1: upload the compiled model")

        self.assertLess(load, assignment)
        self.assertLess(assignment, upload_step)

    def test_mjwarp_scaffold_refreshes_host_kinematics_before_check(self) -> None:
        source = (ROOT / "mujoco/part2/so101_mjwarp.py").read_text()

        completion = source.index("def report_completion()")
        headless_branch = source.index("if headless_steps > 0:", completion)
        self.assertIn("report_completion()", source[headless_branch:])
        forward = source.index("mujoco.mj_forward(mjm, mjd)", completion)
        read_positions = source.index("red = mjd.xpos", completion)
        self.assertLess(read_positions, headless_branch)

        self.assertLess(forward, read_positions)


if __name__ == "__main__":
    unittest.main()
