"""Behavioral checks for the optional physical two-cube receiving-box task."""
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = (Path(__file__).resolve().parents[1] / "notebooks")
PART1 = ROOT / "newton/part3"
sys.path.insert(0, str(PART1))
from box_task import BoxTask, CubeEvidence, ReceivingBox, assert_mjwarp_capacity, native_contacts
from robots import get_robot


def points(center):
    return np.asarray(list(product((-0.022, 0.022), repeat=3))) + center


class Trace:
    """Synthetic measured samples exercise acceptance, not simulated physics."""

    def __init__(self, name="red_cube", start=0):
        self.frame = start
        self.box = ReceivingBox(np.array([-.1, .1, .006]), np.array([.1, .3, .2]))
        self.evidence = CubeEvidence(name, self.box, 0., points([0., -.15, .022]))

    def sample(self, phase, center, *, count=1, contacts=(1, 1), opening=0., speed=0.):
        for _ in range(count):
            self.frame += 1
            self.evidence.observe(frame=self.frame, time=self.frame / 50, phase=phase,
                points=points(center), speed=speed, counts=contacts,
                forces=np.asarray(contacts) * 5., opening=opening)

    def lifted(self):
        self.sample("close", [0., -.15, .022], count=10)
        self.sample("lift", [0., -.15, .08], count=25)
        return self

    def carried(self):
        self.lifted()
        for y in np.linspace(-.15, .2, 15):
            self.sample("carry", [0., y, .08])
        return self

    def released(self):
        self.carried()
        self.sample("open", [0., .2, .08], contacts=(0, 0), opening=1.)
        return self

    def complete(self):
        self.released()
        self.sample("settle", [0., .2, .028], count=50, contacts=(0, 0), opening=1.)
        return self


class BoxAcceptanceTests(unittest.TestCase):
    def test_sustained_physical_observation_sequence_is_accepted(self):
        report = Trace().complete().evidence.report()
        self.assertTrue(report["success"])
        self.assertGreaterEqual(report["full_lift_frames"], 25)
        self.assertGreater(report["carry_distance_m"], .05)
        self.assertLess(report["grasp_time"], report["lift_time"])
        self.assertLessEqual(report["carry_time"], report["release_command_time"])

    def test_good_carry_followed_by_a_drop_cannot_be_released_successfully(self):
        trace = Trace().carried()
        trace.sample("hold", [0., .2, .06], contacts=(1, 0))
        with self.assertRaisesRegex(ValueError, "retained whole-object carry"):
            trace.sample("open", [0., .2, .028], contacts=(0, 0), opening=1.)
        self.assertFalse(trace.evidence.report()["success"])

    def test_release_must_still_be_over_box(self):
        trace = Trace().carried()
        with self.assertRaisesRegex(ValueError, "retained whole-object carry"):
            trace.sample("open", [0., -.15, .08], contacts=(0, 0), opening=1.)

    def test_center_inside_does_not_hide_a_corner_outside(self):
        trace = Trace().released()
        trace.sample("settle", [.095, .2, .028], count=50, contacts=(0, 0), opening=1.)
        self.assertTrue(trace.box.contains(np.array([[.095, .2, .028]])))
        self.assertFalse(trace.evidence.report()["inside"])
        self.assertFalse(trace.evidence.report()["success"])

    def test_supported_transport_does_not_count_as_airborne_carry(self):
        trace = Trace()
        trace.sample("close", [0., -.15, .022], count=10)
        for y in np.linspace(-.15, .2, 15):
            trace.sample("carry", [0., y, .022])
        trace.sample("lift", [0., .2, .08], count=25)
        trace.sample("carry", [0., .2, .08], count=15)
        self.assertFalse(trace.evidence.carried)
        with self.assertRaises(ValueError):
            trace.sample("open", [0., .2, .08], contacts=(0, 0), opening=1.)

    def test_contact_with_support_requires_a_new_continuous_lift(self):
        trace = Trace().carried()
        trace.sample("carry", [0., .2, .022])
        for y in np.linspace(.1, .2, 10):
            trace.sample("carry", [0., y, .08])
        self.assertFalse(trace.evidence.lifted)
        self.assertFalse(trace.evidence.carried)
        with self.assertRaises(ValueError):
            trace.sample("open", [0., .2, .08], contacts=(0, 0), opening=1.)

    def test_unilateral_contacts_and_short_lifts_are_insufficient(self):
        trace = Trace()
        trace.sample("close", [0., -.15, .022], count=20, contacts=(1, 0))
        trace.sample("lift", [0., -.15, .08], count=30, contacts=(1, 0))
        self.assertFalse(trace.evidence.grasped)
        trace = Trace()
        trace.sample("close", [0., -.15, .022], count=10)
        trace.sample("lift", [0., -.15, .08], count=24)
        self.assertFalse(trace.evidence.lifted)

    def test_open_command_without_measured_opening_is_not_release(self):
        trace = Trace().carried()
        trace.sample("open", [0., .2, .08], count=50, contacts=(0, 0), opening=.79)
        self.assertTrue(trace.evidence.release_commanded)
        self.assertFalse(trace.evidence.released)

    def test_duplicate_samples_cannot_manufacture_settling(self):
        trace = Trace().released()
        trace.sample("settle", [0., .2, .028], contacts=(0, 0), opening=1.)
        with self.assertRaisesRegex(ValueError, "nonconsecutive"):
            trace.evidence.observe(frame=trace.frame, time=trace.frame / 50, phase="settle",
                points=points([0., .2, .028]), speed=0., counts=[0, 0], forces=[0., 0.], opening=1.)

    def test_settling_must_be_consecutive_and_detached(self):
        trace = Trace().released()
        trace.sample("settle", [0., .2, .028], count=49, contacts=(0, 0), opening=1.)
        trace.sample("settle", [0., .2, .028], contacts=(1, 0), opening=1.)
        trace.sample("settle", [0., .2, .028], count=49, contacts=(0, 0), opening=1.)
        self.assertFalse(trace.evidence.report()["success"])

    def test_missing_second_pickup_and_overlapping_pickups_are_rejected(self):
        task = BoxTask(get_robot("so101"))
        task.phase, task.tool_clearance = "done", .1
        task.evidence = {"red_cube": Trace().complete().evidence}
        self.assertFalse(task.report()["success"])
        task.evidence["blue_cube"] = Trace("blue_cube").complete().evidence
        self.assertFalse(task.report()["success"], "A single gripper cannot carry both accepted cycles simultaneously")

    def test_final_jaw_withdrawal_is_required(self):
        task = BoxTask(get_robot("so101"))
        task.evidence = {"red_cube": Trace().complete().evidence,
                         "blue_cube": Trace("blue_cube", start=150).complete().evidence}
        task.phase, task.tool_clearance = "done", .019
        self.assertFalse(task.report()["success"])
        task.tool_clearance = .021
        self.assertTrue(task.report()["success"])


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


class BoxCapacityTests(unittest.TestCase):
    def state(self, contacts=8, constraints=(16,)):
        return SimpleNamespace(nacon=Array([contacts]), nefc=Array(constraints), naconmax=8, njmax=16)

    def test_exact_capacity_is_valid_and_raw_overshoots_are_rejected(self):
        self.assertEqual(assert_mjwarp_capacity(self.state())["contact_count"], 8)
        for state in (self.state(contacts=9), self.state(constraints=(17,)), self.state(constraints=(16, 17))):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "capacity overflow"):
                assert_mjwarp_capacity(state)

    def test_newer_sticky_capacity_flags_cannot_hide_behind_zero_counts(self):
        fake_mjw = SimpleNamespace(OverflowType=SimpleNamespace(ITERATIONS=512, LS_ITERATIONS=1024))
        with patch.dict(sys.modules, {"mujoco_warp": fake_mjw}):
            state = self.state(contacts=0, constraints=(0,))
            state.overflow = Array([512 | 1024])
            self.assertEqual(assert_mjwarp_capacity(state)["overflow_flags"], 1536)
            state.overflow = Array([1024 | 1])
            with self.assertRaisesRegex(ValueError, "capacity overflow"):
                assert_mjwarp_capacity(state)




if __name__ == "__main__":
    unittest.main()
