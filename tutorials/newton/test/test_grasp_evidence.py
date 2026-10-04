"""Synthetic measurement traces test task gates, never claim physical success."""
from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

PART3 = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
sys.path.insert(0, str(PART3))
from clean_table_task import Bin, GraspEvidence


def envelope(center):
    return np.asarray(center) + 0.02 * np.asarray(list(product((-1.0, 1.0), repeat=3)))


class ObservationTrace:
    """Clocked sensor-data fixture; no poses are passed off as a rollout."""
    def __init__(self, name="shirt"):
        self.frame = 0
        self.evidence = GraspEvidence(name, Bin(np.array([0., -.1, 0.]), np.array([.2, .1, .08])),
                                      .1, envelope([.1, -.2, .12]))

    def sample(self, phase, center=(.1, -.2, .12), **overrides):
        self.frame += 1
        measured = dict(phase=phase, frame=self.frame, time=self.frame / 50,
                        points=envelope(center), speed=0.01, jaw_contacts=[True, True],
                        jaw_normal_forces=[0.2, 0.2], open_fraction=0.1, hand_clear=True)
        measured.update(overrides)
        self.evidence.observe(**measured)

    def repeat(self, frames, phase, center=(.1, -.2, .12), **overrides):
        for _ in range(frames):
            self.sample(phase, center, **overrides)

    def grasp_and_lift(self):
        self.repeat(10, "close")
        self.repeat(25, "lift", (.1, -.2, .18))

    def arrive(self):
        self.grasp_and_lift()
        for y in np.linspace(-.18, 0., 10):
            self.sample("carry", (.1, y, .18))

    def release(self):
        self.sample("open", (.1, 0., .18), jaw_contacts=[False, False],
                    jaw_normal_forces=[0., 0.], open_fraction=0.9)

    def settle(self, frames=50, **overrides):
        measured = dict(jaw_contacts=[False, False], jaw_normal_forces=[0., 0.], open_fraction=0.9)
        measured.update(overrides)
        self.repeat(frames, "settle", (.1, 0., .04), **measured)


class GraspEvidenceTests(unittest.TestCase):
    def test_ordered_loaded_transport_and_release_accepts_all_payload_labels(self):
        for name in ("red_cube", "blue_cube", "shirt", "cable"):
            with self.subTest(name=name):
                trace = ObservationTrace(name)
                trace.arrive()
                trace.release()
                trace.settle()
                report = trace.evidence.report()
                self.assertTrue(report["success"])
                self.assertEqual(report["detached_settled_frames"], 50)
                self.assertGreaterEqual(report["min_lift_clearance_m"], .01)
                self.assertAlmostEqual(report["carry_distance_m"], .2)
                self.assertLess(report["grasp_time"], report["lift_time"])
                self.assertLess(report["carry_time"], report["release_command_time"])
                self.assertLessEqual(report["release_command_time"], report["release_time"])

    def test_containment_or_a_unilateral_touch_is_not_a_grasp(self):
        trace = ObservationTrace()
        trace.settle(60)
        self.assertFalse(trace.evidence.report()["success"])
        self.assertFalse(trace.evidence.report()["grasped"])
        trace = ObservationTrace()
        trace.repeat(10, "close", jaw_contacts=[True, False], jaw_normal_forces=[.2, 0.])
        trace.repeat(25, "lift", (.1, -.2, .18), jaw_contacts=[True, False], jaw_normal_forces=[.2, 0.])
        self.assertFalse(trace.evidence.report()["grasped"])
        self.assertFalse(trace.evidence.report()["lifted"])
        with self.assertRaisesRegex(ValueError, "release"):
            trace.release()

    def test_transient_lifts_do_not_accumulate_into_a_sustained_lift(self):
        trace = ObservationTrace()
        trace.repeat(10, "close")
        trace.repeat(24, "lift", (.1, -.2, .18))
        trace.sample("lift", (.1, -.2, .12))
        trace.repeat(24, "lift", (.1, -.2, .18))
        self.assertFalse(trace.evidence.report()["lifted"])
        self.assertEqual(trace.evidence.report()["full_lift_frames"], 24)

    def test_lost_loaded_contact_invalidates_a_previous_lift(self):
        trace = ObservationTrace()
        trace.grasp_and_lift()
        trace.sample("carry", (.1, -.1, .18), jaw_contacts=[True, False], jaw_normal_forces=[.2, 0.])
        trace.repeat(20, "carry", (.1, 0., .18))
        self.assertTrue(trace.evidence.report()["dropped_before_release"])
        self.assertFalse(trace.evidence.report()["success"])
        with self.assertRaisesRegex(ValueError, "release"):
            trace.release()

    def test_table_dragging_does_not_count_as_airborne_carry_distance(self):
        trace = ObservationTrace()
        trace.repeat(10, "close")
        for y in np.linspace(-.2, -.02, 20):
            trace.sample("lift", (.1, y, .12))
        trace.repeat(25, "lift", (.1, -.02, .18))
        for y in np.linspace(-.02, 0., 10):
            trace.sample("carry", (.1, y, .18))
        self.assertFalse(trace.evidence.report()["carried"])
        with self.assertRaisesRegex(ValueError, "release"):
            trace.release()

    def test_support_contact_resets_the_uninterrupted_carry_proof(self):
        trace = ObservationTrace()
        trace.arrive()
        self.assertTrue(trace.evidence.report()["carried"])
        trace.sample("hold", (.1, 0., .12))
        trace.repeat(30, "hold", (.1, 0., .18))
        self.assertFalse(trace.evidence.report()["carried"])
        with self.assertRaisesRegex(ValueError, "release"):
            trace.release()

    def test_a_prior_bin_arrival_cannot_authorize_release_elsewhere(self):
        trace = ObservationTrace()
        trace.arrive()
        trace.sample("hold", (.1, -.2, .18))
        with self.assertRaisesRegex(ValueError, "release"):
            trace.sample("open", (.1, -.2, .18), open_fraction=.9,
                         jaw_contacts=[False, False], jaw_normal_forces=[0., 0.])

    def test_carry_cannot_borrow_a_persistent_lift_from_before_a_table_touch(self):
        trace = ObservationTrace()
        trace.grasp_and_lift()
        trace.sample("carry", (.1, -.2, .12))
        for y in np.linspace(-.18, 0., 10):
            trace.sample("carry", (.1, y, .18))
        # The second airborne interval has only ten frames, despite the old
        # interval's 25-frame lift. It must earn its own persistent lift.
        self.assertFalse(trace.evidence.report()["lifted"])
        self.assertFalse(trace.evidence.report()["carried"])
        with self.assertRaisesRegex(ValueError, "release"):
            trace.release()

    def test_settling_requires_a_continuous_detached_still_clear_window(self):
        for interrupted in ({"jaw_contacts": [False, True]}, {"speed": .04}, {"hand_clear": False}):
            with self.subTest(interrupted=interrupted):
                trace = ObservationTrace()
                trace.arrive()
                trace.release()
                trace.settle(49)
                trace.settle(1, **interrupted)
                self.assertEqual(trace.evidence.report()["detached_settled_frames"], 0)
                trace.settle(49)
                self.assertFalse(trace.evidence.report()["success"])
                trace.settle(1)
                self.assertTrue(trace.evidence.report()["success"])

    def test_repeated_or_gapped_samples_cannot_manufacture_a_measured_window(self):
        trace = ObservationTrace()
        trace.repeat(9, "close")
        with self.assertRaises(ValueError):
            trace.sample("close", frame=9, time=.18)
        trace = ObservationTrace()
        trace.repeat(9, "close")
        try:
            trace.sample("close", frame=12, time=.24)
        except ValueError:
            pass  # Rejecting a missing observation is also safely fail-closed.
        else:
            self.assertFalse(trace.evidence.report()["grasped"])

    def test_malformed_observations_fail_before_updating_evidence(self):
        for changed in (
            {"time": float("nan")}, {"frame": True}, {"frame": -1},
            {"speed": -.01}, {"open_fraction": -1.}, {"open_fraction": 1.1},
            {"points": [[0., 0.]]}, {"points": [[0., 0., float("nan")]]},
            {"jaw_normal_forces": [-.2, .2]}, {"jaw_contacts": [float("nan"), 1]},
        ):
            with self.subTest(changed=changed):
                trace = ObservationTrace()
                with self.assertRaises(ValueError):
                    trace.sample("close", **changed)


class JawClearanceTests(unittest.TestCase):
    def test_clearance_uses_both_transformed_jaw_shapes_and_the_higher_surface(self):
        import newton
        import warp as wp
        from clean_table_task import CleanScene

        builder = newton.ModelBuilder()
        bodies, shapes = [], []
        for x in (-.1, .1):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0., .15), wp.quat_identity()))
            shape = builder.add_shape_box(body, hx=.04, hy=.01, hz=.01,
                                          xform=wp.transform(wp.vec3(0., 0., .005), wp.quat_identity()))
            bodies.append(body)
            shapes.append([shape])
        scene = object.__new__(CleanScene)
        scene.model = builder.finalize(device="cpu")
        scene.jaw_bodies, scene.jaw_shapes = tuple(bodies), tuple(shapes)
        scene.table_z = .1
        scene.bin = Bin(np.array([-.2, -.2, 0.]), np.array([.2, .2, .08]))
        state = scene.model.state()
        before = state.body_q.numpy().copy()
        # Shape-local offset raises the box center to .155; its bottom is .145.
        self.assertAlmostEqual(scene.tool_clearance(state), .045, places=6)
        # Rotating only the second jaw lowers its long edge to .11. A center or
        # grasp-site approximation would miss this loss of 2cm clearance.
        q = before.copy()
        q[bodies[1], 3:7] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), np.pi / 2))
        state.body_q.assign(q)
        self.assertAlmostEqual(scene.tool_clearance(state), .01, places=6)
        np.testing.assert_array_equal(state.body_q.numpy(), q)
        # Now the receiving rim, rather than the table, is the limiting plane.
        scene.bin = Bin(np.array([-.2, -.2, 0.]), np.array([.2, .2, .13]))
        self.assertAlmostEqual(scene.tool_clearance(state), -.02, places=6)


if __name__ == "__main__":
    unittest.main()
