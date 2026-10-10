"""Complete one Newton substep for the two-cube receiving-box exercise.

Run the completed reference first: box_newton_demo.py --task box --report ...
Then implement the method below. Scene construction, online commands, observers,
capacity diagnostics and strict physical validation remain shared with that run.
The earlier so101_newton.py scaffold is the optional stacking exercise.
"""
from box_newton_demo import BoxExample, main
from benchmark_box_protocol import DT


class Example(BoxExample):
    def integrate_substep(self, state, other):
        # TODO: clear this input state's external forces; call
        # self.solver.step(state, other, self.control, None, DT); return
        # (other, state) so the resulting state becomes the next input.
        # SolverMuJoCo detects contacts in this example. Never edit payload
        # coordinates or advance its internal MuJoCo backend directly.
        raise NotImplementedError("Complete integrate_substep in box_newton_exercise.py")


if __name__ == '__main__':
    main(example_type=Example)
