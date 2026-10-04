# Step 9 — capture the substep loop into a CUDA graph.
#
# Same technique as Part 2, one level up the stack. In Part 2 we captured a
# single mjw.step; here we capture the whole frame — collide plus every
# substep plus the state swaps — as one replayable graph.
#
# Capture happens at the END of __init__, after the model, states, control and
# solver all exist. Capturing earlier would record operations against buffers
# that do not exist yet.
#
# Recording runs the Python scheduling code; the captured CUDA work runs on
# replay. Native MuJoCo-C is not captured by this example. CPU regression tests
# do not verify this CUDA path or its performance; test it on CUDA hardware.

self.graph = None
if self.model.device.is_cuda:
    with wp.ScopedCapture() as capture:
        self.simulate()
    self.graph = capture.graph
