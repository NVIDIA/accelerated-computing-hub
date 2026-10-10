# Step 7 — capture the step into a CUDA graph.
#
# mjw.step issues dozens of small kernel launches. Each launch has a fixed CPU
# cost, and at small-to-medium world counts that overhead can dominate the
# actual physics work. A CUDA graph records the whole launch sequence once and
# replays it as a single submission.
#
# Rules to remember:
#   * Capture AFTER the state is initialised and after any Model field
#     overrides (domain randomisation), because the graph freezes the
#     operations, not the data pointers you rebind later.
#   * Replaying reuses the same buffers, so write new controls into the SAME
#     arrays rather than reassigning d.ctrl to a fresh array.
#   * Graph capture is CUDA-only, hence the device check and CPU fallback.

graph = None
if device.is_cuda:
    with wp.ScopedCapture() as capture:
        mjw.step(m, d)
    graph = capture.graph


def advance() -> None:
    if graph is not None:
        wp.capture_launch(graph)
    else:
        mjw.step(m, d)
