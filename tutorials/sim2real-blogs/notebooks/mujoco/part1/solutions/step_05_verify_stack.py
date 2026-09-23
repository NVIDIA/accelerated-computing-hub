# Step 5 — verify the stack.
#
# data.xpos holds world-frame body positions after the last forward pass. A
# successful stack means the red cube sits directly above the blue one:
#   * xy_err  -> horizontal distance between the two cube centers (want ~0)
#   * dz      -> vertical gap (want ~2 * CUBE_HALF = 0.044 m)

red = data.xpos[red_body]
blue = data.xpos[blue_body]
xy_err = float(np.linalg.norm(red[:2] - blue[:2]))
dz = float(red[2] - blue[2])
print(f"Headless complete. red={np.round(red, 3)} blue={np.round(blue, 3)}")
print(f"stack check: xy_err={xy_err:.3f} m  dz={dz:.3f} m")
