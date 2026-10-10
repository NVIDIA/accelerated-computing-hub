# Third-party notices — Article 2

The scripts in `part1/` and `part2/` download robot models from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on
first use (a sparse git clone of just that robot's folder, checked out at a
pinned commit). Normal runs cache the models locally. Retained diagnostic archives also preserve
the exact model assets used in their checks; every robot folder carries its
own upstream `LICENSE` file, which the download and archives preserve. A copy of the Apache License 2.0 is included at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## SO-101 (`robotstudio_so101`)

- Source: <https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101>
- Pinned commit: `feadf76d42f8a2162426f7d226a3b539556b3bf5`
- License: Apache-2.0 (see the `LICENSE` file inside the robot folder)
- Model of TheRobotStudio SO-ARM100 project's SO-101 arm, packaged for MuJoCo
  by the Menagerie maintainers.
- Local modification at load time only: `load_pick_place_model()` raises the
  arm `actuator_forcerange` (see `pick_place_common.py`); the downloaded model
  files are used unmodified.

## Seeed reBot DevArm (`seeed_rebot_devarm`)

- Source: <https://github.com/google-deepmind/mujoco_menagerie/tree/main/seeed_rebot_devarm>
- Pinned commit: `da76818e269b82289eba39808e2fb91d679d6994` (the upstream
  squash-merge commit of google-deepmind/mujoco_menagerie#300, which added
  this model)
- License: MIT, Copyright (c) 2026 Seeed Studio (see the `LICENSE` file inside
  the robot folder)

## Scene boilerplate

The `<visual>` block and groundplane `<asset>` definitions in the generated
pick-and-place scene (`pick_place_common._scene_xml()`) are adapted from the
standard `scene.xml` files shipped with MuJoCo Menagerie (Apache-2.0).
