# Third-party notices — Newton Robot Tasks

The scripts in `part3/` use two separately pinned robot sources. The rigid
migration and controller scratch model (IK and reBot gravity feedforward) download MJCF from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on
first use (a sparse git clone of just that robot's folder, checked out at a
pinned commit). The clean-table dynamic arm instead comes from the official
[Newton assets repository](https://github.com/newton-physics/newton-assets/tree/f8fb7abcbeba2318814a74f3eeb02780ad7925d6),
using the structured USD entries below. No USD failure silently falls back to MJCF.

The models are cached locally and are **not redistributed** in this repository;
downloads preserve each robot folder's upstream `LICENSE` and, for USD, its
referenced geometry/physics layers. A copy of the Apache License 2.0 is included at
[`../LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt), and the reBot asset's
MIT notice is retained at
[`../LICENSES/Seeed-Studio-MIT.txt`](../LICENSES/Seeed-Studio-MIT.txt).
Automatic caches are checked against their pinned Git revision and for changes
to tracked files in the selected robot folder. Explicit Menagerie path overrides
remain available for intentionally customized models.

## SO-101 (`robotstudio_so101`)

- MJCF source: <https://github.com/google-deepmind/mujoco_menagerie/tree/feadf76d42f8a2162426f7d226a3b539556b3bf5/robotstudio_so101>
- Pinned Menagerie commit: `feadf76d42f8a2162426f7d226a3b539556b3bf5`
- Official USD entry: `robotstudio_so101/usd_structured/so101.usda` at
  `newton-assets` commit `f8fb7abcbeba2318814a74f3eeb02780ad7925d6`.
- License: Apache-2.0 (see the robot folder's
  [LICENSE](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/robotstudio_so101/LICENSE)).
- Model of TheRobotStudio SO-ARM100 project's SO-101 arm, packaged for MuJoCo
  by the Menagerie maintainers.
- Local modification at load time only: the Newton scene builder raises the
  arm joint effort limits and the imported `mujoco:actuator_forcerange`
  attributes, matching Part 1's arm-only boost. Gripper limits are preserved.
  The clean-table scene uses the imported gripper geometry and adds local
  table, bin and payload geometry. Downloaded model files are used unmodified.

## Seeed reBot DevArm (`seeed_rebot_devarm`)

- MJCF source: <https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/seeed_rebot_devarm>
- Pinned Menagerie commit: `da76818e269b82289eba39808e2fb91d679d6994` (the upstream
  squash-merge commit of google-deepmind/mujoco_menagerie#300, which added
  this model)
- Official USD entry: `seeed_rebot_devarm/usd_structured/seeed_rebot_devarm.usda`
  at `newton-assets` commit `f8fb7abcbeba2318814a74f3eeb02780ad7925d6`.
- License: MIT, Copyright (c) 2026 Seeed Studio (see the robot folder's
  [LICENSE](https://github.com/newton-physics/newton-assets/blob/f8fb7abcbeba2318814a74f3eeb02780ad7925d6/seeed_rebot_devarm/LICENSE)).

The official structured robot USDs are MuJoCo-model conversions; Newton 1.6.0's
[release notes](https://github.com/newton-physics/newton/blob/v1.6.0/CHANGELOG.md)
identify the `mujoco-usd-converter` 0.5.0 update. The helper supplies both Mjc and
Newton schema resolvers, retains authored filters and mimic/equality metadata,
and does not edit downloaded files. Runtime target gains, initial poses, task
geometry and the SO-101 arm-only force policy are authored locally in the builder.
The repository-level `newton-assets` license does not replace per-robot notices.

For the clean-table task, SO-101 position-servo channels use kp/kd=300/30.
reBot arm channels use 400/40, while its finger slides retain the source model's
5000/41.28 gains. reBot also receives robot-only gravity feedforward through
bounded arm position targets. The SO-101 arm actuator range is raised to ±30,
while its gripper remains ±2.94. reBot retains ±36 for joints 1–3, ±14 for joints
4–6 and ±1904 for the finger slides. These settings are authored in memory;
official assets do not imply unchanged dynamics or a validated hardware
torque/force envelope.

## Scene boilerplate

The `<visual>` block and groundplane `<asset>` definitions in the generated
pick-and-place scene (`pick_place_common._scene_xml()`) are adapted from the
standard `scene.xml` files shipped with MuJoCo Menagerie (Apache-2.0).

## Newton and generated figures

Newton is an Apache-2.0 dependency, pinned to
[v1.6.0](https://github.com/newton-physics/newton/tree/v1.6.0), source commit
`c2ca70bec5998062b0b2d35869865cbe3a49feee`. The tutorial cites
its public docs and examples, including the experimental MuJoCo–VBD Proxy
coupling API. Upstream source and its license remain in the installed package.
The cited Newton documentation files carry CC-BY-4.0 and their own 2025 or 2026
copyright notices for The Newton Developers; source-code licensing and
documentation licensing are distinct.
The companion pins `newton-usd-schemas` 0.5.0 with its importer dependencies.

The clean-table bin, single-layer garment silhouette and rod
centerline are authored procedurally in this repository. The offline clean-table
renderer defaults to analytic collision geometry and an articulation skeleton.
Its optional GL mode replays recorded poses with the matching official robot
visual meshes, retaining the asset attribution above. Both modes render existing
states without advancing physics; downloaded mesh files are not redistributed.

The tutorial's `clean-table-so101.png` and `clean-table-rebot.png`, and the
accompanying GIFs, replay verified gripper pickups of both cubes, cloth and cable.
Their robot assets retain the SO-101 Apache-2.0 and reBot MIT notices above.
Each figure is backed by its matching simulation report and trajectory in
`assets/`; `clean-table-results.json` records their hashes and independent
trajectory audits. See [the verification record](VERIFICATION.md) for scope.

## Companion code attribution

The relocated companion code retains its existing MIT notices, copyright (c) 2026 Johnny Nuñez Cano. The full notice is retained in [Blog-companion-MIT.txt](../LICENSES/Blog-companion-MIT.txt). The frozen MuJoCo/MJWarp baselines under `../mujoco/` are included solely to make the migration checks self-contained.
