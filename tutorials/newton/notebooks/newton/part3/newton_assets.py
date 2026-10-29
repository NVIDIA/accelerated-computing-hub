"""Official robot USDs, pinned independently of upstream main."""
import sys
from pathlib import Path

from newton.usd import SchemaResolverMjc, SchemaResolverNewton
from newton.utils import download_asset

from utils import validate_pinned_asset_cache

NEWTON_ASSETS_REF = "f8fb7abcbeba2318814a74f3eeb02780ad7925d6"
_ROBOT_USDS = {
    "so101": ("robotstudio_so101", "usd_structured/so101.usda"),
    "rebot": ("seeed_rebot_devarm", "usd_structured/seeed_rebot_devarm.usda"),
}


def pinned_robot_folder(folder: str) -> Path:
    """Reuse a verified shared asset, or fetch a separate pinned copy.

    An edited or differently pinned shared cache remains untouched. The
    replacement is verified too; download and validation errors propagate.
    """
    asset_path = Path(download_asset(folder, ref=NEWTON_ASSETS_REF))
    try:
        validate_pinned_asset_cache(
            asset_path.parent, folder, NEWTON_ASSETS_REF, cache_env="NEWTON_CACHE_PATH"
        )
        return asset_path
    except RuntimeError as exc:
        fallback = asset_path.parent.parent / f"verified_newton_assets_{NEWTON_ASSETS_REF[:12]}"
        print(f"Note: {exc} Using a separate verified copy in {fallback} instead.", file=sys.stderr)
    asset_path = Path(download_asset(folder, cache_dir=str(fallback), ref=NEWTON_ASSETS_REF))
    validate_pinned_asset_cache(
        asset_path.parent, folder, NEWTON_ASSETS_REF, cache_env="NEWTON_CACHE_PATH"
    )
    return asset_path


def add_robot_usd(builder, robot: str, *, xform=None) -> dict:
    """Append an official fixed-base robot, returning Newton's raw import maps.

    Register ``SolverMuJoCo.register_custom_attributes(builder)`` before adding
    anything to the builder. MuJoCo's USD resolver preserves authored passive
    damping; registering custom attributes alone is not enough. Authored actuators,
    mimic/equality constraints and collision filters are retained. The cache
    must match the pinned Git revision with unchanged
    tracked robot files. No downloaded asset is edited and import errors propagate.
    With usd-core 26.3, start Python with ``PXR_WORK_THREAD_LIMIT=1`` to avoid
    native parallel-parser crashes observed on macOS and Linux. This helper
    requires that limit on every platform and fails fast otherwise; it never
    changes process-wide concurrency settings.

    ``xform`` composes with the authored root pose. Fixed joints are kept, so
    the fixed base and welded links have their own ``path_body_map`` entries.
    Read actuated joints by label rather than assuming fixed array positions.
    These maps are build-time snapshots, not stable IDs after later restructuring.
    Set Newton's target-layout policy in the caller, before building the scene.
    """
    try:
        folder, entry = _ROBOT_USDS[robot]
    except KeyError:
        raise ValueError(f"Unknown robot {robot!r}; expected one of {tuple(_ROBOT_USDS)}") from None
    # A bare Newton builder already has mujoco equality attributes, so the
    # resolver's generic namespace check alone does not establish registration.
    if "mujoco:actuator_trnid" not in builder.custom_attributes:
        raise RuntimeError("Call SolverMuJoCo.register_custom_attributes(builder) before adding the robot USD")
    # OpenUSD 26.3's parallel physics parser can invalid-free rigid descriptors
    # on macOS and Linux. Require the mitigation for this version on every
    # platform; do not silently alter Work's process-wide thread limit or
    # discard physics/filter information.
    from pxr import Usd, Work

    if Usd.GetVersion() == (0, 26, 3) and Work.GetConcurrencyLimit() != 1:
        raise RuntimeError(
            "usd-core 26.3 needs PXR_WORK_THREAD_LIMIT=1 before starting Python "
            "for these USD imports (native parallel-parser crashes observed on macOS and Linux)"
        )
    source = pinned_robot_folder(folder) / entry
    return builder.add_usd(
        str(source),
        xform=xform,
        floating=False,
        # Preserve fixed links and their parent/body collision exclusions.
        collapse_fixed_joints=False,
        enable_self_collisions=True,
        convert_mjc_equality_constraints=True,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
    )
