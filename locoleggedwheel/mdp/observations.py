from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply_inverse, quat_from_euler_xyz
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


from isaaclab.assets import Articulation

def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


import isaaclab.utils.math as math_utils
def base_lin_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The linear acceleration of the base link of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze()
    base_lin_acc_w = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids].squeeze()
    return math_utils.quat_apply_inverse(body_quat, base_lin_acc_w)


def custom_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    terrain_height_threshold: Tuple[float, float] = (-0.2, 0.2),
) -> torch.Tensor:
    """The height scan of the sensor."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    base_ray_hits_w = sensor.data.ray_hits_w[..., 2]
    # Clamp base_ray_hits_w to avoid NaN and Inf (including -Inf/Inf) before usage
    base_ray_hits_w = torch.nan_to_num(base_ray_hits_w, nan=0.0, posinf=terrain_height_threshold[1], neginf=terrain_height_threshold[0])
    base_ray_hits_w = torch.clamp(base_ray_hits_w, min=terrain_height_threshold[0], max=terrain_height_threshold[1])
    return sensor.data.pos_w[:, 2].unsqueeze(1) - base_ray_hits_w - offset



