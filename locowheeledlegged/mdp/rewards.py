from __future__ import annotations
import math
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_apply_inverse, euler_xyz_from_quat, quat_inv, quat_mul
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 需要修改 params["asset_cfg"].joint_names = leg_joint_names
    use_gravity_gating: bool = False,
    gating_max: float = 0.7,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    if use_gravity_gating:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, gating_max) / gating_max
    return reward


def hip_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return torch.sum(torch.square(q - q0), dim=1)


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = leg_joint_names
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return torch.sum(torch.square(q - q0), dim=1)


def hip_action_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    """Penalize hip joint actions (L2 squared)."""
    action = env.action_manager.action
    joint_ids = asset_cfg.joint_ids 

    reward = torch.sum(torch.square(action[:, joint_ids]), dim=1)
    return reward


def custom_track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-lin_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -(env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return reward

def custom_track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    reward = torch.exp(-lin_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -(env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return reward

def custom_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -(env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return reward


def custom_action_rate_l2_with_clip(
    env: ManagerBasedRLEnv,
    threshold: float = 7.0,
) -> torch.Tensor:


    delta_action = env.action_manager.action - env.action_manager.prev_action
    if torch.max(torch.abs(delta_action)) > threshold:
        print(f"[WARN] custom_action_rate_l2_with_clip: delta_action exceeds threshold {threshold}!")
        delta_action = torch.clamp(delta_action, min=-threshold, max=threshold)
    pen = torch.sum(torch.square(delta_action), dim=1)
    return pen

def custom_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    terrain_height_threshold: Tuple[float, float] = (-0.2, 0.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        base_ray_hits_w = sensor.data.ray_hits_w[..., 2]
        # Clamp base_ray_hits_w to avoid NaN and Inf (including -Inf/Inf) before usage
        base_ray_hits_w = torch.nan_to_num(base_ray_hits_w, nan=0.0, posinf=terrain_height_threshold[1], neginf=terrain_height_threshold[0])
        base_ray_hits_w = torch.clamp(base_ray_hits_w, min=terrain_height_threshold[0], max=terrain_height_threshold[1])
        adjusted_target_height = target_height + torch.mean(base_ray_hits_w, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)




