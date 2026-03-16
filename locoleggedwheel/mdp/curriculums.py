from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, CurriculumTermCfg
import numpy as np
import math
import locoleggedwheel.mdp as mdp


from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter



def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_axis_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,  # track_lin_vel_x_exp
    range_multiplier: Sequence[float] = (0.1, 1.0),  # 课程起点和终点
    delta: float = 0.1,
    upper_threshold: float = 0.8,
    lower_threshold: float = 0.5,
    ema_alpha: float = 0.5,
) -> torch.Tensor:
    """根据跟踪奖励调整 x 速度命令范围; 每个 env 维护自己的 tracking_reward running average，课程用全 env 均值与 threshold 比较。"""
    base_velocity = env.command_manager.get_term("base_velocity")
    axis_name = reward_term_name.split("_")[-2]
    ema_name = f"_tracking_{axis_name}_ema"
    cmd_level_name = f"_command_{axis_name}_level"

    if env.common_step_counter == 0:
        # env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        # env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        # env._final_vel_x = env._original_vel_x * range_multiplier[1]
        # base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()

        # 记录一个num_envs大小的 ema, 对应每个 env 的速度追踪情况
        setattr(base_velocity, ema_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.float32))
        # 尝试分开每个环境的cmd等级, 初始均为 range_multiplier[0]
        setattr(base_velocity, cmd_level_name, torch.ones(env.num_envs, device=env.device, dtype=torch.float32) * range_multiplier[0])
        # 记录每个env的previous cmd等级, 用于cmd采样
        setattr(base_velocity, f"_previous_{cmd_level_name}", torch.ones(env.num_envs, device=env.device, dtype=torch.float32) * range_multiplier[0])

    ema = getattr(base_velocity, ema_name)
    cmd_level = getattr(base_velocity, cmd_level_name)
    previous_cmd_level = getattr(base_velocity, f"_previous_{cmd_level_name}")

    # ema_mean = ema.mean().item()
    cmd_level_mean = cmd_level.mean().item()

    if len(env_ids) > 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        # 刚结束的 episode 的实际时长 (秒)
        # actual_length_s = (env.episode_length_buf[env_ids].float() * env.step_dt).clamp(min=env.step_dt)
        # per_env_reward = (episode_sums[env_ids] / actual_length_s).float()

        # 根据episode最长时间计算, 摔倒活该
        per_env_reward = (episode_sums[env_ids] / env.cfg.episode_length_s).float()

        # 更新每个env的ema
        ema[env_ids] = (1.0 - ema_alpha) * ema[env_ids] + ema_alpha * per_env_reward
        previous_cmd_level[env_ids] = cmd_level[env_ids]

        # cmd_level move up or down
        cmd_level[env_ids] = torch.where(
            ema[env_ids] > (upper_threshold * reward_term_cfg.weight), 
            torch.clamp(cmd_level[env_ids] + delta, min=range_multiplier[0], max=range_multiplier[1]),
            cmd_level[env_ids]
        )
        cmd_level[env_ids] = torch.where(
            ema[env_ids] < (lower_threshold * reward_term_cfg.weight), 
            torch.clamp(cmd_level[env_ids] - delta, min=range_multiplier[0], max=range_multiplier[1]),
            cmd_level[env_ids]
        )

        # global_mean = ema.mean().item()

        # if global_mean > upper_threshold * reward_term_cfg.weight:
        #     delta_command = torch.tensor([-delta, delta], device=env.device)
        #     new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
        #     new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
        #     base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
        # elif global_mean < lower_threshold * reward_term_cfg.weight:
        #     delta_command = torch.tensor([delta, -delta], device=env.device)
        #     new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
        #     new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
        #     base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
    
    return torch.tensor(cmd_level_mean, device=env.device)