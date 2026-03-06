import gymnasium as gym
from .agents import rsl_rl_ppo_cfg
from . import locomotion_env_cfg

# ----------------------------------- Locomotion Go2W  -----------------------------------
gym.register(
    id="Isaac-LocomotionGo2W-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-LocomotionGo2W-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionPlayEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)
"""
Go2W 轮腿机器人运动控制

训练命令:
python locoleggedwheel/scripts/train.py --task Isaac-LocomotionGo2W-v1 --num_envs=4096 --headless

测试命令:
python locoleggedwheel/scripts/play.py --task Isaac-LocomotionGo2W-Play-v1 --num_envs=20 --load_run=xxxx --chekpoint=xxxx

"""


