# LocoLeggedWheel: Legged-Wheeled Robot Reinforcement Learning Locomotion Control

**Legged-Wheeled Robot locomotion reinforcement learning project** based on NVIDIA Isaac Lab. If you are looking for the IsaacGym version, please refer to [DreamWaQ_Go2W](https://github.com/ShengqianChen/DreamWaQ_Go2W).
This project mainly optimizes **reward functions** and **training stability** for legged-wheeled robots, and provides sim2real deployment interfaces and configurations for the Unitree Go2W robot.

<div align="center">
  <p align="right">
    <span> 🌎 <a href="README.md"> English </a> | <a href="README_CN.md"> 中文 </a>
  </p>
</div>

🦾 Real Robot Demo
---
<p align="center">
  <img src="docs/demos/gross.gif" alt="" width="23%"/>
  <img src="docs/demos/stair.gif" alt="" width="23%"/>
  <img src="docs/demos/stone.gif" alt="" width="23%"/>
  <img src="docs/demos/unilateral-bridge.gif" alt="" width="23%"/>
</p>

🖥️ Simulation Training
---
+ Install Isaac Sim v5.5.1; For any issues, please refer to the [official Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)
  ```bash
  conda create -n env_isaaclab python=3.11
  conda activate env_isaaclab

  pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

  pip install -U torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

  # Check if Isaac Sim is correctly installed
  isaacsim
  ```
+ Install Isaac Lab v2.2.1 and lock to the specified commit. For any issues, please refer to the [official Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).
  ```bash
  git clone git@github.com:isaac-sim/IsaacLab.git
  cd IsaacLab
  git checkout c91a125c73

  # Verify commit
  git rev-parse HEAD
  # Expected output
  # c91a125c73c8b574878419a9583afc0b63b99f0a
  ```
+ Training and testing
  ```bash
  # Training
  python locoleggedwheel/scripts/train.py \
    --task Isaac-LocomotionGo2W-v1 \
    --num_envs=4096 \
    --headless

  # Testing
  python locoleggedwheel/scripts/play.py \
    --task Isaac-LocomotionGo2W-Play-v1 \
    --num_envs=20 \
    --load_run=xxxx \
    --checkpoint=xxxx
  ```

🚀 Real Deployment
---
+ Install [unitree_sdk2_python](https://support.unitree.com/home/en/Go2-W_developer/Python);
+ Align simulation and real parameters in `deploy_real/configs/go2w.yaml`. Pay special attention: the joint order is different between the simulator and real robot, so conversion is required;
+ The code supports two deployment modes (you may need to change network config):
  + Running on upper computer, send control commands through ethernet
  + Running onboard the robot itself
  ```bash
  # Train
  python deploy_real/deploy_real.py 
  python deploy_real/deploy_real.py <network> <config_filename>
  ```

✨ Core Features
---
+ **Low-pass filtering** on policy actions to prevent joint jitter on complex terrain.
  ```python
  class JointPositionLowPassAction(JointPositionAction):
    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._order = cfg.order
        self._weights = _compute_lowpass_weights(
            cfg.control_frequency,
            cfg.cut_off_frequency,
            cfg.order,
        )
        self._prev_model_output = torch.zeros_like(self._raw_actions)
        self._prev_prev_model_output = torch.zeros_like(self._raw_actions) if cfg.order >= 2 else None

    def process_actions(self, actions: torch.Tensor):
        if self._order == 1:
            filtered = self._weights[0] * actions + self._weights[1] * self._prev_model_output
        else:
            filtered = (
                self._weights[0] * actions
                + self._weights[1] * self._prev_model_output
                + self._weights[2] * self._prev_prev_model_output
            )

        if self._order >= 2:
            self._prev_prev_model_output[:] = self._prev_model_output.clone()
        self._prev_model_output[:] = actions.clone()
        super().process_actions(filtered)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_model_output[env_ids] = 0.0
        if self._prev_prev_model_output is not None:
            self._prev_prev_model_output[env_ids] = 0.0
        super().reset(env_ids)
  ```
+ Added **posture regularization** terms to constrain joint deviation from default posture, keeping robot standing naturally, especially keeping hip joints at the default angle;
  ```python
    def hip_deviation_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = hip_joint_names
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        q = asset.data.joint_pos[:, asset_cfg.joint_ids]
        q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return torch.sum(torch.square(q - q0), dim=1)

    def joint_deviation_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # params["asset_cfg"].joint_names = leg_joint_names
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        q = asset.data.joint_pos[:, asset_cfg.joint_ids]
        q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        return torch.sum(torch.square(q - q0), dim=1)
  ```
+ **Clipping for action_rate** to enhance training stability:
  ```python
  def custom_action_rate_l2_with_clip(
      env: ManagerBasedRLEnv,
      threshold: float = 7.0,
  ) -> torch.Tensor:
      delta_action = env.action_manager.action - env.action_manager.prev_action
      if torch.max(torch.abs(delta_action)) > threshold:
          print(f"[WARN] custom_action_rate_l2_with_clip: delta_action exceeds threshold {threshold}!")
          delta_action = torch.clamp(delta_action, min=-threshold, max=threshold)
      return torch.sum(torch.square(delta_action), dim=1)
  ```
+ Value clamping for RayCaster `ray_hits_w` to prevent IsaacLab NaN/Inf bugs
+ Double value clipping in the value function to improve PPO training stability
+ Set `last_action` observation history length to 1 to avoid control signal self-excitation

🙏 Acknowledgements
---
This project is based on [IsaacLab](https://github.com/isaac-sim/IsaacLab), [robot_lab](https://github.com/fan-ziqi/robot_lab), and [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym). We thank the authors for their contributions to the open source community.

