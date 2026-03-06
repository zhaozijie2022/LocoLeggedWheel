 # LocoLeggedWheel: 轮腿机器人强化学习运动控制

基于 NVIDIA Isaac Lab 的 **Go2W 轮腿机器人 locomotion 强化学习项目**，该项目的IsaacGym 版本，请参考[DreamWaQ_Go2W](https://github.com/ShengqianChen/DreamWaQ_Go2W)。
本项目主要在针对轮腿机器人结构，对 **奖励函数** 与 **训练稳定性** 进行了优化，，并提供了在宇树 Unitree Go2W 上 **sim2real 部署** 的接口与配置。

<div align="center">
  <p align="right">
    <span> 🌎 <a href="README.md"> English </span> | <a href="README_CN.md"> 中文 </a>
  </p>
</div>


🦾 真机演示
---
<p align="center">
  <img src="docs/demos/gross.gif" alt="" width="23%"/>
  <img src="docs/demos/stair.gif" alt=“”" width="23%"/>
  <img src="docs/demos/stone.gif" alt="" width="23%"/>
  <img src="docs/demos/unilateral-bridge.gif" alt="" width="23%"/>
</p>



🖥️ 仿真训练
---
+ 安装 Issaac Sim v5.5.1，有问题请参考 [Isaac Sim 官方文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)
  ``` bash
  conda create -n env_isaaclab python=3.11
  conda activate env_isaaclab

  pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

  pip install -U torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

  # 检查 Isaac Sim 是否正常安装
  isaacsim
  ```
+ 安装 Isaac Lab v2.2.1，并固定 commit，有问题请参考 [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
  ``` bash
  git clone git@github.com:isaac-sim/IsaacLab.git
  cd IsaacLab
  git checkout c91a125c73

  # 验证 commit
  git rev-parse HEAD
  # 期望输出
  # c91a125c73c8b574878419a9583afc0b63b99f0a
  ```
+ 训练与测试
  ```bash
  # 训练
  python locoleggedwheel/scripts/train.py \
    --task Isaac-LocomotionGo2W-v1 \
    --num_envs=4096 \
    --headless

  # 测试
  python locoleggedwheel/scripts/play.py \
    --task Isaac-LocomotionGo2W-Play-v1 \
    --num_envs=20 \
    --load_run=xxxx \
    --checkpoint=xxxx
  ```

🚀 实体部署
---
+ 安装 [unitree_sdk2_python](https://support.unitree.com/home/zh/Go2-W_developer/Python)；
+ 在 `deploy_real/configs/go2w.yaml` 中对齐仿真训练参数，特别注意机器人关节顺序仿真器与实体中并不一致，需要转换；
+ 代码支持两种部署方式（需更改网卡名）
  + 上位机运行，通过网线发送控制指令
  + 机器人本体运行
运行示例：
```bash
# Train
python deploy_real/deploy_real.py 
python deploy_real/deploy_real.py <网卡名> <配置文件名>
```

✨ 核心特性
---
+ 对策略输出的动作进行**低通滤波**，防止机器人在复杂地形下的关节抖动
  ``` python
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
+ 增加**姿态正则项**，限制关节偏离默认姿态，使机器人保持自然站姿，尤其是令髋关节保持默认角度；
  ``` python
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
+ 对 `action_rate` 进行**幅值裁剪**，增强训练稳定性
  ``` python
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
+ 对 RayCaster 的 `ray_hits_w` 加入数值裁剪，防止 IsaacLab 出现bug
+ 对 value 进行了双重裁剪，提高 PPO 训练稳定性
+ 将 `last_action` 的观测历史长度设定为 1，避免出现控制信号的自激

🙏 致谢
---
本项目基于 [IsaacLab](https://github.com/isaac-sim/IsaacLab), [robot_lab](https://github.com/fan-ziqi/robot_lab) 和 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)，感谢这些项目作者对开源社区的贡献。











