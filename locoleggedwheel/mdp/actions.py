from __future__ import annotations
import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions import JointVelocityActionCfg
from isaaclab.envs.mdp.actions import JointVelocityAction


# region Low Pass Actions

def _compute_lowpass_weights(
    control_frequency: float,
    cut_off_frequency: float,
    order: int,
) -> list[float]:
    """根据控制频率和截止频率计算低通滤波权重（对模型原始输出做 FIR 平滑）。
    - alpha = 1.0 - exp(-2π*f_c/f_s)  
    - order=1: at = alpha*at + (1-alpha)*at-1
    - order=2: at = alpha²*at + 2*alpha*(1-alpha)*at-1 + (1-alpha)²*at-2
    50Hz 控制 / 5Hz 截止时：一阶 [0.47, 0.53]，二阶 [0.2209, 0.4982, 0.2809]
    """
    alpha = 1.0 - math.exp(-2.0 * math.pi * cut_off_frequency / control_frequency)
    if order == 1: 
        return [alpha, 1.0 - alpha]
    return [alpha * alpha, 2.0 * alpha * (1.0 - alpha), (1.0 - alpha) ** 2]


class JointPositionLowPassAction(JointPositionAction):
    """可配置阶数的低通滤波,仅对模型原始输出做平滑, at-1/at-2 均为模型输出非滤波结果

    order=1: at = w0*at + w1*at-1
    order=2: at = w0*at + w1*at-1 + w2*at-2
    权重由 control_frequency 与 cut_off_frequency 计算 50Hz/5Hz 时一阶 0.47/0.53, 二阶 0.2209/0.4982/0.2809
    """

    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._order = cfg.order
        self._weights = _compute_lowpass_weights(
            cfg.control_frequency,
            cfg.cut_off_frequency,
            cfg.order,
        )
        # 历史为模型原始输出（非滤波后）
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
        # 更新历史：at-2 <- at-1, at-1 <- 当前模型输出
        if self._order >= 2:
            self._prev_prev_model_output[:] = self._prev_model_output.clone()
        self._prev_model_output[:] = actions.clone()
        super().process_actions(filtered)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_model_output[env_ids] = 0.0
        if self._prev_prev_model_output is not None:
            self._prev_prev_model_output[env_ids] = 0.0
        super().reset(env_ids)


@configclass
class JointPositionLowPassActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = JointPositionLowPassAction
    control_frequency: float = 50.0  # Hz
    cut_off_frequency: float = 5.0   # Hz
    order: int = 1  # 1 或 2

    def __post_init__(self) -> None:
        assert self.order >= 1 and self.order <= 2, "order must be 1 or 2"


# region Velocity Low Pass

class JointVelocityLowPassAction(JointVelocityAction):
    """轮子速度控制的一阶/二阶低通滤波，与 JointPositionLowPassAction 逻辑一致。

    对模型输出的速度 action 先做 FIR 低通（at-1/at-2 为模型上一时刻/上上时刻输出），
    再将滤波结果交给父类做 scale/offset 等 process。
    """

    def __init__(self, cfg: JointVelocityLowPassActionCfg, env: ManagerBasedEnv):
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


@configclass
class JointVelocityLowPassActionCfg(JointVelocityActionCfg):
    class_type: type[ActionTerm] = JointVelocityLowPassAction
    control_frequency: float = 50.0  # Hz
    cut_off_frequency: float = 5.0   # Hz
    order: int = 1  # 1 或 2

    def __post_init__(self) -> None:
        assert self.order >= 1 and self.order <= 2, "order must be 1 or 2"

# endregion

