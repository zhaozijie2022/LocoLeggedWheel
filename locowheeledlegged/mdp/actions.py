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

def _compute_lowpass_alpha(
        control_frequency: float,
        cut_off_frequency: float,
    ) -> float:
    """一阶低通(EMA/IIR)系数: alpha = 1 - exp(-2π f_c / f_s)。
    alpha 越大截止频率越高、平滑越弱; 50Hz 控制 / 5Hz 截止 → alpha ≈ 0.4665。
    order=2 时对同一 alpha 做两级级联
    """
    return 1.0 - math.exp(-2.0 * math.pi * cut_off_frequency / control_frequency)


class JointPositionLowPassAction(JointPositionAction):
    """对模型输出做一阶/二阶低通(IIR/EMA)平滑, 再交给父类做 scale/offset 等 process。

    order=1: y[t] = alpha*x[t] + (1-alpha)*y[t-1]
    order=2: 两级级联的一阶 EMA,
    历史项为**滤波后**输出 y[t-1](而非模型原始输出)
    alpha 由 control_frequency 与 cut_off_frequency 计算, 50Hz/5Hz 时 alpha≈0.4665
    """

    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._order = cfg.order
        self._alpha = _compute_lowpass_alpha(cfg.control_frequency, cfg.cut_off_frequency)
        # IIR 状态: 上一时刻各级滤波输出 y[t-1]
        self._filtered_1 = torch.zeros_like(self._raw_actions)
        self._filtered_2 = torch.zeros_like(self._raw_actions) if cfg.order >= 2 else None

    def process_actions(self, actions: torch.Tensor):
        a = self._alpha
        # 一级 EMA: y = a*x[t] + (1-a)*y[t-1]
        y = a * actions + (1.0 - a) * self._filtered_1
        self._filtered_1[:] = y
        if self._order >= 2:
            # 二级级联, 输入为一级输出
            y = a * y + (1.0 - a) * self._filtered_2
            self._filtered_2[:] = y
        super().process_actions(y)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._filtered_1[env_ids] = 0.0
        if self._filtered_2 is not None:
            self._filtered_2[env_ids] = 0.0
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
    """轮子速度控制的一阶/二阶低通(IIR/EMA)滤波，与 JointPositionLowPassAction 逻辑一致。

    对模型输出的速度 action 做 EMA 低通(历史项为滤波后输出 y[t-1])，
    再将滤波结果交给父类做 scale/offset 等 process。
    """

    def __init__(self, cfg: JointVelocityLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._order = cfg.order
        self._alpha = _compute_lowpass_alpha(cfg.control_frequency, cfg.cut_off_frequency)
        self._filtered_1 = torch.zeros_like(self._raw_actions)
        self._filtered_2 = torch.zeros_like(self._raw_actions) if cfg.order >= 2 else None

    def process_actions(self, actions: torch.Tensor):
        a = self._alpha
        y = a * actions + (1.0 - a) * self._filtered_1
        self._filtered_1[:] = y
        if self._order >= 2:
            y = a * y + (1.0 - a) * self._filtered_2
            self._filtered_2[:] = y
        super().process_actions(y)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._filtered_1[env_ids] = 0.0
        if self._filtered_2 is not None:
            self._filtered_2[env_ids] = 0.0
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

