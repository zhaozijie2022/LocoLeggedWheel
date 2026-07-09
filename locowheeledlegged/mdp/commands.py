from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg


# ----------------- Go2W -----------------
from isaaclab.managers import CommandTerm, CommandTermCfg

class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand




class UniformVelocityCommandMultiSampling(UniformVelocityCommand):
    """UniformVelocityCommand + multi-sampling ranges + initial zero command steps + binary maximal command."""

    def __init__(
        self, 
        cfg: UniformVelocityCommandMultiSamplingCfg, 
        env: "ManagerBasedEnv | ManagerBasedRLEnv"
    ):
        super().__init__(cfg, env)

        # prepare for 三桶采样
        p = float(self.cfg.new_command_probs)
        mid = 1.0 - 2.0 * p
        if mid < 0:
            raise ValueError(f"new_command_probs too large: {p}, must satisfy 1-2p >= 0.")
        self.sampling_probs = torch.tensor([p, mid, p], device=self.device, dtype=torch.float32)

        # command buffer + init-zero logic
        self.vel_command_b_buffer = torch.zeros_like(self.vel_command_b)  # (num_envs, 3) 记录采样出来的, 没经过 zero_mask 的
        self.initial_zero_command_steps = int(self.cfg.initial_zero_command_steps)

        # bang_bang control combos
        combos = []
        for i in [self.cfg.ranges.lin_vel_x[0], 0, self.cfg.ranges.lin_vel_x[1]]:
            for j in [self.cfg.ranges.lin_vel_y[0], 0, self.cfg.ranges.lin_vel_y[1]]:
                for k in [self.cfg.ranges.ang_vel_z[0], 0, self.cfg.ranges.ang_vel_z[1]]:
                    if i > 1e-2 or j > 1e-2 or k > 1e-2:
                        combos.append([i, j, k])  # 避免采样到 0 命令
        self.bang_bang_commands = torch.tensor(combos, device=self.device, dtype=torch.float32)

        
        self.metrics["tracking_x_ema"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_y_ema"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_z_ema"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["initial_zero_command_steps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rel_standing_envs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["bang_bang_envs"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        
        super()._update_metrics()  # 记录了 error_vel_xy, error_vel_yaw

        if hasattr(self, "_command_x_level"):
            self.metrics["tracking_x_ema"][:] = float(self._tracking_x_ema.mean().item())
            
        if hasattr(self, "_command_y_level"):
            self.metrics["tracking_y_ema"][:] = float(self._tracking_y_ema.mean().item())
            
        if hasattr(self, "_command_z_level"):
            self.metrics["tracking_z_ema"][:] = float(self._tracking_z_ema.mean().item())
            
            
        self.metrics["initial_zero_command_steps"][:] = float(self.initial_zero_command_steps)
        self.metrics["rel_standing_envs"][:] = float(self.cfg.rel_standing_envs)
        self.metrics["bang_bang_envs"][:] = float(self.cfg.bang_bang_envs)

        
        
    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        
        super()._resample_command(env_ids)

        # 一部分 bang-bang 采样，一部分分桶采样
        p_bang_bang = float(self.cfg.bang_bang_envs)
        use_bang_bang = torch.rand(env_ids.numel(), device=self.device) < p_bang_bang
        bang_ids = env_ids[use_bang_bang]
        bucket_ids = env_ids[~use_bang_bang]
        if bang_ids.numel() > 0:
            idx = torch.randint(0, self.bang_bang_commands.shape[0], (bang_ids.numel(),), device=self.device)
            self.vel_command_b[bang_ids] = self.bang_bang_commands[idx]
        if bucket_ids.numel() > 0:
            if hasattr(self, "_command_x_level"):
                self._sample_dim_with_bins(bucket_ids, dim=0, ranges=self.cfg.ranges.lin_vel_x)
            if hasattr(self, "_command_y_level"):
                self._sample_dim_with_bins(bucket_ids, dim=1, ranges=self.cfg.ranges.lin_vel_y)
            if hasattr(self, "_command_z_level"):
                self._sample_dim_with_bins(bucket_ids, dim=2, ranges=self.cfg.ranges.ang_vel_z)

        self.vel_command_b_buffer[env_ids] = self.vel_command_b[env_ids].clone()

        self._set_zero_command_for_beginning_steps()

    def _sample_dim_with_bins(self, env_ids: torch.Tensor, dim: int, ranges: tuple[float, float]):
        """
        根据 command level 与 previous level 动态构造三个采样区间并采样。
        ranges: (min, max)，即该维度的基础范围，如 self.cfg.ranges.lin_vel_x。
        属性由 curriculums 设置 _command_{x|y|z}_level、_previous__command_{x|y|z}_level
        """
        # 对于 level <= prev_level 的环境: 采样范围 [min_v*level, max_v*level]
        # 对于 level > prev_level 的环境: 三桶
        #   bin1 [min_v*lvl, min_v*prev_lvl],  p
        #   bin2 [min_v*prev_lvl, max_v*prev_lvl],  1-2p
        #   bin3 [max_v*prev_lvl, max_v*lvl], p
        min_v, max_v = ranges

        axis_name = "xyz"[dim]
        level = getattr(self, f"_command_{axis_name}_level")[env_ids]
        prev_level = getattr(self, f"_previous__command_{axis_name}_level")[env_ids]

        # case 1: level <= previous_level，单区间 [min_v*level, max_v*level]
        mask_same = level <= prev_level
        if mask_same.any():
            low = min_v * level[mask_same]
            high = max_v * level[mask_same]
            n = mask_same.sum().item()
            u = torch.rand(n, device=self.device, dtype=torch.float32)
            self.vel_command_b[env_ids[mask_same], dim] = low + u * (high - low)

        # case 2: level > previous_level，三桶 + multinomial
        mask_expand = level > prev_level
        if mask_expand.any():
            n = mask_expand.sum().item()
            u = torch.rand(n, device=self.device, dtype=torch.float32)

            # multinomial 分配
            bin_idx = torch.multinomial(self.sampling_probs, n, replacement=True)
            lvl = level[mask_expand]
            prev_lvl = prev_level[mask_expand]

            # bin 0: [min_v*lvl, min_v*prev_lvl], p
            mask = bin_idx == 0
            if mask.any():
                low = min_v * lvl[mask]
                high = min_v * prev_lvl[mask]
                uu = torch.rand(mask.sum().item(), device=self.device, dtype=torch.float32)
                self.vel_command_b[env_ids[mask_expand][mask], dim] = low + uu * (high - low)

            # bin 1: [min_v*pl, max_v*pl]，1-2p
            mask = bin_idx == 1
            if mask.any():
                low = min_v * prev_lvl[mask]
                high = max_v * prev_lvl[mask]
                uu = torch.rand(mask.sum().item(), device=self.device, dtype=torch.float32)
                self.vel_command_b[env_ids[mask_expand][mask], dim] = low + uu * (high - low)

            # bin 2: [max_v*pl, max_v*l]，p
            mask = bin_idx == 2
            if mask.any():
                low = max_v * prev_lvl[mask]
                high = max_v * lvl[mask]
                uu = torch.rand(mask.sum().item(), device=self.device, dtype=torch.float32)
                self.vel_command_b[env_ids[mask_expand][mask], dim] = low + uu * (high - low)

    def _update_command(self):
        
        self._set_zero_command_for_beginning_steps()
        self._recover_command_for_beginning_steps()
        super()._update_command()

    def _set_zero_command_for_beginning_steps(self):
        """
        对于刚起步的环境, 前 initial_zero_command_steps 步, cmd 始终为 0
        """
        if self.initial_zero_command_steps <= 0:
            return
        mask = self._env.episode_length_buf < self.initial_zero_command_steps
        if mask.any():
            ids = mask.nonzero(as_tuple=True)[0]
            self.vel_command_b[ids] = 0.0

    def _recover_command_for_beginning_steps(self):
        """
        在 initial_zero_command_steps 步后, 恢复 cmd 为 buffer 中的值
        """
        if self.initial_zero_command_steps <= 0:
            return
        mask = self._env.episode_length_buf == self.initial_zero_command_steps
        if mask.any():
            ids = mask.nonzero(as_tuple=True)[0]
            self.vel_command_b[ids] = self.vel_command_b_buffer[ids].clone()


@configclass
class UniformVelocityCommandMultiSamplingCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandMultiSampling

    new_command_probs: float = 0.15
    initial_zero_command_steps: int = 50 # episode最开始的xxx步数, cmd 始终为 0
    bang_bang_envs: float = 0.05  # 采用 bang-bang 控制的环境数量比例, 训练极限性能


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """




