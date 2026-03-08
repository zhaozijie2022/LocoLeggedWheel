from __future__ import annotations
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    SceneEntityCfg,
    EventTermCfg,
    RewardTermCfg,
    ObservationTermCfg,
    ObservationGroupCfg,
    CurriculumTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import locoleggedwheel.mdp as mdp
import isaaclab.envs.mdp.rewards as isaaclab_rewards
import locoleggedwheel.mdp.rewards as custom_rewards
from locoleggedwheel.assets.go2w import Go2W_CFG as Robot_CFG
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import locoleggedwheel.terrains as custom_terrain_gen



BASE_LINK_NAME = "base"
FOOT_LINK_NAME = ".*_foot"
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
HIP_JOINT_NAMES = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]


# =============================================================================
# region -- Scene --
# =============================================================================
@configclass
class SceneCfg(InteractiveSceneCfg):
    replicate_physics = False
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.20), platform_width=2.0
                ),
                "perlin_rough": custom_terrain_gen.HfPerlinNoiseTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.10), noise_step=0.005,
                    frequency=0.7, octaves=2, lacunarity=2.0, persistence=0.5, border_width=0.25,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.05), noise_step=0.005, border_width=0.25,
                ),
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.2, step_height_range=(0.05, 0.23), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2, step_height_range=(0.05, 0.23), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
            },
            seed=1,
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_contact_senosr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(?!sensor.*).*",
        history_length=3,
        track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.6, 0.6, 0.6), intensity=1000.0),
    )
# endregion -- Scene --

# =============================================================================
# region -- Commands --
# =============================================================================

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandMultiSamplingCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        final_rel_standing_envs=0.1,
        initial_zero_command_steps=50,
        final_initial_zero_command_steps=50,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandMultiSamplingCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-math.pi / 6, math.pi / 6),
        ),
    )

# endregion -- Commands --

# =============================================================================
# region -- Observations --
# =============================================================================

@configclass
class ObservationsCfg:
    @configclass
    class NoisyProprioceptionCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            scale=1.0,
            params={"command_name": "base_velocity"},
            history_length=6,
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=6,
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            scale=1.0,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=6,
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel_without_wheel,
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=6,
            params={"wheel_asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=6,
        )
        last_action = ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            history_length=1,
        )
        base_lin_vel = None
        height_scan = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = None  # 这个设置了之后会对所有项一起生效

    @configclass
    class DenoisedProprioceptionCfg(NoisyProprioceptionCfg):
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        height_scan = ObservationTermCfg(
            func=mdp.custom_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = None

    policy: NoisyProprioceptionCfg = NoisyProprioceptionCfg()
    critic: DenoisedProprioceptionCfg = DenoisedProprioceptionCfg()

# endregion -- Observations --

# =============================================================================
# region -- Actions --
# =============================================================================

@configclass
class ActionsCfg:
    leg_joint_pos = mdp.JointPositionLowPassActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
        control_frequency=50.0,
        cut_off_frequency=5.0,
        order=1,
    )
    wheel_joint_vel = mdp.JointVelocityLowPassActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        control_frequency=50.0,
        cut_off_frequency=15.0,
        order=1,
    )

# endregion -- Actions --


# =============================================================================
# region -- Rewards --
# =============================================================================

@configclass
class RewardsCfg:
    # 速度追踪（isaaclab 自带）
    track_lin_vel_x_exp = RewardTermCfg(
        func=custom_rewards.custom_track_lin_vel_x_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_lin_vel_y_exp = RewardTermCfg(
        func=custom_rewards.custom_track_lin_vel_y_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=custom_rewards.custom_track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 姿态与角速度（isaaclab 自带）
    lin_vel_z_l2 = RewardTermCfg(func=isaaclab_rewards.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=isaaclab_rewards.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewardTermCfg(func=isaaclab_rewards.flat_orientation_l2, weight=-0.5)
    # 关节力矩/加速度（isaaclab 自带 joint_acc_l2）
    joint_torques_l2 = RewardTermCfg(
        func=isaaclab_rewards.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES)},
    )
    leg_joint_acc_l2 = RewardTermCfg(
        func=isaaclab_rewards.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    wheel_joint_acc_l2 = RewardTermCfg(
        func=isaaclab_rewards.joint_acc_l2,
        weight=-2.5e-9,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # 高度与接触（isaaclab 自带 base_height_l2, undesired_contacts）
    base_height_l2 = RewardTermCfg(
        func=custom_rewards.custom_base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.40,
        },
    )
    undesired_contacts = RewardTermCfg(
        func=isaaclab_rewards.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 0.1,
        },
    )
    action_rate_l2 = RewardTermCfg(
            func=custom_rewards.custom_action_rate_l2_with_clip,
            weight=-0.01,
            params={
                "threshold": 7.0,
            }
        )
    # locoleggedwheel/mdp/rewards.py
    stand_still_without_cmd = RewardTermCfg(
        func=custom_rewards.stand_still_without_cmd,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    hip_deviation_l2 = RewardTermCfg(
        func=custom_rewards.hip_deviation_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_NAMES)},
    )
    joint_deviation_l2 = RewardTermCfg(
        func=custom_rewards.joint_deviation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )


# =============================================================================
# region -- Events --
# =============================================================================

@configclass
class EventCfg:
    randomize_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "mass_distribution_params": (-1, 2),
            "operation": "add",
        },
    )
    randomize_foot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.5),
            "make_consistent": True,
            "num_buckets": 4000,
        },
    )
    randomize_rigid_body_inertia = EventTermCfg(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    randomize_com_positions = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "com_range": {
                "x": (-0.05, 0.05), 
                "y": (-0.05, 0.05), 
                "z": (-0.05, 0.05)
            },
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.35, 0.35),
            },
        },
    )
    randomize_reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0), 
            "velocity_range": (0.0, 0.0)
        },
    )
    randomize_apply_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.3, 0.3),
                "z": (-0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

# endregion -- Events --


# =============================================================================
# region -- Terminations --
# =============================================================================

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # base_orientation = TerminationTermCfg(
    #     func=mdp.bad_orientation,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": math.pi / 2},
    # )
    # base_height_below_minimum = TerminationTermCfg(
    #     func=mdp.root_height_below_minimum,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.15},
    # )
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact_senosr", body_names=[BASE_LINK_NAME]), "threshold": 1.0},
    )
    hip_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact_senosr", body_names=".*hip"), "threshold": 1.0},
    )
    terrain_out_of_bounds = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


# =============================================================================
# Curriculum
# =============================================================================

@configclass
class CurriculumCfg:
    velocity_commands = None


# =============================================================================
# LocomotionEnvCfg
# =============================================================================

def _smaller_scene_for_playing(env_cfg: "LocomotionEnvCfg") -> None:
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):

    base_link_name = BASE_LINK_NAME
    foot_link_name = FOOT_LINK_NAME
    leg_joint_names = LEG_JOINT_NAMES
    wheel_joint_names = WHEEL_JOINT_NAMES
    joint_names = JOINT_NAMES

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    viewer = ViewerCfg(
        eye=(5.0, 5.0, 4.0),
        resolution=(1920, 1080),
        lookat=(-2.0, -2.0, 0.0),
        origin_type="world",
        env_index=0,
        asset_name="robot",
    )
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        # sim.physics_material 需引用 scene.terrain, 在运行时设置
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # 传感器 update_period 依赖 self.sim.dt, 在运行时设置
        if getattr(self.scene, "robot_contact_senosr", None) is not None:
            self.scene.robot_contact_senosr.update_period = self.sim.dt
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt


        # ---------- Curriculum ----------
        self.curriculum.command_xy_levels = CurriculumTermCfg(
            func=mdp.command_xy_levels_vel,
            params={"reward_term_name": "track_lin_vel_xy_exp", "range_multiplier": (0.1, 1.0)},
        )
        self.curriculum.command_z_levels = CurriculumTermCfg(
            func=mdp.command_z_levels_vel,
            params={"reward_term_name": "track_ang_vel_z_exp", "range_multiplier": (0.1, 1.0)},
        )
        if self.scene.terrain.terrain_type == "generator":
            self.curriculum.terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)
            self.curriculum.command_xy_levels = None
            self.curriculum.command_z_levels = None

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if getattr(self.scene.terrain, "terrain_generator", None) is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if getattr(self.scene.terrain, "terrain_generator", None) is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self._disable_zero_weight_rewards()

    def _disable_zero_weight_rewards(self):
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and getattr(reward_attr, "weight", None) == 0:
                    setattr(self.rewards, attr, None)


@configclass
class LocomotionPlayEnvCfg(LocomotionEnvCfg):

    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _smaller_scene_for_playing(self)

        if self.scene.terrain.terrain_type != "generator":
            play_command_maximum_ranges = [
                self.commands.base_velocity.ranges.lin_vel_x[1],
                self.commands.base_velocity.ranges.lin_vel_y[1],
                self.commands.base_velocity.ranges.ang_vel_z[1],
            ]
            self.commands.base_velocity.ranges.lin_vel_x = (-play_command_maximum_ranges[0], play_command_maximum_ranges[0])
            self.commands.base_velocity.ranges.lin_vel_y = (-play_command_maximum_ranges[1], play_command_maximum_ranges[1])
            self.commands.base_velocity.ranges.ang_vel_z = (-play_command_maximum_ranges[2], play_command_maximum_ranges[2])
            self.commands.base_velocity.initial_zero_command_steps = self.commands.base_velocity.final_initial_zero_command_steps
            self.commands.base_velocity.rel_standing_envs = self.commands.base_velocity.final_rel_standing_envs
            if getattr(self, "curriculum", None) is not None:
                if getattr(self.curriculum, "command_xy_levels", None) is not None:
                    self.curriculum.command_xy_levels.params["range_multiplier"] = (1.0, 1.0)
                if getattr(self.curriculum, "command_z_levels", None) is not None:
                    self.curriculum.command_z_levels.params["range_multiplier"] = (1.0, 1.0)
