import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


Go2W_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"locoleggedwheel/assets/go2w/urdf/go2w.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg( 
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=1.0e-9,
            rest_offset=-0.004,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,
            max_delay=15,
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            min_delay=0,
            max_delay=15,
        ),
    },
)



Go2W_PLAY_CFG = Go2W_CFG.copy()