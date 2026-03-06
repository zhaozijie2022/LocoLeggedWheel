import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.joint2motor_idx = config["joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]

            self.default_sim_angles = np.array(config["default_sim_angles"], dtype=np.float32)
            self.default_real_angles = np.array(config["default_real_angles"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            
            
            # locomotion
            self.velocity_commands_scale = config['velocity_commands_scale']
            self.base_ang_vel_scale = config['base_ang_vel_scale']
            self.projected_gravity_scale = config['projected_gravity_scale']
            self.joint_pos_scale = config['joint_pos_scale']
            self.joint_vel_scale = config['joint_vel_scale']
            self.last_action_scale = config['last_action_scale']
            
            self.action_scale = config["action_scale"]
            self.wheel_action_scale = config['wheel_action_scale']
            
            self.history_length = config['history_length']
            self.actor_hidden_dims = config['actor_hidden_dims']

            self.fc_leg = config['fc_leg']
            self.fc_wheel = config['fc_wheel']
            self.fs = config['fs']

            
            



'''
control_dt: 控制周期, 单位为秒, 表示控制系统执行一次控制更新的时间间隔.
msg_type: 消息类型, 用于指定通信中使用的消息格式或协议类型.
imu_type: IMU类型, 表示使用的惯性测量单元的型号或种类.
weak_motor: 弱电机列表, 包含一些电机的索引或标识符, 可能用于标识性能较弱或需要特殊处理的电机.
lowcmd_topic: 低级命令主题, 用于指定在机器人控制系统中发布低级命令的通信主题名称.
lowstate_topic: 低级状态主题, 用于指定接收低级状态信息的通信主题名称.
policy_path: 策略路径, 表示存储机器人控制策略(如训练好的神经网络模型)的文件路径, 其中{LEGGED_GYM_ROOT_DIR}会被实际的路径替换.
leg_joint2motor_idx: 腿关节到电机索引的映射, 用于将腿关节的位置或指令映射到对应的电机索引.
kps: 位置控制的比例增益, 用于PID控制器中的比例部分, 影响关节位置控制的响应速度和精度.
kds: 位置控制的微分增益, 用于PID控制器中的微分部分, 有助于减少系统的振荡和提高稳定性.
default_angles: 默认关节角度, 表示机器人在初始状态或某些特定状态下各关节的目标角度.
arm_waist_joint2motor_idx: 手臂腰部关节到电机索引的映射, 用于将手臂腰部关节的位置或指令映射到对应的电机索引.
arm_waist_kps: 手臂腰部位置控制的比例增益, 用于控制手臂腰部关节的位置响应.
arm_waist_kds: 手臂腰部位置控制的微分增益, 用于提高手臂腰部关节控制的稳定性.
arm_waist_target: 手臂腰部目标角度, 表示手臂腰部关节在某些控制模式下的目标位置.
ang_vel_scale: 角速度缩放因子, 用于对角速度数据进行缩放, 可能用于归一化或调整数据范围.
dof_pos_scale: 自由度位置缩放因子, 用于对关节位置数据进行缩放.
dof_vel_scale: 自由度速度缩放因子, 用于对关节速度数据进行缩放.
action_scale: 动作缩放因子, 用于对控制动作进行缩放, 可能用于调整控制输入的范围或幅度.
cmd_scale: 命令缩放因子数组, 用于对不同的控制命令进行分别缩放.
max_cmd: 最大命令值数组, 表示各控制命令允许的最大值.
num_actions: 动作数量, 表示控制系统中需要输出的控制动作的维度或数量.
num_obs: 观测数量, 表示控制系统中用于状态观测的输入数据的维度或数量.
'''