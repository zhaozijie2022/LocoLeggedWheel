from typing import Union
import numpy as np
import time
import os
import torch  


# region -- Unitree communication related imports --
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config



def trans_r2s(qj):

    tmp = qj.copy()
    
    tmp[0] = qj[3]  
    tmp[1] = qj[0] 
    tmp[2] = qj[9]  
    tmp[3] = qj[6]   

    tmp[4] = qj[4] 
    tmp[5] = qj[1] 
    tmp[6] = qj[10]
    tmp[7] = qj[7] 

    tmp[8] = qj[5] 
    tmp[9] = qj[2]
    tmp[10] = qj[11]
    tmp[11] = qj[8]  

    tmp[12] = qj[13] 
    tmp[13] = qj[12]
    tmp[14] = qj[15] 
    tmp[15] = qj[14] 

    return tmp

def trans_s2r(qj):

    tmp = qj.copy()
    
    tmp[0] = qj[1] 
    tmp[1] = qj[5]
    tmp[2] = qj[9] 
    tmp[12] = qj[13]

    tmp[3] = qj[0]  
    tmp[4] = qj[4]  
    tmp[5] = qj[8]
    tmp[13] = qj[12]

    tmp[6] = qj[3]
    tmp[7] = qj[7]   
    tmp[8] = qj[11] 
    tmp[14] = qj[15] 

    tmp[9] = qj[2]
    tmp[10] = qj[6]
    tmp[11] = qj[10]
    tmp[15] = qj[14]

    return tmp

class Controller:
    def __init__(self, config: Config) -> None:
        # region -- Initialization phase --
        # Load configuration
        self.config = config
        self.remote_controller = RemoteController()

        # Load model file
        from model_actor_critic import ActorCritic
        actor_critic = ActorCritic(
            # num_actor_obs=self.config.num_obs * self.config.history_length,
            num_actor_obs=262,
            num_actions=self.config.num_actions,
            actor_hidden_dims=self.config.actor_hidden_dims,
        )
        loaded_dict = torch.load(self.config.policy_path, weights_only=False, map_location=torch.device('cpu'))
        model_state_dict = {
            k.replace("actor.", ""): v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("actor")
        }
        actor_critic.actor.load_state_dict(model_state_dict)
        self.policy = actor_critic.actor.eval()

        # Process variable initialization
        self.qj = np.zeros(config.num_actions, dtype=np.float32)  # Joint position
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)  # Joint velocity
        self.action = np.zeros(config.num_actions, dtype=np.float32)  # Action
        self.last_action = np.zeros(16, dtype=np.float32)       # Last action (after filtering)
        self.cmd = np.array([0.0, 0.0, 0.0])  # Remote controller command
        self.counter = 0  # Unused

        self.raw_action = self.action.copy()  # Before filtering, direct model output
        self.last_raw_action = self.action.copy()
        
        
        self.history_len = self.config.history_length
        # Define dimensions for each term
        self.dims = {
            "cmd": 3,
            "ang_vel": 3,
            "gravity": 3,
            "pos": 16,  # Absolute joint positions
            "vel": 16,
            "action": 16
        }

        # Initialize buffer for each term (Shape: history_len, dim)
        self.hist_buffers = {
            k: np.zeros((self.history_len, v), dtype=np.float32) 
            for k, v in self.dims.items()
        }

        # DDS communication initialization
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # Wait for low-level state connection
        self.wait_for_low_state()

        init_cmd_go(self.low_cmd)
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()  

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandUp()
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    # region -- State callback handling --
    def LowStateHgHandler(self, msg: LowStateHG):
        """Handle H1 Gen2 low-level state message."""
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        """Handle Go2 low-level state message.""" 
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
    # endregion

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        """Send command (with automatic CRC)."""
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        """Wait until low-level state data is received."""
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # region -- State machine flow --
    def zero_torque_state(self):
        """
        Zero torque state (safe preparation phase).
        1. Send zero torque command.
        2. Wait for Start button trigger.
        """
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 1
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.joint2motor_idx
        kps = 70.0
        kds = 5.0
        default_pos = self.config.default_real_angles
        dof_size = len(dof_idx)
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 80
                self.low_cmd.motor_cmd[motor_idx].kd = 5
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # Leg joint control
            for i in range(12):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 70.0
                self.low_cmd.motor_cmd[motor_idx].kd = 5.0
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    # endregion

    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code

    def run(self):
        """Main control loop (executed each control period)."""
        t_start = time.time()
        self.counter += 1

        # 1. Get robot sensor data

        # cmd
        self.cmd[0] = self.remote_controller.ly  # Forward/backward velocity
        self.cmd[1] = self.remote_controller.lx * -1  # Lateral velocity
        self.cmd[2] = self.remote_controller.rx * -1  # Yaw velocity
        # ly, lx: left stick y/x; rx: right stick x; all in [-1, 1]
        # After velocity_commands_scale: [-2, 2], [-2, 2], [-0.25, 0.25]
        # Keeps commanded velocity within training range to avoid damage

        # Base angular velocity
        base_ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # Gravity vector (projected_gravity)
        quat = self.low_state.imu_state.quaternion  # Quaternion: w, x, y, z
        projected_gravity = get_gravity_orientation(quat)  # From IMU quaternion to gravity direction

        # Robot motor data: joint_pos & joint_vel
        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q  # Joint position (rad)
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq  # Joint velocity
        print("qj:", self.qj)

            
        # Extract current frame feature data
        curr_data = {}
        curr_data["cmd"] = self.cmd * self.config.velocity_commands_scale
        curr_data["ang_vel"] = base_ang_vel * self.config.base_ang_vel_scale
        curr_data["gravity"] = projected_gravity * self.config.projected_gravity_scale
        
        # Joint position error processing
        sim_qj = trans_r2s(self.qj) 
        print("sim_qj:", sim_qj)
        err_obs = (sim_qj - self.config.default_sim_angles) * self.config.joint_pos_scale
        err_obs[12:16] = 0.0  # Zero wheel positions
        curr_data["pos"] = err_obs
        print("err_obs:", err_obs)
        
        # Joint velocity processing
        sim_dqj = trans_r2s(self.dqj)
        curr_data["vel"] = sim_dqj * self.config.joint_vel_scale
        
        # Action processing
        # sim_action_all = trans_r2s(self.action)
        sim_action_all = self.action.copy()
        curr_data["action"] = sim_action_all * self.config.last_action_scale


        # Update history and concatenate vectors
        flat_obs_list = []
        for key in ["cmd", "ang_vel", "gravity", "pos", "vel", "action"]:
            # Update sliding window for this feature
            buf = self.hist_buffers[key]
            buf[:-1] = buf[1:]
            buf[-1] = curr_data[key]
            if key != "action":
                flat_obs_list.append(buf.flatten())
            else:
                flat_obs_list.append(buf[-1].flatten())

        obs_final = np.concatenate(flat_obs_list)

        obs_tensor = torch.from_numpy(obs_final).unsqueeze(0).float()
        with torch.no_grad():
            # Policy includes normalization (obs_normalizer)
            action_output = self.policy(obs_tensor).detach().cpu().numpy().squeeze()

        # self.action = trans_s2r(action_output)
        self.raw_action = action_output.copy()
        # self.action = action_output.copy()        
        
        alpha_leg = 1 - np.exp(-2 * np.pi * self.config.fc_leg / self.config.fs)
        alpha_wheel = 1 - np.exp(-2 * np.pi * self.config.fc_wheel / self.config.fs)
        
        for i in range(len(self.config.joint2motor_idx)):
            if i >= 12:
                motor_idx = self.config.joint2motor_idx[i]
                self.action[i] = alpha_wheel * self.raw_action[i] + (1 - alpha_wheel) * self.last_raw_action[i]
            else:
                motor_idx = self.config.joint2motor_idx[i]
                self.action[i] = alpha_leg * self.raw_action[i] + (1 - alpha_leg) * self.last_raw_action[i]
        
        self.last_action = self.action.copy()  # Update last action
        self.last_raw_action = self.raw_action.copy()
        
        print("curr_data", curr_data)
        print("obs:", obs_final)
        print("action:", self.action)

        for i in range(len(self.config.joint2motor_idx)):
            if i >= 12:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = 0.0
                
                if i == 12:
                    self.low_cmd.motor_cmd[motor_idx].dq = self.action[13] * self.config.wheel_action_scale
                elif i == 13:
                    self.low_cmd.motor_cmd[motor_idx].dq = self.action[12] * self.config.wheel_action_scale
                elif i == 14:
                    self.low_cmd.motor_cmd[motor_idx].dq = self.action[15] * self.config.wheel_action_scale
                elif i == 15:
                    self.low_cmd.motor_cmd[motor_idx].dq = self.action[14] * self.config.wheel_action_scale
                    
                # self.low_cmd.motor_cmd[motor_idx].dq = self.action[i] * self.config.wheel_action_scale
                self.low_cmd.motor_cmd[motor_idx].kp = 0.0
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            else:
                motor_idx = self.config.joint2motor_idx[i]
                    
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i] + self.action[i] * self.config.action_scale
                # self.low_cmd.motor_cmd[motor_idx].q = self.action[i] * self.config.action_scale
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
                
                print(i, self.config.action_scale, self.config.default_real_angles[i] + self.action[i] * self.config.action_scale)

        self.send_cmd(self.low_cmd)  # Send command
        
        t_cost = time.time() - t_start
        if t_cost < self.config.control_dt:
            time.sleep(self.config.control_dt - t_cost)  # Sleep for control interval


if __name__ == "__main__":
    # region -- Program entry --
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name", default="go2w.yaml")
    args = parser.parse_args()

    # Load config file
    config_path = f"./configs/{args.config}"
    config = Config(config_path)    

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    # Create controller instance
    controller = Controller(config)

    # region -- State machine execution flow --
    # Phase 1: Zero torque safe state
    controller.zero_torque_state()
    
    # Phase 2: Smooth move to default pose
    controller.move_to_default_pos()
    
    # Phase 3: Hold default pose
    controller.default_pos_state()
    
    print('RL Begin---------')
    # Phase 4: Main control loop
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    # Phase 5: Enter damping mode on exit
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
    # endregion
    # endregion