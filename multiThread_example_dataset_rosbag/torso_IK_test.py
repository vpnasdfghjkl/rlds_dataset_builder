import sys
import os
import time
import math
# sys.path.append("/usr/lib/python3/dist-packages")
import numpy as np
import pydrake
import tensorflow_datasets as tfds
import scipy.spatial.transform as st
import rospy

from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.math import RollPitchYaw

from drake_ik import DrakeIK

from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser,PiecewisePolynomial
)
from pydrake.all import StartMeshcat, AddMultibodyPlantSceneGraph, MeshcatVisualizer

sys.path.append(os.path.abspath(os.path.dirname(__file__) + r'../../../'))
from common.IK import *
from common.utils import *

# from octo_deplay.utils.model_log import model_logger
# from leju_files.dynamic_biped.msg import robot_hand_eff  # 替换为你的包名和消息文件名


class ArmIk:
    def __init__(self,model_file, end_frames_name, use_drake_ik=False):
        builder = DiagramBuilder()
        self.__plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
        parser = Parser(self.__plant)
        robot = parser.AddModelFromFile(model_file)
        self.__plant.Finalize()

        # self.__visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        self.__diagram = builder.Build()
        self.__diagram_context = self.__diagram.CreateDefaultContext()

        self.__plant_context = self.__plant.GetMyContextFromRoot(self.__diagram_context)
        self.__q0 = self.__plant.GetPositions(self.__plant_context)
        self.__v0 = self.__plant.GetVelocities(self.__plant_context)
        self.__r0 = self.__plant.CalcCenterOfMassPositionInWorld(self.__plant_context)

        self.__base_link_name = end_frames_name[0]
        self.__left_eef_name = end_frames_name[1]
        self.__right_eef_name = end_frames_name[2]

        self.__IK = TorsoIK(self.__plant, end_frames_name, 1e-4, 1e-4)

        self.use_drake_ik = use_drake_ik
        self.drake_ik = DrakeIK(model_file)
    
    def q0(self):
        return self.__q0
    
    def get_cur_pos(self):
        joint_state = rospy.wait_for_message("/robot_arm_q_v_tau", robotArmInfo) 
        ret = self.chain.forward_kinematics(joint_state.q[0:7])
        return ret.pos, ret.rot_euler
    
    def init_state(self, torso_yaw_deg, torso_height):
        self.__torso_yaw_rad = math.radians(torso_yaw_deg)       
        self.__torso_height = torso_height       
        self.__q0[6] = torso_height

    def computeIK(self, q0, l_hand_pose, r_hand_pose, l_hand_RPY=None, r_hand_RPY=None):
            if self.use_drake_ik:
                is_success, q = self.drake_ik.solve(l_hand_pose, l_hand_RPY, q0)
                if not is_success:
                    # model_logger.log_info("SOLVER FAILED")
                    # model_logger.log_info(f"pose: {pose_list[0][0]}, {pose_list[0][1]}")
                    # model_logger.log_info(f"lhand: {l_hand_pose}, {l_hand_RPY}")
                    # model_logger.log_info(f"rhand: {pose_list[2][0]}, {pose_list[2][1]}")
                    # raise RuntimeError("Failed to IK0!")
                    return None
                else:
                    return q 
                
            torsoR = [0.0, self.__torso_yaw_rad, 0.0]
            r = [0.0, 0.0, self.__torso_height]
            
            pose_list = [
                [torsoR, r],
                [l_hand_RPY, l_hand_pose],
                [r_hand_RPY, r_hand_pose],
            ]
            is_success, q = self.__IK.solve(pose_list, q0=q0)
            if not is_success:
                # model_logger.log_info(f"pose: {pose_list[0][0]}, {pose_list[0][1]}")
                # model_logger.log_info(f"lhand: {pose_list[1][0]}, {pose_list[1][1]}")
                # model_logger.log_info(f"rhand: {pose_list[2][0]}, {pose_list[2][1]}")
                # raise RuntimeError("Failed to IK0!")
                return None
            else:
                return q 

    # def start_recording(self):
    #     # self.__visualizer.StartRecording()

    # def stop_andpublish_recording(self):
    #     self.__visualizer.StopRecording()
    #     self.__visualizer.PublishRecording()

    def visualize_animation(self, q_list, start_time=0.0, duration=1.1):
        t_sol = np.arange(start_time, start_time+duration, 1)  
        q_sol = np.array(q_list).T
        # model_logger.log_info(f"q_sol: {q_sol.shape}, t_sol: {t_sol.shape}")
        q_pp = PiecewisePolynomial.FirstOrderHold(t_sol, q_sol)
        t0 = t_sol[0]
        tf = t_sol[-1]
        t = t0
        # self.__visualizer.StartRecording()
        while t < tf:
            q = q_pp.value(t)
            self.__plant.SetPositions(self.__plant_context, q)
            self.__diagram_context.SetTime(t)
            self.__diagram.ForcedPublish(self.__diagram_context)
            t += 0.01
        # self.__visualizer.StopRecording()
        # self.__visualizer.PublishRecording()
        # while True:
        time.sleep(0.1)
    
    def left_hand_jacobian(self, q):
        self.__plant.SetPositions(self.__plant_context, q)
        J_hand_in_world = self.__plant.CalcJacobianSpatialVelocity(
            self.__plant_context, JacobianWrtVariable.kV,
            self.__plant.GetFrameByName(self.__left_eef_name), [0, 0, 0], self.__plant.world_frame(), self.__plant.world_frame())
        return J_hand_in_world

    def right_hand_jacobian(self, q):
        self.__plant.SetPositions(self.__plant_context, q)
        J_hand_in_world = self.__plant.CalcJacobianSpatialVelocity(
            self.__plant_context, JacobianWrtVariable.kV,
            self.__plant.GetFrameByName(self.__right_eef_name), [0, 0, 0], self.__plant.world_frame(), self.__plant.world_frame())
        return J_hand_in_world
    
    def left_hand_pose(self, q):
        self.__plant.SetPositions(self.__plant_context, q)
        l_hand_in_base = self.__plant.GetFrameByName(self.__left_eef_name).CalcPose(self.__plant_context, self.__plant.GetFrameByName(self.__base_link_name))
        # model_logger.log_info("left hand position in base:", l_hand_in_base.translation())
        return l_hand_in_base
    
    def right_hand_pose(self, q):
        self.__plant.SetPositions(self.__plant_context, q)
        r_hand_in_base = self.__plant.GetFrameByName(self.__right_eef_name).CalcPose(self.__plant_context, self.__plant.GetFrameByName(self.__base_link_name))
        # model_logger.log_info("right hand position in base:", r_hand_in_base.translation())
        return r_hand_in_base


def get_dataset_obs():
    # create RLDS dataset builder
    builder = tfds.builder_from_directory(
        builder_dir='/home/yu/mzs/rlds/example_dataset/1.0.0')
    ds = builder.as_dataset(split='train[:1]')

    episode = next(iter(ds))
    steps = list(episode['steps'])
    # images = [cv2.resize(np.array(step['observation']['third_camera_color']), (256, 256)) for step in steps]

    # eef_position = [np.array(step['observation']['state_eef_inverse_kinematics'][:3]) for step in steps]
    # eef_rotation = [np.array(step['observation']['state_eef_inverse_kinematics'][3:]) for step in steps]
    eef_position = [np.array(step['action']['command_eef_inverse_kinematics'][:3]) for step in steps]
    eef_rotation = [np.array(step['action']['command_eef_inverse_kinematics'][3:]) for step in steps]
    # eef_ = [np.array(step['action']['command_eef_inverse_kinematics'][:]) for step in steps]
    gripper_joint_position = [np.array([step['observation']['hand_position'][0]]) for step in steps]
    joint_state = [np.array(step['action']['command_position'][:7]) for step in steps]


    matrix = st.Rotation.from_rotvec(eef_rotation)
    qua = matrix.as_euler("xyz")
    proprio = np.concatenate(
    [np.array(eef_position), np.array(qua)],
    axis=1
)
    obs = {
        # 'images': np.array(images),
        'proprio': proprio,
    }
    
    np.save("rosbag_s", proprio)
    return obs, joint_state


if __name__ == "__main__":
    # import kinpy as kp

    # test, joint_state= get_dataset_obs()    

    np.set_printoptions(linewidth=240)
    np.set_printoptions(threshold=2000)
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)

    meshcat = StartMeshcat()
    model_file = "/home/lab/hx/rlds_dataset_builder/multiThread_example_dataset_rosbag/leju_files/biped_s4/urdf/biped_s4.urdf"

    base_link_name = 'torso'
    end_frames_name = [base_link_name,'l_hand_roll','l_hand_roll']

    arm_ik = ArmIk(model_file, end_frames_name)
    torso_yaw_deg = 0.0
    torso_height = 0.0
    arm_ik.init_state(torso_yaw_deg, torso_height)
    q0 = arm_ik.q0()
    q_list = [q0]
    last_q = q0
    # arm_ik.start_recording()
    t = 0.0
    l_pose = arm_ik.left_hand_pose(q0)
    eef_list = []
    kinpy_eef_list = []
    # model_logger.log_info(f"left_hand_pose: {l_pose.translation()}, {l_pose.rotation()}")
    for i in range(250):
        x = q0
        # x[7:14] = joint_state[i]
        x[7:14] = [1,2,3,4,5,6,7]
        l_pose = arm_ik.left_hand_pose(x)
        rpy = RollPitchYaw(l_pose.rotation())
        matrix = st.Rotation.from_euler('xyz', rpy).as_rotvec()

        
        var = np.concatenate([l_pose.translation(), rpy.vector()])
        eef_list.append(var)
        

        # l_hand_pose = [0.1, 0.3, 0.55]
    #     l_hand_pose = control_pos # 不计算就给None
    #     l_hand_RPY = test["proprio"][i][3:6]
    #     r_hand_RPY = None
    #     r_hand_RPY = None
    #     r_hand_pose = None
    #     time_0 = time.time()
    #     q = arm_ik.computeIK(q0, l_hand_pose, r_hand_pose, l_hand_RPY, r_hand_RPY)
    #     time_cost = time.time() - time_0
    #     model_logger.log_info(f"time cost: {1e3*time_cost:.3f} ms")
    #     model_logger.log_info(q)

    #     if q is not None:
    #         q_list.append(q)
    #         q0 = q
    #         # animate trajectory
    #         arm_ik.visualize_animation([last_q, q], t)
    #         last_q = q
    #         t = t + 1.0
    #     else:
    #         model_logger.log_info(f"Failed to IK in step {i}!")
    #     model_logger.log_info(f"i: {i}")
    # # arm_ik.stop_andpublish_recording()
    # np.save("rosbag_joint", eef_list)
    # model_logger.log_info('Program end, Press Ctrl + C to exit.')
    # while True:
    #     time.sleep(0.01)

