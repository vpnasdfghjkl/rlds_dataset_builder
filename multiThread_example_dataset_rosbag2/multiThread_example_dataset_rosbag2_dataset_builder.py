from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from multiThread_example_dataset.conversion_utils import MultiThreadedDatasetBuilder

from PIL import Image
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True
import re,os


import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# from cv_bridge import CvBridge


import sys
import os
import time
import math
# sys.path.append("/usr/lib/python3/dist-packages")
import numpy as np
# import pydrake
import tensorflow_datasets as tfds
import scipy.spatial.transform as st
from sensor_msgs.msg import CompressedImage
from scipy.spatial.transform import Rotation as R
import shutil
import math
# bridge = CvBridge()

# # 创建一个 VideoWriter 对象
# # 参数依次为：输出文件名、编码方式、帧率、图像大小
# output_video = 'output_video.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者使用其他编码方式如 'MJPG', 'MP4V', 'X264'
# frame_rate = 30  # 可以根据需要调整帧率
# img_size = (640, 480)  # 根据你的图像大小调整，例如 cv_img.shape[1], cv_img.shape[0]
# video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, img_size)


# export CUDA_VISIBLE_DEVICES=
CAM_HZ=30
TRAIN_HZ=10
TASK_TIME=1000
bag_folder_name="2024-9-7"
bag_folder_path="/IL/rlds_dataset_builder/multiThread_example_dataset_rosbag2/bag/"+bag_folder_name
save_plt_folder = f"{bag_folder_path}/plt"
save_lastPic_folder=f"{bag_folder_path}/last_pic"

def check_folder(CHECK_PIC_SAVE_FOLDER):
    if not os.path.exists(CHECK_PIC_SAVE_FOLDER):
        os.makedirs(CHECK_PIC_SAVE_FOLDER)
    else:
        # 清空文件夹中的所有内容
        shutil.rmtree(CHECK_PIC_SAVE_FOLDER)
        os.makedirs(CHECK_PIC_SAVE_FOLDER)
def adjust_pose_rpy(pose):
    threshold=3
    pre_eef = pose[0][3:6]  
    for i in range(len(pose)):
        for j in range(3): 
            diff = pose[i][3+j] - pre_eef[j]
            if diff > threshold:
                pose[i][3+j] -= 2 * math.pi
            elif diff < -threshold:
                pose[i][3+j] += 2 * math.pi
            pre_eef[j] = pose[i][3+j]

def use_rosbag_to_show(bag_name):
    # bridge = CvBridge()
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    # 读取rosbag文件并提取所需数据
    bag = rosbag.Bag(bag_name, 'r')

    start_time = bag.get_start_time()
    end_time = start_time + TASK_TIME

    cmd_joint=[]
    cmd_joint_time_stamp=[]
    state_joint=[]
    state_joint_time_stamp=[]

    cmd_eef_pose=[]
    cmd_eef_pose_time_stamp=[]
    state_eef_pose=[]
    state_eef_pose_time_stamp=[]

    cmd_hand=[]
    cmd_hand_time_stamp=[]
    state_hand=[]
    state_hand_time_stamp=[]

    img=[]
    img_stamp=[]
    
    for topic, msg, t in bag.read_messages(topics=[ '/kuavo_arm_traj',\
                                                    '/robot_arm_q_v_tau',\
                                                    '/drake_ik/cmd_arm_hand_pose',\
                                                    '/drake_ik/real_arm_hand_pose', \
                                                    '/robot_hand_eff',\
                                                    '/robot_hand_position',\
                                                    '/head_camera/color/image_raw/compressed',\
                                                  ]):
        # msg_time = msg.header.stamp.to_sec()  # 将时间戳转换为秒
        # if msg_time > end_time:
        #     break  # 超过时间限制，停止读取
        
        if topic == '/kuavo_arm_traj':
            # cmd_joint.append(msg.position)
            cmd_joint_time_stamp.append(msg.header.stamp)
            cmd_joint.append(np.radians(msg.position)[:7])

        elif topic == '/robot_arm_q_v_tau':
            # 将弧度转换为角度
            # state_joint.append(np.rad2deg(msg.q))
            state_joint_time_stamp.append(msg.header.stamp)
            state_joint.append((msg.q)[:7])
            
        elif topic=='/drake_ik/cmd_arm_hand_pose':
            cmd_eef_pose_time_stamp.append(msg.header.stamp)
            xyz=np.array(msg.left_pose.pos_xyz)
            xyzw=np.array(msg.left_pose.quat_xyzw)
            rotation = R.from_quat(xyzw)
            # 转换为欧拉角 (默认是 'xyz' 顺序，单位是弧度)
            euler_angles = rotation.as_euler('xyz')
            xyzrpy=np.concatenate((xyz,euler_angles))
            cmd_eef_pose.append(xyzrpy)

        elif topic=='/drake_ik/real_arm_hand_pose':
            state_eef_pose_time_stamp.append(msg.header.stamp)
            xyz=np.array(msg.left_pose.pos_xyz)
            xyzw=np.array(msg.left_pose.quat_xyzw)
            rotation = R.from_quat(xyzw)
            # 转换为欧拉角 (默认是 'xyz' 顺序，单位是弧度)
            euler_angles = rotation.as_euler('xyz')
            xyzrpy=np.concatenate((xyz,euler_angles))
            state_eef_pose.append(xyzrpy)

        elif topic=='/robot_hand_eff':
            cmd_hand_time_stamp.append(msg.header.stamp)
            left_hand_pose=msg.data
            if left_hand_pose[-1]==0:
                grip=0
            elif left_hand_pose[-1]==90:
                grip=1
            else:
                print("hand pose error")
            cmd_hand.append(grip)

        elif topic=='/robot_hand_position':
            state_hand_time_stamp.append(msg.header.stamp)

            left_hand_pose=msg.left_hand_position
            if left_hand_pose[-1]==0:
                grip=0
            elif left_hand_pose[-1]==90:
                grip=1
            else:
                print("hand pose error")
            state_hand.append(grip)

        elif topic=='/head_camera/color/image_raw/compressed':
            img_stamp.append(msg.header.stamp)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 使用 cv2 解码图像
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_img = cv2.resize(cv_img, (256, 256))
            # 将调整后的图像添加到 img 列表
            img.append(resized_img)
            # cv2.imshow("s",cv_img)
            # cv2.waitKey(1)

    # cmd_eef_pose_time_stamp=cmd_joint_time_stamp.copy()
    # state_eef_pose_time_stamp=state_joint_time_stamp.copy()

    bag.close()
    adjust_pose_rpy(cmd_eef_pose)
    adjust_pose_rpy(state_eef_pose)
    # 安全判断
    if len(cmd_joint) == 0 or len(state_joint) == 0:
        print("ROS bag file contains empty data for at least one topic.")
        return

    if len(cmd_joint) < 100 or len(state_joint) < 100:
        print("ROS bag file data count is too small (less than 100 data points). Please check again.")
        return
    

    aligned_state_joint = []
    aligned_cmd_joint = []
    aligned_state_hand=[]
    aligned_cmd_hand=[]
    aligned_cmd_eef_pose=[]
    aligned_state_eef_pose=[]
    
    drop=2
    img=img[drop:-drop]
    img_stamp=img_stamp[drop:-drop]
    assert len(img)==len(img_stamp)
    for stamp in img_stamp:
        stamp_sec=stamp.to_sec()
        idx_s = np.argmin(np.abs(np.array([t.to_sec() for t in state_joint_time_stamp]) - stamp_sec))
        aligned_state_joint.append(state_joint[idx_s])

        idx_a = np.argmin(np.abs(np.array([t.to_sec() for t in cmd_joint_time_stamp]) - stamp_sec))
        aligned_cmd_joint.append(cmd_joint[idx_a])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in state_hand_time_stamp]) - stamp_sec))
        aligned_state_hand.append(state_hand[idx_h])
        
        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in cmd_hand_time_stamp]) - stamp_sec))
        aligned_cmd_hand.append(cmd_hand[idx_h])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in cmd_eef_pose_time_stamp]) - stamp_sec))
        aligned_cmd_eef_pose.append(cmd_eef_pose[idx_h])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in state_eef_pose_time_stamp]) - stamp_sec))
        aligned_state_eef_pose.append(state_eef_pose[idx_h])
    

    aligned_cmd_joint = [list(item) for item in aligned_cmd_joint]
    aligned_state_joint = [list(item) for item in aligned_state_joint]
    aligned_cmd_eef_pose=[list(item) for item in aligned_cmd_eef_pose]
    aligned_state_eef_pose=[list(item) for item in aligned_state_eef_pose]

    print("all length==============>:\nimg_stamp,aligned_cmd_joint,aligned_state_joint,aligned_cmd_eef_pose,aligned_state_eef_pose,aligned_cmd_hand,aligned_state_hand")
    print(len(img_stamp),len(aligned_cmd_joint),len(aligned_state_joint),len(aligned_cmd_eef_pose),len(aligned_state_eef_pose),len(aligned_cmd_hand),len(aligned_state_hand))
    assert len(img_stamp)==len(aligned_cmd_joint)==len(aligned_state_joint)==len(aligned_cmd_eef_pose)==len(aligned_state_eef_pose)==len(aligned_cmd_hand)==len(aligned_state_hand)
    
    for i in range(len(img_stamp)):
        aligned_cmd_joint[i].append(aligned_cmd_hand[i])
        aligned_state_joint[i].append(aligned_state_hand[i])
        aligned_cmd_eef_pose[i].append(aligned_cmd_hand[i])
        aligned_state_eef_pose[i].append(aligned_state_hand[i])

 # s 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 # a 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    jump=CAM_HZ//TRAIN_HZ
    aligned_cmd_joint=np.array(aligned_cmd_joint)[::jump].astype(np.float32)
    aligned_state_joint=np.array(aligned_state_joint)[::jump].astype(np.float32)
    aligned_cmd_eef_pose=np.array(aligned_cmd_eef_pose)[::jump].astype(np.float32)
    aligned_state_eef_pose=np.array(aligned_state_eef_pose)[::jump].astype(np.float32)
    aligned_delta_cmd_eef_pose=None
    img=img[::jump]

    print("after jump, all length==============>:")
    print(len(img),len(aligned_state_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint))
    assert len(img)==len(aligned_state_eef_pose)==len(aligned_cmd_eef_pose)==len(aligned_state_joint)==len(aligned_cmd_joint)
    import matplotlib
    matplotlib.use('Agg')
    aligned_cmd_joint=aligned_cmd_joint[1:]
    aligned_state_joint=aligned_state_joint[1:]

    aligned_state_eef_pose=aligned_state_eef_pose[1:]
    aligned_delta_cmd_eef_pose=aligned_cmd_eef_pose[1:]-aligned_cmd_eef_pose[:-1]
    aligned_delta_cmd_eef_pose[:,6]=aligned_cmd_eef_pose[1:,6]
    aligned_cmd_eef_pose=aligned_cmd_eef_pose[1:]
    img=img[1:]
    
    print("after delete firet frame==============>:")
    print(len(img),len(aligned_state_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint),len(aligned_delta_cmd_eef_pose),len(aligned_cmd_eef_pose))

    # 创建3行5列的图表并进行比较
    num_plots = min(len(aligned_cmd_eef_pose[0]), len(aligned_state_eef_pose[0]), 15)  # 限制最多只显示15个数据对比
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(base_name, fontsize=16)
    for i in range(num_plots):
        # kuavo_position = [data[i] for data in aligned_cmd_joint]
        # robot_q = [data[i] for data in aligned_state_joint]

        cmd_eef=[data[i] for data in aligned_cmd_eef_pose]
        state_eef=[data[i] for data in aligned_state_eef_pose]
        cmd_eef_delta=[data[i] for data in aligned_delta_cmd_eef_pose]
        row = i // 5
        col = i % 5
        # axs[row, col].plot(kuavo_position, label='/kuavo_arm_traj')
        # axs[row, col].plot(robot_q, label='/robot_arm_q_v_tau')
        axs[row, col].plot(cmd_eef, label='/cmd_eef')
        axs[row, col].plot(state_eef, label='/state_eef')
        axs[row, col].plot(cmd_eef_delta, label='/cmd_eef_delta')
        axs[row, col].set_title(f"motor {i+1} state")
        axs[row, col].legend()

    exampl_index=50
    print(f"example index {exampl_index}:")
    print(" cmd_joint:",aligned_cmd_joint[exampl_index],
          "\n state_joint:",aligned_state_joint[exampl_index],
          "\n aligned_delta_cmd_eef_pose:",aligned_delta_cmd_eef_pose[exampl_index],
          "\n state_eef:",aligned_state_eef_pose[exampl_index],
          "\n img shape:",img[exampl_index].shape)   

    plt.tight_layout()

    # 保存图片
    save_path = f"{save_plt_folder}/{base_name}.png"
    plt.savefig(save_path)

    # 保存最后一张img
    cv2.imwrite(f"{save_lastPic_folder}/{base_name}.png",img[-1])
    # # 显示图片
    # plt.show()
    assert len(img)==len(aligned_state_eef_pose)==len(aligned_delta_cmd_eef_pose)==len(aligned_cmd_eef_pose)==len(aligned_state_joint)==len(aligned_cmd_joint)
    print("all length==============>:img,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint")
    print(len(img),len(aligned_state_eef_pose),len(aligned_delta_cmd_eef_pose),len(aligned_cmd_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint))   
    return img,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint




def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(episode_path,jump_index):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # read command and state
            img01,eef_s,delta_eef_a,eef_a,joint_s,joint_a = use_rosbag_to_show(episode_path)
            data = list(zip(img01,eef_s,delta_eef_a))
            grouped_data = data

            episode=[]  
            for i, (img01, state, action) in enumerate(grouped_data):
                episode.append({
                    'observation': {
                        'image01': img01,
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (len(grouped_data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(grouped_data) - 1),
                    'is_terminal': i == (len(grouped_data) - 1),
                    'language_instruction': 'Put orange into the juicer.',
                })
                
            # create output data sample
            yield_id=episode_path+"_"+str(jump_index)
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': yield_id
                }
            }
            print(yield_id,"*************")
           
            # if you want to skip an example for whatever reason, simply return None
            return yield_id, sample

    # for smallish datasets, use single-thread parsing
    # use trange for a progress bar
    from tqdm import trange
    # for sample in trange(paths):
    for sample in paths:
        # for i in range(1):
        # import random
        # random_number = random.randint(0,2)
        # print(random_number)
        yield _parse_example(sample,0)


class rosbag_WRC_juice2(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'juice2,10hz,592+10episodes',
    }
    N_WORKERS =80            # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 160  # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # MODIFY
                        'image01': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Chest camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '1x gripper position].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '1x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define data splits."""
        # MODIFY
        
        check_folder(save_plt_folder)
        check_folder(save_lastPic_folder)
        train_folder=glob.glob(f"{bag_folder_path}/*.bag")
        # train_folder=train_folder
        # val_folder=glob.glob("/home/octo/hx/dataset/raw/rosbag_WRC_juice2/val/pick_up_something*.bag")
        # print(train_folder)
        return {
            'train': train_folder,
            # 'val': val_folder,
        }
