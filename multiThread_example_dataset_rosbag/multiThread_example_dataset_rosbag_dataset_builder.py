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
from cv_bridge import CvBridge



def use_rosbag_to_show(bag_name):
    bridge = CvBridge()
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    # 读取rosbag文件并提取所需数据
    bag = rosbag.Bag(bag_name, 'r')
    kuavo_arm_traj_data = []
    robot_arm_q_v_tau_data = []
    left_hand_state=[]
    img=[]
    start_time = bag.get_start_time()
    end_time = start_time + 10

    kuavo_arm_traj_data_time_stamp=[]
    robot_arm_q_v_tau_data_time_stamp=[]
    left_hand_state_stamp=[]
    img_stamp=[]
    
    for topic, msg, t in bag.read_messages(topics=['/kuavo_arm_traj', '/robot_arm_q_v_tau','/head_camera/color/image_raw','/robot_hand_position']):
        msg_time = msg.header.stamp.to_sec()  # 将时间戳转换为秒
        if msg_time > end_time:
            break  # 超过时间限制，停止读取

        if topic == '/kuavo_arm_traj':
            # kuavo_arm_traj_data.append(msg.position)
            kuavo_arm_traj_data_time_stamp.append(msg.header.stamp)

            kuavo_arm_traj_data.append(np.radians(msg.position)[:7])
            
        elif topic == '/robot_arm_q_v_tau':
            # 将弧度转换为角度
            # robot_arm_q_v_tau_data.append(np.rad2deg(msg.q))
            robot_arm_q_v_tau_data_time_stamp.append(msg.header.stamp)
            robot_arm_q_v_tau_data.append((msg.q)[:7])
            
        elif topic=='/robot_hand_position':
            left_hand_state_stamp.append(msg.header.stamp)

            left_hand_pose=msg.left_hand_position
            if left_hand_pose[-1]==20:
                grip=0
            elif left_hand_pose[-1]==80:
                grip=1
            else:
                print("hand pose error")
            left_hand_state.append(grip)

        elif topic=='/head_camera/color/image_raw':
            img_stamp.append(msg.header.stamp)
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_img = cv2.resize(cv_img, (256, 256))
            # 将调整后的图像添加到 img 列表
            img.append(resized_img)
            # cv2.imshow("s",cv_img)
            # cv2.waitKey(1)


    bag.close()

    print("kuavo_arm_traj_data : ", len(kuavo_arm_traj_data))
    print("robot_arm_q_v_tau_data", len(robot_arm_q_v_tau_data))
    print("left_hand_state_stamp", len(left_hand_state))
    print("img_stamp", len(img))
    
    # 安全判断
    if len(kuavo_arm_traj_data) == 0 or len(robot_arm_q_v_tau_data) == 0:
        print("ROS bag file contains empty data for at least one topic.")
        return

    if len(kuavo_arm_traj_data) < 100 or len(robot_arm_q_v_tau_data) < 100:
        print("ROS bag file data count is too small (less than 100 data points). Please check again.")
        return
    

    aligned_robot_arm_q_v_tau_data = []
    aligned_kuavo_arm_traj_data = []
    aligned_left_hand_state=[]

    for stamp in img_stamp:
        stamp_sec=stamp.to_sec()
        idx_s = np.argmin(np.abs(np.array([t.to_sec() for t in robot_arm_q_v_tau_data_time_stamp]) - stamp_sec))
        aligned_robot_arm_q_v_tau_data.append(robot_arm_q_v_tau_data[idx_s])

        idx_a = np.argmin(np.abs(np.array([t.to_sec() for t in kuavo_arm_traj_data_time_stamp]) - stamp_sec))
        aligned_kuavo_arm_traj_data.append(kuavo_arm_traj_data[idx_a])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in left_hand_state_stamp]) - stamp_sec))
        aligned_left_hand_state.append(left_hand_state[idx_h])

    aligned_left_hand_state_for_action=aligned_left_hand_state.copy()
    jump_index=0
    for i in range(len(aligned_left_hand_state)):
        if aligned_left_hand_state[i]==1:
            jump_index=i
            break
    
    print("jump_index",jump_index)
    for i in range(6):
        aligned_left_hand_state_for_action[jump_index-i]=1
    # jump_index=0
    # for i in range(len(aligned_left_hand_state)):
    #     if aligned_left_hand_state[i]==1:
    #         jump_index=i
    #         break
    # print("jump_index",jump_index)
    # def steep_sigmoid(x, k=1.0, x0=0):
    #     return 1 / (1 + np.exp(-k * (x - x0)))
    # for i in range(len(aligned_left_hand_state)):
    #     aligned_left_hand_state[i]=steep_sigmoid(i,k=0.6,x0=jump_index)
    
    aligned_robot_arm_q_v_tau_data = [list(item) for item in aligned_robot_arm_q_v_tau_data]
    aligned_kuavo_arm_traj_data = [list(item) for item in aligned_kuavo_arm_traj_data]

    for i in range(len(img_stamp)):
        aligned_robot_arm_q_v_tau_data[i].append(aligned_left_hand_state[i])
        aligned_kuavo_arm_traj_data[i].append(aligned_left_hand_state_for_action[i])


    print(aligned_kuavo_arm_traj_data[250])
    # 创建3行5列的图表并进行比较
    num_plots = min(len(aligned_kuavo_arm_traj_data[0]), len(aligned_robot_arm_q_v_tau_data[0]), 15)  # 限制最多只显示15个数据对比
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(base_name, fontsize=16)
    for i in range(num_plots):
        kuavo_position = [data[i] for data in aligned_kuavo_arm_traj_data]
        robot_q = [data[i] for data in aligned_robot_arm_q_v_tau_data]
        row = i // 5
        col = i % 5
        axs[row, col].plot(kuavo_position, label='/kuavo_arm_traj')
        axs[row, col].plot(robot_q, label='/robot_arm_q_v_tau')
        axs[row, col].set_title(f"motor {i+1} state")
        axs[row, col].legend()

    plt.tight_layout()

    # 保存图片
    save_path = f"./bag_picture/{base_name}.png"
    plt.savefig(save_path)

    # 显示图片
    # plt.show()

    return img,aligned_robot_arm_q_v_tau_data,aligned_kuavo_arm_traj_data


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(episode_path,jump_index):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # read command and state
            img01,s,a = use_rosbag_to_show(episode_path)
            data = list(zip(img01,s,a))
            grouped_data = data[jump_index::1]

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
                    'language_instruction': 'Pick up the bottle and place it next to it.',
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
        import random
        random_number = random.randint(0,0)
        # print(random_number)
        yield _parse_example(sample,random_number)


class shenzhen2(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': '修改为256*256',
    }
    N_WORKERS =20             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 40  # number of paths converted & stored in memory before writing to disk
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
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '1x gripper position].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
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

        # train_folder=glob.glob("/home/octo/hx/dataset/raw/pure_bg2/*_Data")
        train_folder=glob.glob("/home/lab/hx/rosbagfiles_0807_2/pick*.bag")
        print(train_folder)
        # val_folder=""
        return {
            'train': train_folder,
            # 'val': self._generate_examples(path=val_folder),
        }
