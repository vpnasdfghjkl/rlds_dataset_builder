import numpy as np
import tqdm
import os

import os,re
from datetime import datetime, timedelta
from typing import List, Optional
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

N_TRAIN_EPISODES = 1
N_VAL_EPISODES = 1

EPISODE_LENGTH = 10

dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir="/home/lab/hanxiao/dataset"

def create_fake_episode(episodes_dir_list,train=True):
    
    for episode_dir in episodes_dir_list:
        episode = []
        p=r'dataset_(grab_cube0.*)'
        match = re.search(p, episode_dir)
        episode_name = match.group(1)
        cam01_rgb_dir = f'{episode_dir}/cam01/rgb'
        cam02_rgb_dir = f'{episode_dir}/cam02/rgb'
        joint_dict=dict()
        with open(f"{episode_dir}/joint/joint_positions.txt","r") as f:
            lines=f.readlines()
            for line in lines:
                timestamp_str, values_str = line.split(': ')
                pattern = r"[-+]?\d*\.\d+|\d+"
                matches = re.findall(pattern, values_str)
                joint = [float(match) for match in matches]
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                joint_dict[timestamp]=joint
                
        joint_timestamps = list(joint_dict.keys())
        cam01_rgb_files = sorted([file for file in os.listdir(cam01_rgb_dir) if file.endswith('.npy')])
        cam02_rgb_files = sorted([file for file in os.listdir(cam02_rgb_dir) if file.endswith('.npy')])


        def extract_timestamp_from_filename(filename):
            pattern = r'color_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)'
            match = re.search(pattern, filename)
            timestamp_str = match.group(1)
            timestamp_str = timestamp_str.replace('_', ':')
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            print("提取的时间戳:", timestamp)
            return timestamp

        def find_closest_timestamp(timestamp: datetime, timestamps: List[datetime]) -> Optional[datetime]:
            if not timestamps:
                return None
            return min(timestamps, key=lambda dt: abs(dt - timestamp))

       
        joints=[]
        imgs01=[]
        imgs02=[]
        image_timestamp02=[extract_timestamp_from_filename(f"{cam02_rgb_dir}/{cam02_rgb_file}") for cam02_rgb_file in cam02_rgb_files]
        for cam01_rgb_file in cam01_rgb_files:
            cam01_rgb_file = f"{cam01_rgb_dir}/{cam01_rgb_file}"
            image_npy01=np.load(cam01_rgb_file)
            image_timestamp01 =extract_timestamp_from_filename(cam01_rgb_file)
            
            closest_joint_to_cam01= find_closest_timestamp(image_timestamp01, joint_timestamps)
            joints.append(joint_dict[closest_joint_to_cam01])
            closest_cam02_to_joint = find_closest_timestamp(closest_joint_to_cam01, image_timestamp02)
            
            image_npy02=np.load(f"{cam02_rgb_dir}/color_{closest_cam02_to_joint.strftime('%Y-%m-%d %H:%M:%S.%f')}.npy")
            imgs01.append(image_npy01)
            imgs02.append(image_npy02)
        imgs01_steps = np.array(imgs01[0:-1])
        imgs02_steps = np.array(imgs02[0:-1])
        joints_steps = np.array(joints[0:-1],dtype="object")
        action_steps = np.array(joints[1:],dtype="object")
        # img=np.asarray(np.random.rand(480, 640, 3) * 255, dtype=np.uint8)
        for img01,img02,joint,action in zip(imgs01_steps,imgs02_steps,joints_steps,action_steps): 
            episode.append({
                'image': img01,
                'wrist_image': img02,
                'state': joint,
                'action': action,
                'language_instruction': 'Grab the sponge cube and put it on the plate',
            })
        if train:
            np.save(f'{dataset_dir}/data/train/episode_{episode_name}.npy', episode)
        else:
            np.save(f'{dataset_dir}/data/val/episode_{episode_name}.npy', episode)

episodes_dir_list = sorted([os.path.join(dataset_dir, folder) for folder in os.listdir(dataset_dir) if folder.startswith('dataset')])
random.shuffle(episodes_dir_list)
t_v_rate=0.9

train_size=int(t_v_rate*len(episodes_dir_list))
train_episodes_list=episodes_dir_list[:train_size]
val_episodes_list=episodes_dir_list[train_size:]

# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs(f'{dataset_dir}/data/train', exist_ok=True)
os.makedirs(f'{dataset_dir}/data/val', exist_ok=True)

# create_fake_episode(train_episodes_list,train=True)
# create_fake_episode(val_episodes_list,train=False)
# os.makedirs('data/train')
# for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
# create_fake_episode(f'{dataset_dir}/data/train/episode_{episodes_dir_list[i]}.npy')

# print("Generating val examples...")
# os.makedirs(f'{dataset_dir}/data/val', exist_ok=True)
# for i in tqdm.tqdm(range(N_VAL_EPISODES)):
#     create_fake_episode(f'{dataset_dir}/data/val/episode_{i}.npy')

print('Successfully created example data!')

val_episodes=os.listdir(f"{dataset_dir}/data/val")
one_episode_name=random.choice(val_episodes)
one_val_episode_path=os.path.join(f'{dataset_dir}/data/val',one_episode_name)

data = np.load("/home/lab/hanxiao/dataset/data/val/episode_grab_cube007.npy", allow_pickle=True)
print(data[0].keys())
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
frame_rate = 30
frame_size = (data[0]["image"].shape[1], data[0]["image"].shape[0]) 
video_writer = cv2.VideoWriter(f'{dataset_dir}/data/train/{one_episode_name}.mp4', fourcc, frame_rate, frame_size)
s,a=[],[]
for step in data:
    video_writer.write(step['image'])
    s.append(step['state'][6])
    a.append(step['action'][6])
plt.plot(s,label="s")
plt.plot(s,label="a")
plt.legend()
plt.show()
video_writer.release()
# print(data[0]["wrist_image"])
# print(data[0]["image"])
# print(data[0]["state"])