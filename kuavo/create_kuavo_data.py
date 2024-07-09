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
from PIL import Image

N_TRAIN_EPISODES = 1
N_VAL_EPISODES = 1

EPISODE_LENGTH = 10

# dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir="/home/rebot801/wangwei/dest"
files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]

print(os.listdir(dataset_dir))
print(files)


def create_fake_episode(episodes_dir_list,train=True):
    for episode_dir in episodes_dir_list:
        episode = []
        pattern = r'(\d+\.\d+_Data)'
        match = re.search(pattern, episode_dir)
        if match:
            episode_name = match.group(1)
            print(episode_name)
        else:
            print("No match found.")
            
        cam01_rgb_dir = f'{episode_dir}/camera'
        # cam02_rgb_dir = f'{episode_dir}/cam02/rgb'
        joint_dict=dict()
        print("episode_dir:",episode_dir)
        state_txt=os.listdir(f"{episode_dir}/state")[0]
        command_txt=os.listdir(f"{episode_dir}/command")[0]
        
        joints=[]
        commands=[]
        imgs01=[]
        imgs02=[]
        with open(f"{episode_dir}/state/{state_txt}","r") as f:
            lines=f.readlines()
            for line in lines:
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, line)
                js=matches[0].split(",")
                joint = [float(match) for match in matches[0].split(",")]
                joints.append(joint)
        with open(f"{episode_dir}/command/{command_txt}","r") as f:
            lines=f.readlines()
            for line in lines:
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, line)
                cs=matches[0].split(",")
                command = [float(match) for match in matches[0].split(",")]
                commands.append(command)

                
        cam01_rgb_files =sorted(os.listdir(cam01_rgb_dir))
        # cam02_rgb_files = sorted([file for file in os.listdir(cam02_rgb_dir) if file.endswith('.npy')])

        for cam01_rgb_file in cam01_rgb_files:
            cam01_rgb_file = f"{cam01_rgb_dir}/{cam01_rgb_file}"
            image = Image.open(cam01_rgb_file)
            image_npy01 = np.array(image)
            imgs01.append(image_npy01)
            
        imgs01_steps = np.array(imgs01)
        # imgs02_steps = np.array(imgs02[0:-1])
        # joints_steps = np.array(joints[0:-1],dtype="float32")
        states_steps=np.array(joints,dtype="float32")
        # action_steps = np.array(joints[1:],dtype="object")
        action_steps = np.array(commands,dtype="float32")
        
        # img=np.asarray(np.random.rand(480, 640, 3) * 255, dtype=np.uint8)
        for img01,state,action in zip(imgs01_steps,states_steps,action_steps): 
            episode.append({
                'image': img01,
                # 'wrist_image': img02,
                'state': state,
                'action': action,
                'language_instruction': 'Grab the bottle and put it in the blue box',
            })
        if train:
            np.save(f'{dataset_dir}/data/train/episode_{episode_name}.npy', episode)
        else:
            np.save(f'{dataset_dir}/data/val/episode_{episode_name}.npy', episode)

episodes_dir_list = sorted([os.path.join(dataset_dir, folder) for folder in os.listdir(dataset_dir) if folder.startswith("1")])
episodes_dir_list = episodes_dir_list[0:2]
random.shuffle(episodes_dir_list)
t_v_rate=0.5

train_size=int(t_v_rate*len(episodes_dir_list))
train_episodes_list=episodes_dir_list[:train_size]
val_episodes_list=episodes_dir_list[train_size:]

# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs(f'{dataset_dir}/data/train', exist_ok=True)
os.makedirs(f'{dataset_dir}/data/val', exist_ok=True)

create_fake_episode(train_episodes_list,train=True)
create_fake_episode(val_episodes_list,train=False)
# os.makedirs('data/train')
# for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
# create_fake_episode(f'{dataset_dir}/data/train/episode_{episodes_dir_list[i]}.npy')

# print("Generating val examples...")
# os.makedirs(f'{dataset_dir}/data/val', exist_ok=True)
# for i in tqdm.tqdm(range(N_VAL_EPISODES)):
#     create_fake_episode(f'{dataset_dir}/data/val/episode_{i}.npy')

print('Successfully created example data!')

# val_episodes=os.listdir(f"{dataset_dir}/data/val")
# one_episode_name=random.choice(val_episodes)
# one_val_episode_path=os.path.join(f'{dataset_dir}/data/val',one_episode_name)

# data = np.load(one_val_episode_path, allow_pickle=True)
# print(data[0].keys())
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# frame_rate = 30
# frame_size = (data[0]["image"].shape[1], data[0]["image"].shape[0]) 
# video_writer = cv2.VideoWriter(f'{dataset_dir}/data/train/{one_episode_name}.mp4', fourcc, frame_rate, frame_size)
# s,a=[],[]
# for step in data:
#     video_writer.write(step['image'])
#     s.append(step['state'][6])
#     a.append(step['action'][6])
# plt.plot(s,label="s")
# plt.plot(s,label="a")
# plt.legend()
# plt.show()
# video_writer.release()
# # print(data[0]["wrist_image"])
# print(data[0]["image"])
# print(data[0]["state"])