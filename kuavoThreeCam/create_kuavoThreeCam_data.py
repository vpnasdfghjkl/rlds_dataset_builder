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


def create_fake_episode(episodes_dir_list,episodes_dir_list_new,train=True):
    for episode_dir,episode_dir_Cartesian in zip(episodes_dir_list,episodes_dir_list_new):
        assert episode_dir.split("/")[-1]==episode_dir_Cartesian.split("/")[-1]
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
        state_txt=os.listdir(f"{episode_dir_Cartesian}/state")[0]
        command_txt=os.listdir(f"{episode_dir_Cartesian}/command")[0]
        
        states=[]
        commands=[]
        imgs01=[]
        imgs02=[]
        with open(f"{episode_dir_Cartesian}/state/{state_txt}","r") as f:
            lines=f.readlines()
            for line in lines:
                pattern = r"\[(.*?)\]"
                matches = re.findall(pattern, line)
                joint = [float(match) for match in matches[0].split(",")]
                states.append(joint)
        with open(f"{episode_dir_Cartesian}/command/{command_txt}","r") as f:
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
            image = cv2.imread(cam01_rgb_file)
            image_npy01 = np.array(image)
            # cv2.imshow("image",image_npy01)
            # cv2.waitKey(1)
            imgs01.append(image_npy01)
            

        imgs01_steps = np.array(imgs01)
        imgs02_steps = imgs01_steps
        imgs03_steps = imgs01_steps
        states_steps=np.array(states,dtype="float32")
        action_steps = np.array(commands,dtype="float32")

        # imgs01_steps = np.array(imgs01[0:-1])
        # imgs02_steps = np.array(imgs02[0:-1])
        # states_steps=np.array(joints[0:-1],dtype="float32")
        # action_steps = np.array(joints[1:],dtype="float32")
        
        for img01,img02,img03,state,action in zip(imgs01_steps,imgs02_steps,imgs03_steps,states_steps,action_steps): 
            episode.append({
                'image': img01,
                'third_cam': img02,
                'extra_cam': img03,
                'state': state,
                'action': action,
                'language_instruction': 'Grab the bottle and put it in the blue box',
            })
        if train:
            np.save(f'{target_dir}/train/episode_{episode_name}.npy', episode)
        else:
            np.save(f'{target_dir}/val/episode_{episode_name}.npy', episode)


dataset_dir="/home/rebot801/wangwei/dest"
dataset_dir="/media/smj/PortableSSD/dest"
dataset_dir_new="/media/smj/PortableSSD/dest_new"
target_dir=f"{dataset_dir}/data_threeCam"
slice_k=5
episodes_dir_list = sorted([os.path.join(dataset_dir, folder) for folder in os.listdir(dataset_dir) if folder.startswith("1")])
episodes_dir_list_new= sorted([os.path.join(dataset_dir_new, folder) for folder in os.listdir(dataset_dir_new) if folder.startswith("1")])

episodes_dir_list,episodes_dir_list_new = episodes_dir_list[0:slice_k],episodes_dir_list_new[0:slice_k]
assert len(episodes_dir_list)==len(episodes_dir_list_new)
# random.shuffle(episodes_dir_list)
t_v_rate=0.8
train_size=int(t_v_rate*len(episodes_dir_list))

indices = random.sample(range(len(episodes_dir_list)-1), train_size)

# 使用下标列表从两个列表中选择元素
train_episodes_list = [episodes_dir_list[i] for i in indices]
train_episodes_list_new = [episodes_dir_list_new[i] for i in indices]

# 选择剩余的元素
val_episodes_list = [e for i, e in enumerate(episodes_dir_list) if i not in indices]
val_episodes_list_new = [e for i, e in enumerate(episodes_dir_list_new) if i not in indices]

# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs(f'{target_dir}/train', exist_ok=True)
os.makedirs(f'{target_dir}/val', exist_ok=True)

create_fake_episode(sorted(train_episodes_list),sorted(train_episodes_list_new),train=True)
create_fake_episode(sorted(val_episodes_list),sorted(val_episodes_list_new),train=False)


print('Successfully created example data!')
