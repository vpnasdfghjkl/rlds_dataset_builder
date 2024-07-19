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


def create_kuavo_episode(episodes_dir_list,train=True):
    error_file_path = 'error_dirs.txt'
    for episode_dir in episodes_dir_list:
        try:
            episode = []
            pattern = r'(\d+\.\d+_Data)'
            match = re.search(pattern, episode_dir)
            if match:
                episode_name = match.group(1)
                print(episode_name)
            else:
                print("No match found.")
                
            cam01_rgb_dir = f'{episode_dir}/camera1'
            cam02_rgb_dir = f'{episode_dir}/camera2'
            state_txt=os.listdir(f"{episode_dir}/state")[0]
            command_txt=os.listdir(f"{episode_dir}/command")[0]
            
            states=[]
            commands=[]
            imgs01=[]
            imgs02=[]

            rec_time = os.path.basename(episode_dir).split('_')[0]
            command_txt = f'{rec_time}_command_1.txt'
            state_txt = f'{rec_time}_state_1.txt'

            with open(f"{episode_dir}/state/{state_txt}","r") as f:
                lines=f.readlines()
                for line in lines:
                    pattern = r"\[(.*?)\]"
                    matches = re.findall(pattern, line)
                    joint = [float(match) for match in matches[0].split(",")]
                    states.append(joint)

            with open(f"{episode_dir}/command/{command_txt}","r") as f:
                lines=f.readlines()
                for line in lines:
                    pattern = r"\[(.*?)\]"
                    matches = re.findall(pattern, line)
                    cs=matches[0].split(",")
                    command = [float(match) for match in matches[0].split(",")]
                    commands.append(command)


            cam01_rgb_files =sorted(os.listdir(cam01_rgb_dir))
            cam02_rgb_files = sorted(os.listdir(cam02_rgb_dir))

            for cam01_rgb_file, cam02_rgb_file in zip(cam01_rgb_files, cam02_rgb_files):
                try:
                    cam01_rgb_file_path = f"{cam01_rgb_dir}/{cam01_rgb_file}"
                    cam02_rgb_file_path = f"{cam02_rgb_dir}/{cam02_rgb_file}"

                    image01 = Image.open(cam01_rgb_file_path)
                    image02 = Image.open(cam02_rgb_file_path)
                    image_npy01 = np.array(image01)
                    image_npy02 = np.array(image02)
                    
                    # cv2.imshow("image",image_npy01)
                    # cv2.waitKey(1)
                    
                    imgs01.append(image_npy01)
                    imgs02.append(image_npy02)
                except Exception as e:
                    print(f"Error processing {cam01_rgb_file_path} or {cam02_rgb_file_path}: {e}")
                    with open(error_file_path, 'a') as ef:
                        ef.write(f"{episode_dir}\n")
                    continue_f=1
                    break
            if continue_f:
                continue
                

            imgs01_steps = np.array(imgs01)
            imgs02_steps = np.array(imgs02)
            states_steps=np.array(states,dtype="float32")
            action_steps = np.array(commands,dtype="float32")

            min_length = min(len(imgs01_steps), len(imgs02_steps), len(states_steps), len(action_steps))
            imgs01_steps = imgs01_steps[:min_length]
            imgs02_steps = imgs02_steps[:min_length]
            states_steps = states_steps[:min_length]
            action_steps = action_steps[:min_length]

            for img01,img02,state,action in zip(imgs01_steps,imgs02_steps,states_steps,action_steps): 
                episode.append({
                    'image01': img01,
                    'image02': img02,
                    'state': state,
                    'action': action,
                    'language_instruction': 'Grab the bottle and put it in the blue box',
                })
            if train:
                np.save(f'{target_dir}/train/episode_{episode_name}.npy', episode)
            else:
                np.save(f'{target_dir}/val/episode_{episode_name}.npy', episode)
        except Exception as e:
            print(f"Error processing episode {episode_dir}: {e}")
            with open(error_file_path, 'a') as ef:
                ef.write(f"{episode_dir}\n")

dataset_dir="/media/iuucb/新加卷/dataset/data_20s"
# dataset_dir_new="/media/iuucb/my passport/Human_Dataset/dataset_0710_cup/Dataset_0710_edEEF"

target_dir=f"/media/iuucb/PortableSSD/data_20s_npy"
# create fake episodes for train and validation
print("Generating train examples...")

os.makedirs(f'{target_dir}/train', exist_ok=True)
os.makedirs(f'{target_dir}/val', exist_ok=True)

dataset_dir_files=os.listdir(dataset_dir)
# dataset_dir_new_files=os.listdir(dataset_dir)
target_dir_files=os.listdir(f"{target_dir}/train")

for target_dir_file in target_dir_files:
    pattern = r"episode_(\d+\.\d+_Data).npy"
    match=re.search(pattern,target_dir_file)
    if match:
        episode_name = match.group(1)
    else:
        print("No match found.")
    if episode_name in dataset_dir_files:
        dataset_dir_files.remove(episode_name)
        # dataset_dir_new_files.remove(episode_name)
dataset_dir_files.remove("1721052845.2522707_Data")
dataset_dir_files.remove("1721052902.06321_Data")
dataset_dir_files.remove("1721053233.4030063_Data")
dataset_dir_files.remove("1721053296.4659796_Data")
dataset_dir_files.remove("1721114963.284922_Data")
dataset_dir_files.remove("1721114999.123133_Data")
dataset_dir_files.remove("1721115115.9404957_Data")
dataset_dir_files.remove("1721115147.1282399_Data")
dataset_dir_files.remove("1721115309.4953942_Data")
dataset_dir_files.remove("1721115990.8754497_Data")
dataset_dir_files.remove("1721116277.0035179_Data")
# 1721116436.3171334_Data
# 1721117056.8426318_Data
# 1721116373.5965765_Data

episodes_dir_list = sorted([os.path.join(dataset_dir, folder) for folder in dataset_dir_files if folder.startswith("1")])


create_kuavo_episode(sorted(episodes_dir_list),train=True)


print('Successfully created example data!')
