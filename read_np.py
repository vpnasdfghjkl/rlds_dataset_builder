import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# data = np.load("/media/smj/新加卷/octo_demo_dataset/train/episode_1720601623.977485_Data.npy", allow_pickle=True)
# data = np.load("/media/smj/PortableSSD/data_npy/train/episode_1720580579.6915085_Data.npy", allow_pickle=True)
def generate_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    frame_rate = 30
    frame_size = (data[0]["image01"].shape[1], data[0]["image01"].shape[0]) 
    video_writer = cv2.VideoWriter(f'./one_episode.mp4', fourcc, frame_rate, frame_size)
    s,a=[],[]
    for step in data:
        video_writer.write(step['image01'])
        s.append(step['state'][1])
        a.append(step['action'][1])
    plt.plot(s,label="s")
    plt.plot(a,label="a")
    plt.legend()
    plt.show()
    video_writer.release()

def show_image():
    img=np.array(data[0]["image02"])
    img=Image.fromarray(img).convert("RGB")
    plt.imshow(img)
    plt.show()

def plot_state_action():
    s,a=[],[]
    for step in data:
        s.append(step['state'] [3])
        a.append(step['action'][3])
    plt.plot(s,label="s")
    # plt.plot(a,label="a")
    plt.legend()
    plt.show()

def check_other_info():
    print(data[0].keys())
    print(data[0]["state"])
    print(data[0]["action"])
    print(data[0]["language_instruction"])

def modify_to_cartesian():
    data = np.load("/media/smj/my passport/Human_Dataset/dataset_0710_cup/Dataset_0710_ed/data_Cartesian/train/episode_1720618627.4341943_Data.npy", allow_pickle=True)
    for step in data:
        print(step['state'])
        print(step['action'])
        step['state'] *= 2
        step['action'] *= 2
        break
    np.save("/home/smj/hx/episode_1720580579.6915085_Data_modified.npy", data)

# show_image()
# plot_state_action()
# check_other_info()
# modify_to_cartesian()
data = np.load("/media/smj/PortableSSD/data_npy/71/episode_1720580579.6915085_Data.npy", allow_pickle=True)

plot_state_action()
check_other_info()