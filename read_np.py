import numpy as np
import cv2
import matplotlib.pyplot as plt
data=np.load("/media/rebot801/PortableSSD/data_npy/train/episode_1720599244.7651644_Data.npy",allow_pickle=True)


rgb_image = cv2.cvtColor(data[0]["image01"], cv2.COLOR_BGR2RGB)
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
print(data[0].keys())
print(data[0]["image01"].shape)
cv2.imshow("image",data[0]["image01"])
cv2.imshow("3image",data[0]["image02"])
cv2.waitKey(0)

print(data[0]["state"])