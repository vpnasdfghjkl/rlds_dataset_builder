import numpy as np
import tqdm
import os

N_TRAIN_EPISODES = 20
N_VAL_EPISODES = 10

EPISODE_LENGTH = 10


def create_fake_episode(path):
    episode = []
    for step in range(EPISODE_LENGTH):
        img=np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8)
        episode.append({
            'image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'wrist_image': np.asarray(np.zeros_like(img), dtype=np.uint8),
            'state': np.asarray(np.random.rand(7), dtype=np.float32),
            'action': np.asarray(np.random.rand(7), dtype=np.float32),
            'language_instruction': 'dummy instruction',
        })
    np.save(path, episode)


# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs('data/train', exist_ok=True)
# os.makedirs('data/train')
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_fake_episode(f'data/train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs('data/val', exist_ok=True)
for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    create_fake_episode(f'data/val/episode_{i}.npy')

print('Successfully created example data!')

data = np.load("data/val/episode_0.npy", allow_pickle=True)
print(data[0].keys())
print(data[0]["wrist_image"])
print(data[0]["image"])
print(data[0]["state"])