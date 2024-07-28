from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow_datasets.core.utils import gcs_utils
import re,os
from PIL import Image
gcs_utils._is_gcs_disabled = True
import os
# os.environ['NO_GCE_CHECK'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES']=""

'''
tfds build --data_dir /media/rebot801/P1_LT2/tfds_pure_bg --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=15"
'''
class jump3_161m3_resize(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Kuavo dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        self.cnt=0
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image01': tfds.features.Image(
                            # shape=(360, 640, 3),
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Chest camera RGB observation.',
                        ),
                        'image02': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Third camera RGB observation.',
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

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        train_folder="/home/octo/hx/dataset/raw/pure_bg2/*_Data"
        # val_folder=""
        return {
            'train': self._generate_examples(path=train_folder),
            # 'val': self._generate_examples(path=val_folder),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        def _parse_example(episode_path,jump_index):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # read command and state
            
            pattern = r'(\d+\.\d+_Data)'
            match = re.search(pattern, episode_path)
            if match:
                episode_name = match.group(1)
                print(episode_name)
            else:
                print("No match found.")
            
            states=[]
            commands=[]
            imgs01=[]
            imgs02=[]
            data_time = os.path.basename(episode_path).split('_')[0]
            command_txt = f'{data_time}_command_1.txt'
            state_txt = f'{data_time}_state_1.txt'
            with open(f"{episode_path}/state/{state_txt}","r") as f:
                lines=f.readlines()
                for line in lines:
                    pattern = r"\[(.*?)\]"
                    matches = re.findall(pattern, line)
                    joint = [float(match) for match in matches[0].split(",")]
                    states.append(joint)

            with open(f"{episode_path}/command/{command_txt}","r") as f:
                lines=f.readlines()
                for line in lines:
                    pattern = r"\[(.*?)\]"
                    matches = re.findall(pattern, line)
                    command = [float(match) for match in matches[0].split(",")]
                    commands.append(command)

            states_steps=np.array(states,dtype="float32")
            action_steps = np.array(commands,dtype="float32")
        
            
            index=np.where(action_steps[:,-1]==1)[0]
            index=index-10
            for i,step in enumerate(action_steps):
                if i in index:
                    step[-1]=1
                else:
                    step[-1]=0

            cam01_rgb_dir = f'{episode_path}/camera1'
            cam02_rgb_dir = f'{episode_path}/camera2'
            cam01_rgb_files =sorted(os.listdir(cam01_rgb_dir))
            cam02_rgb_files = sorted(os.listdir(cam02_rgb_dir))

            for cam01_rgb_file,cam02_rgb_file in zip(cam01_rgb_files,cam02_rgb_files):
                cam01_rgb_file = f"{cam01_rgb_dir}/{cam01_rgb_file}"
                cam02_rgb_file = f"{cam02_rgb_dir}/{cam02_rgb_file}"
                image01 = Image.open(cam01_rgb_file)
                image02 = Image.open(cam02_rgb_file)
                width, height = 256, 256
                image01 = image01.resize((width, height))  
                image02 = image02.resize((128, 128))  

                image_npy01 = np.array(image01)
                image_npy02 = np.array(image02)
                imgs01.append(image_npy01)
                imgs02.append(image_npy02)
               
                       
            imgs01_steps = np.array(imgs01)
            imgs02_steps = np.array(imgs02)
            min_len=0
            min_length=min(len(imgs01_steps),len(imgs02_steps),len(states_steps),len(action_steps))
            if min_length>960:
                min_len=960
            else:
                min_len=min_length

            imgs01_steps = imgs01_steps[:min_len]
            imgs02_steps = imgs02_steps[:min_len]
            states_steps = states_steps[:min_len]
            action_steps = action_steps[:min_len]

            data = list(zip(imgs01_steps, imgs02_steps, states_steps, action_steps))
            grouped_data = data[jump_index::3]

            episode=[]  
            for i, (img01, img02, state, action) in enumerate(grouped_data):
                episode.append({
                    'observation': {
                        'image01': img01,
                        'image02': img02,
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(i == (len(grouped_data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(grouped_data) - 1),
                    'is_terminal': i == (len(grouped_data) - 1),
                    'language_instruction': 'Pick up the bottle and place it next to it.',
                    # 'language_embedding': language_embedding,
                })
                
            # create output data sample
            yield_id=episode_path+"_"+str(jump_index)
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': yield_id
                }
            }
            print(yield_id,"*************",self.cnt)
            self.cnt+=1
            # if you want to skip an example for whatever reason, simply return None
            return yield_id, sample

        # for smallish datasets, use single-thread parsing
        episode_paths = glob.glob(path)
        episode_paths = episode_paths[:100]
        for sample in episode_paths:
            for i in range(3):
                yield _parse_example(sample,i)
    
        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
        # return (
        #     beam.Create([(episode_path, i) for episode_path in episode_paths for i in range(3)])
        #     | beam.MapTuple(lambda episode_path, i: _parse_example(episode_path, i))
        # )
