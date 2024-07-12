import os
import shutil
s_dir="/media/smj/Ventoy/Dataset"
# s_dir="/media/smj/PortableSSD/dest"
t_dir="/media/smj/C8ED-BD85/data_only_txt"
episodes_dir_list=sorted([folder for folder in os.listdir(s_dir) if folder.startswith("1")])
# episodes_dir_list_path = sorted([os.path.join(s_dir, folder) for folder in os.listdir(s_dir) if folder.startswith("1")])

for sub_dir in episodes_dir_list:
    sub_dir_path = os.path.join(s_dir, sub_dir)
    # if os.path.isdir(sub_dir_path) and all(folder in os.listdir(sub_dir_path) for folder in ['command', 'state']):
    if os.path.isdir(sub_dir_path):
        target_dir = os.path.join(t_dir, sub_dir)
        for folder in ['command', 'state','gripper']:
            folder_path = os.path.join(sub_dir_path, folder)
            target_folder_path = os.path.join(target_dir, folder)
            shutil.copytree(folder_path, target_folder_path)
