import ast
import os
from tqdm import tqdm, trange

class LineReader:
    def __init__(self, file_path):
        self.file = open(file_path, 'r')
        self.current_line = None

    def read_next_line(self):
        self.current_line = self.file.readline()
        if self.current_line == '':
            self.current_line = None
        return self.current_line

    def close(self):
        if self.file:
            self.file.close()


def read_dataset(dataset_dir):      # 获取指定目录下的所有文件夹名称
    folders = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    return folders

def scan_files_in_folder_cam(folder_path):       # 获取指定目录下的所有文件名称，并按文件名中的时间戳从小到大排序
    files = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
    files.sort(key=lambda x: float(x.split('.p')[0]))
    return files

def scan_files_in_folder(folder_path):       # 获取指定目录下的所有文件名称
    files = [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))]
    # files.sort(key=lambda x: os.path.getsize(os.path.join(folder_path, x)))
    return files


if __name__ == '__main__':
    results = []
    source_dir = '/home/octo/hx/dataset/raw/pure_bg2'   
    dir_list = read_dataset(source_dir)
    for dir in tqdm(dir_list, desc="Processing directories"):
        j = 0
        time_stamp = dir.split('_')[0]
        print(f"time_stamp: {time_stamp}")

        command_dir_path = os.path.join(source_dir, dir, 'command')
        a = scan_files_in_folder(command_dir_path)[0].split('_')[0]
        command_path = os.path.join(command_dir_path, f"{a}_command.txt")
        command_path_out = os.path.join(command_dir_path, f"{a}_command_1.txt")
        with open(command_path_out, 'w') as f:
            f.write('')
            f.close()
        command_file = LineReader(command_path)

        state_dir_path = os.path.join(source_dir, dir, 'state')
        b = scan_files_in_folder(state_dir_path)[0].split('_')[0]
        state_join = os.path.join(state_dir_path, f"{b}_state.txt")
        state_path_out = os.path.join(state_dir_path, f"{b}_state_1.txt")
        with open(state_path_out, 'w') as f:
            f.write('')
            f.close()
        state_file = LineReader(state_join)

        gripper_dir_path = os.path.join(source_dir, dir, 'gripper')
        gripper_join = os.path.join(gripper_dir_path, scan_files_in_folder(gripper_dir_path)[0])
        gripper_file = LineReader(gripper_join)

        camera1_dir_path = os.path.join(source_dir, dir, 'camera1')
        my_camera1_name = scan_files_in_folder_cam(camera1_dir_path)

        camera2_dir_path = os.path.join(source_dir, dir, 'camera2')
        my_camera2_name = scan_files_in_folder_cam(camera2_dir_path)

        line_count = 0
        with open(state_join, 'r') as file:
            lines = file.readlines()
            line_count = len(lines)
        with open(command_path, 'r') as file:
            lines = file.readlines()
            line_count = len(lines) if len(lines) < line_count else line_count

        min_len=min(len(my_camera1_name),len(my_camera2_name),line_count)

        for _ in trange(min_len-5, desc="Processing directories"):
            j = j + 1
            command_str = command_file.read_next_line()
            state_str = state_file.read_next_line()
            gripper_str = gripper_file.read_next_line()

            if not command_str or not state_str or not gripper_str:
                break
            if not command_str[0] == "[" or not state_str[0] == "[":
                continue
            
            command_str = command_str.split('] ', 1)[1]
            state_str = state_str.split('] ', 1)[1]
            gripper_str = gripper_str.split('] ', 1)[1]
            
            command = ast.literal_eval(command_str)
            state = ast.literal_eval(state_str)
            gripper = ast.literal_eval(gripper_str)
            
            flag = 1 if gripper[8] == 80 else 0
            
            state = state[7:]
            state.append(flag)
            command = command[7:]
            command.append(flag)

            with open(command_path_out, 'a') as f:
                f.write(f'{command}\n')
                f.close()
            with open(state_path_out, 'a') as f:
                f.write(f'{state}\n')
                f.close()
                
        # print(f"j = {j}, camera_name1 = {len(my_camera1_name)}")
        # extra_files_1 = my_camera1_name[j-1:]
        # for file in extra_files_1:
        #     cam_path1 = os.path.join(camera1_dir_path, file)
        #     if os.path.exists(cam_path1):
        #         # os.remove(cam_path)
        #         print(f"File {cam_path1} removed")
        #     else:
        #         print(f"File {cam_path1} does not exist")

        command_file.close()
        state_file.close()