import os

target_txt_path="/media/smj/my passport/Human_Dataset/dataset_0710_cup/Dataset_0710_edEEF/CartEEF_new/data_new/del.txt"
'''
1720580716.1550224_Data/c
1720581488.3378174_Data/c
1720581545.814178_Data/co
'''
# 列出并删除同目录下txt每行中所指的形如"1720580716.1550224_Data"的文件夹
del_cnt=0

with open(target_txt_path, "r") as f:
    import re
    for line in f.readlines():
        pattern=r"(\d+\.\d+_Data)/"
        match=re.search(pattern, line)
        if match:
            folder_name=match.group(1)
            folder_path=os.path.join(os.path.dirname(target_txt_path), folder_name)
            # print(folder_path)
            if os.path.exists(folder_path):
                del_cnt+=1
                # print(f"deleting {folder_path}")
                os.system(f"sudo rm -rf {folder_path}")
            else:
                print(f"{folder_path} not exist")
        else:
            print(f"no match in {line}")
print(f"deleted {del_cnt} folders")