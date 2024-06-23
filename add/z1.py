import tensorflow_datasets as tfds

# 加载本地数据集，指定数据集的路径
data_dir = '/home/lab/tensorflow_datasets/'
dataset_name = 'bridge_dataset'  # 例如 'mnist'
data_splits, info = tfds.load(name=dataset_name, data_dir=data_dir, split='train',with_info=True)

# 打印数据集信息
print(info)
for eps in data_splits.take(1):
    for ts in eps['steps']:
        print(ts['action'].numpy())
        break
    break
print('*'*88)