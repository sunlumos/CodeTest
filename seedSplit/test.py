import os
import random
import shutil
import pandas as pd

# 设置文件夹路径
folder_path = './all'

# 获取所有Excel文件名
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 分割比例
train_ratio = 2/3
val_ratio = 1/6
test_ratio = 1/6

# 遍历每个Excel文件
for file in files:
    # 读取Excel文件并合并所有工作表
    df = pd.read_excel(os.path.join(folder_path, file), sheet_name=None)
    df = pd.concat(df, ignore_index=True)
    
    # 打乱数据顺序并计算训练集、验证集和测试集的大小
    indices = list(range(len(df)))
    random.shuffle(indices)
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    test_size = int(len(df) * test_ratio)
    
    # 根据分割比例划分数据集
    train_df = df.iloc[indices[:train_size]]
    val_df = df.iloc[indices[train_size:(train_size+val_size)]]
    test_df = df.iloc[indices[(train_size+val_size):(train_size+val_size+test_size)]]
    
    # 创建保存目录，并将数据集保存为新的Excel文件
    save_path = os.path.join(folder_path, 'split_datasets', file.split('.')[0])
    os.makedirs(save_path, exist_ok=True)
    train_df.to_excel(os.path.join(save_path, 'train.xlsx'), index=False)
    val_df.to_excel(os.path.join(save_path, 'val.xlsx'), index=False)
    test_df.to_excel(os.path.join(save_path, 'test.xlsx'), index=False)

print('数据集分割完成！')