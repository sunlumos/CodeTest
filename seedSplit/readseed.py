import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def split_dataset(filename):
    # 读取xlsx文件
    data = pd.read_excel(filename, header=None)

    # 随机打乱数据
    data = shuffle(data)

    # 计算数据集的大小
    total_samples = len(data)
    train_samples = int(0.8 * total_samples)
    test_samples = int(0.1 * total_samples)

    # 分割数据集
    train_data = data[:train_samples].reset_index(drop=True,)
    test_data = data[train_samples:train_samples+test_samples].reset_index(drop=True)
    validation_data = data[train_samples+test_samples:].reset_index(drop=True)

    # 提取输入文件名（去除路径和扩展名）
    input_filename = filename.split('\\')[-1].split('.')[0]

    # 保存为CSV文件
    train_data.to_csv(f'D:\\S\\start\\code\\CodeTest\\seedSplit\\{input_filename}_train_data.csv', index=False, header=False)
    test_data.to_csv(f'D:\\S\\start\\code\\CodeTest\\seedSplit\\{input_filename}_test_data.csv', index=False, header=False)
    validation_data.to_csv(f'D:\\S\\start\\code\\CodeTest\\seedSplit\\{input_filename}_val_data.csv', index=False, header=False)

# 示例调用
split_dataset('D:\S\start\code\CodeTest\seedSplit\\all\jiafengyou2_weilaohua.xlsx')
