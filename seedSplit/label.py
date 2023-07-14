import pandas as pd
import random

# 读取xlsx文件，假设数据在第一个sheet中
filename = 'D:\S\start\code\CodeTest\seedSplit\\all\yongyou1540_laohua96h.xlsx'
data = pd.read_excel(filename, header=None)

# 获取数据总数
total_samples = len(data)

# 设置比例
train_ratio = 2/3
test_ratio = 1/6
val_ratio = 1/6

# 计算每个数据集的数量
train_samples = int(total_samples * train_ratio)
test_samples = int(total_samples * test_ratio)
val_samples = int(total_samples * val_ratio)

# 随机打乱数据
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 分割数据集
train_data = data[:train_samples]
test_data = data[train_samples:train_samples+test_samples]
val_data = data[train_samples+test_samples:train_samples+test_samples+val_samples]

# 检查每个标签的数量是否符合要求
label_counts = data[224].value_counts()
train_label_counts = train_data[224].value_counts()
test_label_counts = test_data[224].value_counts()
val_label_counts = val_data[224].value_counts()

for label in label_counts.index:
    label_ratio = label_counts[label] / total_samples
    train_ratio = train_label_counts.get(label, 0) / train_samples
    test_ratio = test_label_counts.get(label, 0) / test_samples
    val_ratio = val_label_counts.get(label, 0) / val_samples
    
    if label_ratio != train_ratio or label_ratio != test_ratio or label_ratio != val_ratio:
        print(f"Label {label} ratio is not maintained in train, test, and val sets.")

# 将结果输出为CSV文件
# 提取输入文件名（去除路径和扩展名）
input_filename = filename.split('\\')[-1].split('.')[0]
train_data.to_csv(f'D:\S\start\code\CodeTest\seedSplit\\{input_filename}_train_data.csv', index=False, header=False)
test_data.to_csv(f'D:\S\start\code\CodeTest\seedSplit\\{input_filename}_test_data.csv', index=False, header=False)
val_data.to_csv(f'D:\S\start\code\CodeTest\seedSplit\\{input_filename}_val_data.csv', index=False, header=False)
