import pandas as pd
from sklearn.model_selection import train_test_split

# 从 Excel 文件中读取数据
df = pd.read_excel('jiafengyou2_laohua96h.xlsx', header=None)

# 划分训练集、验证集和测试集
train_df, remaining_df = train_test_split(df, test_size=200, random_state=42)
val_df, test_df = train_test_split(remaining_df, test_size=100, random_state=42)

# 写出训练集到新的 Excel 文件
train_df.to_excel('train_set.xlsx', header=None, index=None)

# 写出验证集到新的 Excel 文件
val_df.to_excel('val_set.xlsx', header=None, index=None)

# 写出测试集到新的 Excel 文件
test_df.to_excel('test_set.xlsx', header=None, index=None)