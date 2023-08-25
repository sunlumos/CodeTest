import pandas as pd

# 读取 Excel 文件
input_file = 'D:\S\start\code\CodeTest\\test.xlsx'
output_file = 'D:\S\start\code\CodeTest\\result.xlsx'

# 读取数据
df = pd.read_excel(input_file, engine='openpyxl', header=None)

# 处理数据
def process_data(value):
    if pd.isnull(value):
        return 0
    elif value <= 5:
        return 2
    else:
        return 1

# 对每个单元格应用处理函数
df_processed = df.applymap(process_data)

# 将处理后的数据写入新的 Excel 文件
df_processed.T.to_excel(output_file, index=False, engine='openpyxl')



# 读取之前输出的 Excel 文件
processed_file = 'D:\S\start\code\CodeTest\\result.xlsx'
final_output_file = 'D:\S\start\code\CodeTest\\final_output.xlsx'

# 读取数据
df = pd.read_excel(processed_file, engine='openpyxl')

# 将所有列数据展开成一列
data_list = []
for col in df.columns:
    data_list.extend(df[col].dropna())

# 创建新的 DataFrame
df_flattened = pd.DataFrame({'Flattened Data': data_list})

# 将展开后的数据写入新的 Excel 文件
df_flattened.to_excel(final_output_file, index=False, engine='openpyxl')
