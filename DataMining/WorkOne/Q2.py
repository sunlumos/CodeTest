
# 读取并打印文件
import numpy as np
from scipy.stats import zscore

# 读取文件
path = "E:\S\code\CodeTest\DataMining\WorkOne\dataset\data.online.scores"
f = open(path, 'r')
content=f.readlines()
# print(content)

# 提取第一列和第二列数据
data1 = []
data2 = []
for line in content:
    line=line.split("\t")
    data1.append(int(line[1]))
    data2.append(int(line[2]))
    

# 计算样本均值和方差
def MandVcount(data):
    # 使用mean()函数计算样本均值，将结果存储在mean变量中。
    mean = np.mean(data)
    # 使用var()函数计算样本方差，需要设置ddof=1以使用样本方差，将结果存储在variance变量中。
    variance = np.var(data, ddof=1) 
    return mean, variance
    
    
# 标准化data数据 
def standardization(data):
    z_data = zscore(data)
    return z_data

# 计算原始分数为90
def count90(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_score = (90 - mean) / std
    return normalized_score

# 计算两个样本标准化前的方差和均值
data1mean, data1variance = MandVcount(data1)
print("标准化前样本1的均值为：" + str(data1mean))
print("标准化前样本1的方差为：" + str(data1variance))

# 标准化数据
stData1 = standardization(data1)
stData2 = standardization(data2)

# 计算两个样本标准化后的方差和均值
print("标准化后样本1的均值为：" + str(MandVcount(stData1)[0]))
print("标准化后样本1的方差为：" + str(MandVcount(stData1)[1]))

# 给定原始分数 90，归一化后的相应分数
print("给定原始分数 90，data1归一化后的相应分数是:" + str(count90(data1)))
print("给定原始分数 90，data2归一化后的相应分数是:" + str(count90(data2)))

