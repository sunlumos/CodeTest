import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

df = pd.read_excel("D:\S\start\code\jiafengyou-train.xlsx", header=None)
train_targets = df.values[:,224]
train_data = df.values[0:800,19:201]
#print(train_targets)
df2 = pd.read_excel("D:\S\start\code\jiafengyou2-test.xlsx", header=None)
test_targets = df2.values[:,224]
test_data = df2.values[0:800,19:201]
#print(df2.head(5))
df3 = pd.read_excel("D:\S\start\code\jiafengyou-val.xlsx", header=None)
pre_targets = df3.values[:,224]
pre_data = df3.values[0:800,19:201]

train_tensor_len = len(train_data)
test_tensor_len =  len(test_data)
pre_tensor_len = len(pre_data)
print("训练数据集长度为：{}".format(train_tensor_len))#"训练数据集长度为：{}".format(train_data_size)：格式化字符串。会把{}换成train_data_size的内容
print("验证数据集长度为：{}".format(test_tensor_len))
print("测试数据集长度为：{}".format(pre_tensor_len))

# svm.SVC(C=2,kernel='linear',gamma=10,) 训练集： 0.76875 验证集： 0.635 测试集： 0.635
# svm.SVC(C=1,kernel='poly',degree=3,) 训练集： 0.94625 验证集： 0.86 测试集： 0.85
# svm.SVC(C=1000,kernel='rbf',gamma=1,) 训练集： 0.9375 验证集： 0.83 测试集： 0.845
# svm.SVC(C=1000,kernel='rbf',gamma=0.1,) 0.89625 0.79 0.76
model = svm.SVC(C=1,kernel='poly',degree=4,)
model.fit(train_data,train_targets)

train_score = model.score(train_data,train_targets)
print("训练集：",train_score)
test_score = model.score(test_data,test_targets)
print("验证集：",test_score)
pre_score = model.score(pre_data,pre_targets)
print("测试集：",pre_score)
