
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取数据
iris_df = pd.read_csv('E:\S\code\CodeTest\DataMining\WorkOne\dataset\iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# 将特征矩阵和目标向量分离
X = iris_df.iloc[:, :-1].values
y = iris_df.iloc[:, -1].values

# 对特征矩阵进行PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 绘制散点图
colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
markers = {'Iris-setosa': 'o', 'Iris-versicolor': 's', 'Iris-virginica': '^'}

for target, color in colors.items():
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], c=color, label=target, marker=markers[target])

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

