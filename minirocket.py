# 导入所需的库
import numpy as np

f = open('D:\S\start\code\CodeTest\minirocket.txt','w')

# 定义MiniRocket类
class MiniRocket:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = None
        self.biases = None
        
    def fit(self, X):
        # 生成随机权重和偏置
        self.weights = np.random.randn(self.num_features, X.shape[1])
        self.biases = np.random.uniform(0, 2*np.pi, self.num_features)
        
    def transform(self, X):
        # 使用MiniRocket算法进行特征转换
        transformed_features = np.cos(np.dot(X, self.weights.T) + self.biases)
        return transformed_features

# 导入所需的库和数据
from sklearn.datasets import load_iris

# 加载示例数据集（这里以鸢尾花数据集为例）
data = load_iris()
X = data.data  # 特征数据

# 创建并拟合MiniRocket模型
num_features = 100  # 设置特征数量
minirocket = MiniRocket(num_features)
minirocket.fit(X)

# 转换特征数据
transformed_features = minirocket.transform(X)
print(transformed_features, file=f)
print(transformed_features.shape)  # 输出转换后的特征矩阵的形状
print(transformed_features[:5, :])  # 输出前5个样本的转换后的特征
