import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from minirocket import MiniRocket

# 加载数据集
X, y = load_iris(return_X_y=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 初始化MiniRocket模型
rocket = MiniRocket(input_length=X_train.shape[1], num_features=10_000, random_state=42)

# 计算MiniRocket特征
X_train_transformed = rocket.fit_transform(X_train)
X_test_transformed = rocket.transform(X_test)

# 训练逻辑回归分类器
clf = LogisticRegression(random_state=42)
clf.fit(X_train_transformed, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test_transformed)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")