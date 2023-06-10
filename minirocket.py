import numpy as np
from scipy.stats import norm
from numba import njit

@njit
def miniRocket(X, n_features, time_steps, K, D):
    # 计算每个时间步的滑动窗口大小
    window_size = (time_steps - K) // D + 1

    # 计算每个子序列的平均值和标准差
    X_mean = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)

    # 初始化特征矩阵和偏差矩阵
    features = np.zeros((n_features, window_size))
    biases = np.zeros((n_features, window_size))

    # 计算随机权重矩阵
    weights = np.random.normal(size=(n_features, K))

    # 计算特征矩阵和偏差矩阵
    for i in range(n_features):
        for j in range(window_size):
            start = j * D
            end = start+ K
            subsequence = X[i, start:end]

            # 计算子序列的滑动平均值和滑动标准差
            subsequence_mean = np.mean(subsequence)
            subsequence_std = np.std(subsequence)

            # 计算子序列的特征值和偏差值
            feature_value = np.dot(subsequence - subsequence_mean, weights[i])
            feature_value /= subsequence_std * np.sqrt(K)
            features[i, j] = feature_value
            biases[i, j] = subsequence_mean - X_mean[i] * (1 - K / time_steps) + subsequence_std / np.sqrt(K) * norm.ppf((1 + 1 / n_features) / 2)

    # 将特征矩阵和偏差矩阵拼接成一维数组
    features = features.reshape(-1)
    biases = biases.reshape(-1)

    # 计算特征矩阵和偏差矩阵的均值和标准差
    features_mean = np.mean(features)
    features_std = np.std(features)
    biases_mean = np.mean(biases)
    biases_std = np.std(biases)

    # 对特征矩阵和偏差矩阵进行归一化
    features = (features - features_mean) / features_std
    biases = (biases - biases_mean) / biases_std

    # 将特征矩阵和偏差矩阵拼接起来作为最终的特征向量
    feature_vector = np.concatenate((features, biases))

    return feature_vector