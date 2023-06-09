import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
iris_df = pd.read_csv('E:\S\code\CodeTest\DataMining\WorkOne\dataset\iris.data', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = iris_df.loc[:, features].values

sc = StandardScaler()
x_sc = sc.fit_transform(x)

def subplotFormat(fig, n, targets, colors, finalDf, str1, str2):
    ax = fig.add_subplot(1, 3, n)  
    ax.set_xlabel(str1, fontsize=15)
    ax.set_ylabel(str2, fontsize=15)
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, str1], finalDf.loc[indicesToKeep, str2], c=color, s=50)
    plt.legend(targets)  
    plt.grid()  
    

pca = PCA(n_components=2)
principalCompoments = pca.fit_transform(x_sc)  
principalDf = pd.DataFrame(data=principalCompoments,columns=['pc1', 'pc2'])  

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

finalDf = pd.concat([principalDf, iris_df[['target']]], axis=1)  
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

for target, color in zip(targets, colors):  
    indicesToKeep = finalDf['target'] == target  
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
                   c=color, s=50)  
    ax.legend(targets) 
    ax.grid() 
    plt.show()
    fig.savefig('work03scatter.jpg', transparent=True)

    pca = PCA(n_components=3)
    principalCompomentsb = pca.fit_transform(x_sc)  
    principalDfB = pd.DataFrame(data=principalCompomentsb, columns=['pc1', 'pc2', 'pc3'])

    finalDf = pd.concat([principalDfB, iris_df[['target']]], axis=1)
    fig1 = plt.figure(figsize=(21, 8))

    subplotFormat(fig1, 1, targets, colors, finalDf, 'pc1', 'pc2')
    subplotFormat(fig1, 2, targets, colors, finalDf, 'pc1', 'pc3')
    subplotFormat(fig1, 3, targets, colors, finalDf, 'pc2', 'pc3')

    plt.suptitle("PCA scatter plots", ha="center", fontsize=25)
    plt.tight_layout()  
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig1.savefig('work03b.jpg', transparent=True)

