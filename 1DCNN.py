from random import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
import os
import random
import time

# ? 修改位置

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
seed_torch()

# 注意力机制
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[1, 2], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[1, 2], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

df = pd.read_csv("Cdataset/train.csv", header=None)
train_targets = df.values[:,224]  #? 这里修改为224
train_data = df.values[0:400,19:201]  # 扫描的数据中的行 列
#print(train_targets)
df2 = pd.read_csv("Cdataset/val.csv", header=None)
test_targets = df2.values[:,224]  #? 这里修改为224
test_data = df2.values[0:100,19:201]
#print(df2.head(5))

df3 = pd.read_csv("Cdataset/test.csv", header=None)
pre_targets = df3.values[:,224]  #? 这里修改为224
pre_data = df3.values[0:100,19:201]  


train_data_size = len(train_data)
test_data_size = len(test_data)
pre_data_size = len(pre_data)

print("训练数据集长度为：{}".format(train_data_size))#"训练数据集长度为：{}".format(train_data_size)：格式化字符串。会把{}换成train_data_size的内容
print("验证数据集长度为：{}".format(test_data_size))
print("测试预测集长度为：{}".format(pre_data_size))

train_data = train_data.reshape([-1,1,90])
test_data = test_data.reshape([-1,1,90])
pre_data = pre_data.reshape([-1,1,90])



train_data =torch.tensor(train_data,dtype=torch.float32)
train_targets = torch.tensor(train_targets)

test_data = torch.tensor(test_data,dtype=torch.float32)
test_targets = torch.tensor(test_targets)

pre_data = torch.tensor(pre_data,dtype=torch.float32)
pre_targets = torch.tensor(pre_targets)

train_set = TensorDataset(train_data,train_targets)
test_set = TensorDataset(test_data,test_targets)
pre_set = TensorDataset(pre_data,pre_targets)

# ! 过拟合时修改size和学习率
BATCH_SIZE = 20
learning_rate = 0.0001
DataLoader_train_data = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True,)
DataLoader_test_data = DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=True,)
DataLoader_pre_data = DataLoader(dataset=pre_set,batch_size=BATCH_SIZE,shuffle=True,)
class CNN(nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        self.BN = nn.BatchNorm1d(1)
        self.att = simam_module()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=2),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(8),
            nn.ReLU(),

        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(8,16,2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(16,32,2),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

        )

# ! 如果过拟合  删除一层或者两层
        self.layer4 = nn.Sequential(
            nn.Conv1d(32,64,2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(64,128,2),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(1152,512),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),

          #  nn.Linear(4096,1024),
            nn.Linear(512,256),
            nn.Dropout(0.4),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.BatchNorm1d(128),

            nn.Linear(128, 32),
            nn.Dropout(0.4),
            nn.BatchNorm1d(32),

            nn.Linear(32, 2),

        )

    def forward(self, x):
       # input = torch.randn(40,4,180)
        out = self.BN(x)
        out = self.att(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
      #  out = self.fc4(out)
       # out = self.fc3(out)
        return out

zh = CNN()

if torch.cuda.is_available():
    zh = zh.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 定义优化器
# ! 过拟合时修改weight_decay
optimizer = torch.optim.Adam(zh.parameters(),lr=learning_rate,weight_decay= 1e-5)

# 设置训练网络的一些参数
# 设置训练的次数
total_train_step = 0
# 设置测试的次数
total_test_step = 0
# 设置训练的轮数
# ! 过拟合时修改训练轮数
epoch = 2000

start_time = time.time()

for i in range(epoch):
    total_train_loss = 0
    total_train_acc = 0
    print("-------第{}轮训练开始-------".format(i+1))
    #训练步骤开始
    zh.train()
    for data in DataLoader_train_data:
        imgs,targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = zh(imgs)
        loss = loss_fn(outputs,targets.long())
        total_train_loss = total_train_loss + loss
        #优化器优化模型
        accuracy = (outputs.argmax(1) == targets).sum()  # 详情见tips_1.py
        total_train_acc = total_train_acc + accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

       # if total_train_step % 100 ==0:
           # end_time = time.time()
            #print("Runtime:{}".format(end_time-start_time))
       # print("训练次数：{}，Loss：{}".format(total_train_step,loss))
    print("训练集的Loss:{}".format(total_train_loss))
    print("训练集的正确率：{}".format(total_train_acc/train_data_size))

    zh.eval()
    #测试步骤开始
    total_test_loss = 0
    total_acc = 0
    # best_acc = 0
    # best_acc_epo = 0
    with torch.no_grad():
        for data in DataLoader_test_data:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zh(imgs)
            loss = loss_fn(outputs,targets.long())
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()#详情见tips_1.py
            total_acc = total_acc + accuracy
            test_acc = total_acc/test_data_size
    print("验证集的Loss:{}".format(total_test_loss))
    print("验证集的正确率:{}".format(test_acc))
    total_test_step = total_test_step + 1

    total_pre_loss = 0
    total_acc1 = 0
    # best_acc = 0
    # best_acc_epo = 0
    with torch.no_grad():
        for data in DataLoader_pre_data:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = zh(imgs)
            loss = loss_fn(outputs, targets.long())
            total_pre_loss = total_pre_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()  # 详情见tips_1.py
            total_acc1 = total_acc1 + accuracy
            pre_acc = total_acc1 / pre_data_size
    print("测试集的Loss:{}".format(total_pre_loss))
    print("测试集的正确率:{}".format(pre_acc))
    total_test_step = total_test_step + 1

    torch.save(zh.state_dict(),"baizhuo333_CNN_method_{}.pth".format(i+1))
    end_time = time.time()
    print("Runtime:{}".format(end_time-start_time))
    # print("模型已保存")
    # if test_acc >= best_acc:
    #     best_acc = test_acc
    #     best_acc_epo = i
# print("最优正确率为：{}".format(best_acc))
# print("最优正确率所在的轮数为：{}".format(best_acc_epo+1))

