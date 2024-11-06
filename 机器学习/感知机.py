import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class Perceptron:
    def __init__(self, lr=0.01, n_iters=10000):
        self.lr = lr  # 学习率
        self.n_iters = n_iters  # 迭代次数
        self.weights = None  # 权重
        self.bias = None  # 偏置
        self.errors_=[]  # 错误计算列表，记录每次迭代的误分类数

    def sign(self, x):
        return np.where(x >= 0, 1, -1)  # sign函数变换

    def fit(self, X, y):
        n, m = X.shape  # 样本特征大小
        # 初始化权重和偏置
        self.weights = np.zeros(m)
        self.bias = 0
        # 梯度下降更新权重和偏置
        for _ in range(self.n_iters):  # 迭代次数
            errors=0
            for idx, x_i in enumerate(X):  # 判断次数,enumerate输出位置加数
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.sign(linear_output)
                if y[idx] * y_predicted <0:  # 如果误分类
                    self.weights = self.weights + self.lr * y[idx] * X[idx]  # 权重更新
                    self.bias = self.bias + self.lr * y[idx]  # 偏置更新
                    errors+=1 # 增加误分类计数
            self.errors_.append(errors)
            # print(f"Epoch{_+1}:Errors={errors}")
        return self.weights, self.bias
        # print(self.weights)
        # print(self.bias)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.sign(linear_output)
        return y_predicted


# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# # metadata
# print(iris.metadata)
#
# # variable information
# print(iris.variables)

df=pd.concat([X,y],axis=1)  # 将数据集存入df文件中 ，X和y按列合并
print(df.head())

y_1=df.iloc[0:100,4].values  # 提取出第五列的特征值
y_1=np.where(y_1=='Iris-setosa',-1,1)  # 将类别标签转变为数值标签，将y=Iris-setosa转变为0，其余转变成1

x_1=df.iloc[0:100,[0,2]].values

# plt.scatter(x_1[:50,0],x_1[:50,1],color='red',marker='o',label='Setosa')
# plt.scatter(x_1[50:100,0],x_1[50:100,1],color='blue',marker='s',label='Versocolor')
# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Petal length [cm]')

# plt.show()

# 在提取的数据子集上训练感知机算法，此外，绘制每次迭代的分类错误，以检查算法是否收敛，并找到决策边界

ppn=Perceptron(lr=0.1,n_iters=10)
ppn.fit(x_1,y_1)
# plt.plot(range(1,len(ppn.errors_)+1),
#          ppn.errors_,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()

# 感知机在第6个epoch后开始收敛，在此之后，感知机能对训练样本进行分类
def plot_decision_regions(x_1,y_1,classifier,resolution=0.02):
    # 设置标记生成器和色彩图
    markers=('o','s','^','v','<')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y_1))])

    # 绘制决策面
    x1_min,x1_max=x_1[:,0].min()-1, x_1[:,0].max()+1
    x2_min,x2_max=x_1[:,1].min()-1, x_1[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    lab=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    lab=lab.reshape(xx1.shape)
    plt.contourf(xx1,xx2,lab,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制分类例子
    for idx,cl in enumerate(np.unique(y_1)):
        plt.scatter(x=x_1[y_1==cl,0],
                    y=x_1[y_1==cl,1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f"Class{cl}",
                    edgecolors='black')

plot_decision_regions(x_1,y_1,classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

