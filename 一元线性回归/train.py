# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from liner_regression import *


def show_data(x, y, w=None, b=None):
    plt.scatter(x, y, marker='.')
    if w is not None and b is not None:
        plt.plot(x, w*x+b, c='red')

    plt.show()


# # 数据生成
np.random.seed(272)
data_size = 100
x = np.random.uniform(low=1.0, high=10.0, size=data_size)#在1-10之间均匀的生成100数，x轴
y = x * 20 + 10 + np.random.normal(loc=0.0, scale=10.0, size=data_size)#均匀分布+正态分布，y轴

# plt.scatter(x, y, marker='.')
# plt.show()

# 训练集和测试机划分
# 打散数据
shuffled_index = np.random.permutation(data_size)
x = x[shuffled_index]
y = y[shuffled_index]
#训练集70%，测试集30%
split_index = int(data_size * 0.7)
x_train = x[:split_index]
y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]
#
#visualize data
# plt.scatter(x_train, y_train, marker='.')
# plt.show()
# plt.scatter(x_test, y_test, marker='.')
# plt.show()
#
# 训练线性回归模型
regr = LinerRegression(learning_rate=0.01, max_iter=700, seed=314)
regr.fit(x_train, y_train)
print('cost: \t{:.3}'.format(regr.loss()))
print('w: \t{:.3}'.format(regr.w))
print('b: \t{:.3}'.format(regr.b))
show_data(x, y, regr.w, regr.b)

# plot the evolution of cost
plt.scatter(np.arange(len(regr.loss_arr)), regr.loss_arr, marker='o', c='green')
plt.show()