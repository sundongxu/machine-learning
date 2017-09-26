#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 不同次数的多项式模型拟合披萨直径与售价之间的关系
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

import numpy as np
# 在x轴上从0到25均匀采样100个数据点
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
# 以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)

# 对回归预测到的直线进行作图
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1])
plt.show()

# 输出线性回归模型在训练样本上的R-Squared值
print 'The R-squared value of Linear Regressor performing on the training data is', regressor.score(
    X_train, y_train)

# 使用2次多项式回归模型在比萨训练样本上进行拟合
from sklearn.preprocessing import PolynomialFeatures
# 使用PolynomialFeatures（degree=2）映射出2次多项式特征，存储在变量X_train_poly2中
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)

regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_train_poly2, y_train)

# 重新映射绘图用x轴采样数据
xx_poly2 = poly2.transform(xx)
# 使用2次多项式回归模型对应x轴采样数据进行回归预测
yy_poly2 = regressor_poly2.predict(xx_poly2)

# 分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图
plt.scatter(X_train, y_train)

plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2])
plt.show()

# 输出2次多项式回归模型在训练样本上的R-squared值
print 'The R-squared value of Polynomial Regressor(Degree=2) performing on the training data is', regressor_poly2.score(
    X_train_poly2, y_train)

# 使用4次多项式回归模型在比萨训练样本上进行拟合
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

plt.scatter(X_train, y_train)

plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()

# 输出2次多项式回归模型在训练样本上的R-squared值
print 'The R-squared value of Polynomial Regressor(Degree=4) performing on the training data is', regressor_poly4.score(
    X_train_poly4, y_train)

# 评估3种回归模型在测试数据集上的性能表现
# 准备测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 使用测试数据对线性回归模型的性能进行评估
regressor.score(X_test, y_test)

# 使用测试数据对2次多项式回归模型的性能进行评估
X_test_poly2 = poly2.transform(X_test)
regressor_poly2.score(X_test_poly2, y_test)

# 使用测试数据对4次多项式回归模型的性能进行评估
X_test_poly4 = poly4.transform(X_test)
regressor_poly4.score(X_test_poly4, y_test)
