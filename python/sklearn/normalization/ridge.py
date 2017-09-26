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

# 使用4次多项式回归模型在比萨训练样本上进行拟合
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)

from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4, y_train)
print ridge_poly4.score(X_test_poly4, y_test)

# 输出Lasso模型的参数列表
print ridge_poly4.coef_

regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

# 评估3种回归模型在测试数据集上的性能表现
# 准备测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 使用测试数据对4次多项式回归模型的性能进行评估
X_test_poly4 = poly4.transform(X_test)
print regressor_poly4.score(X_test_poly4, y_test)

print regressor_poly4.coef_

print np.sum(regressor_poly4.coef_**2)
print np.sum(ridge_poly4.coef_**2)
