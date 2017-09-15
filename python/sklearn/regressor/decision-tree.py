#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 从sklearn.datasets中导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中
boston = load_boston()

from sklearn.cross_validation import train_test_split
import numpy as np

X = boston.data  # 特征向量
y = boston.target  # 预测目标

# 随机采样25%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

# 预测目标（房价）之间的差异较大，因此需要对特征以及目标值进行标准化处理
from sklearn.preprocessing import StandardScaler
# 分别初始化对特征X和目标值y的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.fit_transform(y_test)

# 从sklearn.tree中导入DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_y_pred = dtr.predict(X_test)

# 使用R-square、MSE和MAE指标对默认配置的回归树模型在相同测试集上进行性能评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 输出linear SVR的性能评估指标
print 'R-squared value of Uniform-Weighted KNeighborRegression is ', dtr.score(
    X_test, y_test)
print 'The value of R-Squared of Linear SVR is', r2_score(y_test, dtr_y_pred)
print 'The mean squared error of Uniform-Weighted KNeighborRegression is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_pred))
print 'The mean absolute error of Uniform-Weighted KNeighborRegression is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_pred))
