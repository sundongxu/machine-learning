#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 从sklearn.datasets中导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中
boston = load_boston()
# 输出数据描述
print boston.DESCR

from sklearn.cross_validation import train_test_split
import numpy as np

X = boston.data  # 特征向量
y = boston.target  # 预测目标

# 随机采样25%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

# 分析回归目标值的差异
print 'The max target value is ', np.max(y)
print 'The min target value is', np.min(y)
print 'The average target value is', np.mean(y)

# 预测目标最大值50，最小值5，平均值22.5328...，
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

from sklearn.linear_model import LinearRegression
# 训练拟合建立模型，回归预测
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_pred = sgdr.predict(X_test)

# 评估LinearRegression
print 'The value of default measurement of LinearRegression is ', lr.score(
    X_test, y_test)
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absolute_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print 'The value of R-Squared of LinearRegression is', r2_score(y_test,
                                                                lr_y_pred)
print 'The mean squared error of LinearRegression is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred))
print 'The mean absolute error of LinearRegression is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred))

# 评估SGDRegressor
print 'The value of default measurement of SGDRegressor is ', sgdr.score(
    X_test, y_test)
# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absolute_error用于回归性能的评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print 'The value of R-Squared of SGDRegressor is', r2_score(y_test,
                                                            sgdr_y_pred)
print 'The mean squared error of SGDRegressor is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_pred))
print 'The mean absolute error of SGDRegressor is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_pred))