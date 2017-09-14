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

# 从sklearn.svm中导入支持向量机（回归）模型
from sklearn.svm import SVR

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_pred = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_pred = poly_svr.predict(X_test)


# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_pred = rbf_svr.predict(X_test)

# 使用R-square、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 输出linear SVR的性能评估指标
print 'R-squared value of linear SVR is ', linear_svr.score(X_test, y_test)
print 'The value of R-Squared of Linear SVR is', r2_score(y_test,
                                                                linear_svr_y_pred)
print 'The mean squared error of LinearRegression is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_pred))
print 'The mean absolute error of LinearRegression is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_pred))

# 输出poly SVR的性能评估指标
print 'R-squared value of linear SVR is ', poly_svr.score(X_test, y_test)
print 'The value of R-Squared of Linear SVR is', r2_score(y_test,
                                                                poly_svr_y_pred)
print 'The mean squared error of LinearRegression is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_pred))
print 'The mean absolute error of LinearRegression is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_pred))

# 输出rbf SVR的性能评估指标
print 'R-squared value of linear SVR is ', rbf_svr.score(X_test, y_test)
print 'The value of R-Squared of Linear SVR is', r2_score(y_test,
                                                                rbf_svr_y_pred)
print 'The mean squared error of LinearRegression is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_pred))
print 'The mean absolute error of LinearRegression is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_pred))
