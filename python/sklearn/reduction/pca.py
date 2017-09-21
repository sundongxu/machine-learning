#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

digits_train = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
    header=None)
digits_test = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
    header=None)

# 分割训练数据的特征向量和标记
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

# 从sklearn.decomposition导入PCA
from sklearn.decomposition import PCA
# 初始化一个可以将高维度特征想来那个（64维）压缩至2个维度的PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_train)

# 显示10类手写体数字图片经PCA压缩后的2维空间分布
from matplotlib import pyplot as plt


def plot_pca_scatter():
    colors = [
        'black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan',
        'orange', 'gray'
    ]
    for i in xrange(len(colors)):
        px = X_pca[:, 0][y_train.as_matrix() == i]
        py = X_pca[:, 1][y_train.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


plot_pca_scatter()

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 以下将对比原始维度特征与经过PCA压缩重建之后的图像特征
# 在相同配置的支持向量机（分类）模型上识别性能的差异
# 导入基于线性核的支持向量机分类器
from sklearn.svm import LinearSVC

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在y_predict
svc = LinearSVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# 使用PCA将原64维的图像数据压缩到20个维度
estimator = PCA(n_components=20)

# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化（transform）
pca_X_test = estimator.transform(X_test)

# 使用默认配置初始化LinearSVC，对压缩过后的20维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_y_predict中
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_pred = pca_svc.predict(pca_X_test)

from sklearn.metrics import classification_report
# 对使用原始图像高维像素特征训练的支持向量机分类器的性能评估
print svc.score(X_test, y_test)
print classification_report(
    y_test, y_pred, target_names=np.arange(10).astype(str))

# 对使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能评估
print pca_svc.score(pca_X_test, y_test)
print classification_report(
    y_test, pca_y_pred, target_names=np.arange(10).astype(str))


