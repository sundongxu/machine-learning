#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 导入pandas和numpy工具包
import pandas as pd
import numpy as np

# 创建特征列表
column_names = [
    'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
    'Uniformity of Cell Shape', 'Marginal Adhesion',
    'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    'Normal Nucleoli', 'Mitoses', 'Class'
]

# 使用pandas.read_csv函数从互联网读取指定数据
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=column_names)
# 将？替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带缺失值的数据(只要有一个维度缺失)
data = data.dropna(how='any')
data.shape

# 使用sklearn.cross_validation里的train_test_split模块用于分割数据
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data[column_names[1:10]],
    data[column_names[10]],
    test_size=0.25,
    random_state=33)
y_train.value_counts()

# 从sklearn中导入数据标准化模块
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Logistics回归是一种分类模型，而不是回归！
from sklearn.linear_model import SGDClassifier

# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型初始化
lr = LogisticRegression()
sgdc = SGDClassifier()

# 训练拟合、预测
lr.fit(X_train, y_train)
sgdc.fit(X_train, y_train)

lr_y_predict = lr.predict(X_test)
sgdc_y_predict = sgdc.predict(X_test)

# 使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print 'The Accuracy of LR Classifier is ', lr.score(X_test, y_test)
print classification_report(
    y_test, lr_y_predict, target_names=['Benign', 'Malignant'])

print 'The Accuracy of SGD Classifier is ', sgdc.score(X_test, y_test)
print classification_report(
    y_test, sgdc_y_predict, target_names=['Benign', 'Malignant'])