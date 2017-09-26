#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

import pandas as pd
titanic = pd.read_csv(
    'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

print len(vec.feature_names_)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

# 从sklearn中导入特征选择器
from sklearn import feature_selection
# 筛选前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)

X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)

# 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并做图展示性能随特征筛选比例的变化
from sklearn.cross_validation import cross_val_score
import numpy as np
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(
        feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())

print results

# 找到体现最佳性能的特征筛选的百分比
opt = np.where(results == results.max())[0]
print np.where(results == results.max())
print 'Optimal number of features %d', percentiles[opt]

import pylab as pl
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentiles = 7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)

# （1）经过初步的特征处理，最终的训练、测试数据均有474个维度
# （2）利用全部474个维度的特征用于训练决策树模型进行分类预测，在测试集上的准确率约为81.76%
# （3）如果筛选前20%维度的特征，在相同的模型配置下进行预测，那么在测试集上的准确性约为82.37%
# （4）按照固定的间隔采用不同百分比的特征进行训练与测试，那么通过交叉验证得出的准确性有很大波动，并且最好的模型性能表现在
#     选取前7%维度的特征的时候
# （5）如果使用前7%维度的特征，那么最终决策树模型可以在该分类预测任务的测试集上表现出85.71%的准确性！