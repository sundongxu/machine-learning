#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

import pandas as pd
titanic = pd.read_csv(
    'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 人工选取pclass、age以及sex作为判别乘客是否能够生还的特征
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 对于缺失的年龄信息，使用平均年龄代替
X['age'].fillna(X['age'].mean(), inplace=True)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

# 使用单一决策树进行模型训练以及预测分析
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

# 性能评估比较
from sklearn.metrics import classification_report

# 输出单一决策树在测试机上的分类准确性、以及更详细的精确率、召回率、F1指标
print 'The Accuracy of Decision Tree is: ', dtc.score(X_test, y_test)
print classification_report(dtc_y_pred, y_test)

# 输出随机森林分类器在测试集上的分类准确性，以及更详细的精确率、召回率、F1指标
print 'The Accuracy of Random Forest is: ', rfc.score(X_test, y_test)
print classification_report(rfc_y_pred, y_test)

# 输出梯度上升决策树在测试集上的分类准确性，以及更详细的精确率、召回率、F1指标
print 'The Accuracy of Gradient Boosting Tree is: ', gbc.score(X_test, y_test)
print classification_report(gbc_y_pred, y_test)

