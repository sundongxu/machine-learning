#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

import pandas as pd
titanic = pd.read_csv(
    'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X = titanic[['sex', 'age', 'pclass']]
y = titanic['survived']

X['age'].fillna(X['age'].mean(), inplace=True)

# 数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

# 使用sklearn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 采用默认配置的随机森林分类器对测试集进行预测
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print 'The accuracy of RandomForest Classifier on testing set:', rfc.score(X_test, y_test)

# 采用默认配置的XGBoost模型对相同的测试集进行预测
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
print 'The accuracy of XGBoost Classifier on testing set:', xgbc.score(X_test, y_test)
