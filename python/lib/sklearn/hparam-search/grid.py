#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

from sklearn.datasets import fetch_20newsgroups
import numpy as np

news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
# 对前3000条新闻文本进行数据分割，25%文本用作未来测试
X_train, X_test, y_train, y_test = train_test_split(
    news.data[:3000], news.target[:3000], random_state=33, test_size=0.25)

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# 使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(
    stop_words='english', analyzer='word')), ('svc', SVC())])

# 这里需要试验的2个超参数的个数分别是4、3，svc_gamma的参数共有10^-2，10^-1...
# 这样一共有3*4=12种超参数的组合，12个不同参数的模型
parameters = {
    'svc__gamma': np.logspace(-2, 1, 4),
    'svc__C': np.logspace(-1, 1, 3)
}

# 导入网格搜索模块
from sklearn.grid_search import GridSearchCV

# 将12组参数组合以及初始化的Pipeline包括3折交叉验证要求全部告知GridSearchCV，务必注意refit = True
# 设置refit = True，那么程序将会以交叉验证训练集得到的最佳超参数，重新对所有可用的训练集与验证集进行，
# 作为最终用于评估性能参数的最佳模型的参数
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索
%time _ = gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性
print gs.score(X_test, y_test)
