#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 从sklearn.datasets里导入新闻数据抓取器fetch_20newgroups
from sklearn.datasets import fetch_20newsgroups
# 下载数据
news = fetch_20newsgroups(subset='all')
# 查验数据规模和细节
print len(news.data)
print news.data[0]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33)

# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 模型使用默认配置初始化
mnb = MultinomialNB()
# 模型拟合
mnb.fit(X_train, y_train)
# 预测
y_predict = mnb.predict(X_test)

# 使用sklearn.metrics里面的classification_report模块用于详细分类性能报告
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=news.target_names)