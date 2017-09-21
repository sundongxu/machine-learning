#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 从sklearn.datasets中导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups
# 从互联网上即时下载新闻样本，subset='all'参数代表下载全部近两万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')

# 从sklearn.cross_validation导入train_test_split模块用于分割数据集
from sklearn.cross_validation import train_test_split
# 对news中的数据data进行分割，25%的文本用于测试集，75%作为训练集
X_train, X_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33)

# 从sklearn.feature_extraction里导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()

# 只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

# 从sklearn.naive_bayes中导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
# 使用朴素贝叶斯分类器，对CountVectorizer（不去除停用词）后的训练样本进行参数学习
mnb_count.fit(X_count_train, y_train)
# 将预测结果存储在变量y_count_pred中
y_count_pred = mnb_count.predict(X_count_test)

# 输出模型准确性结果
print 'The accuracy of classifying 20newsgroup using Naive Bayes(CountVectorizer without filtering stopwords):', mnb_count.score(
    X_count_test, y_test)

from sklearn.metrics import classification_report
print classification_report(
    y_test, y_count_pred, target_names=news.target_names)

# 从sklearn.feature_extraction.text里导入TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()

# 使用tfidf的方式，将原始文本和测试文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

# 依然使用朴素贝叶斯分类器在相同的训练集和测试集上对新的特征量化方式进行性能评估
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, y_train)

y_tfidf_pred = mnb_tfidf.predict(X_tfidf_test)

print 'The accuracy of classifying 20newsgroups with Naive Bayes(TfidfVectorizer without filtering stopwords):', mnb_tfidf.score(
    X_tfidf_test, y_test)

print classification_report(
    y_test, y_tfidf_pred, target_names=news.target_names)

# 分别使用停止词配置初始化CountVectorizer和TfidfVectorizer
count_filter_vec = CountVectorizer(analyzer='word', stop_words='english')
tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')

X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
y_count_filter_pred = mnb_count_filter.predict(X_count_filter_test)

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
y_tfidf_filter_pred = mnb_tfidf_filter.predict(X_tfidf_filter_test)

print 'The accuracy of classifying 20newsgroups with Naive Bayes(CountVectorizer with filtering stopwords):', mnb_count_filter.score(
    X_count_filter_test, y_test)

print classification_report(
    y_test, y_count_filter_pred, target_names=news.target_names)

print 'The accuracy of classifying 20newsgroups with Naive Bayes(TfidfVectorizer with filtering stopwords):', mnb_tfidf_filter.score(
    X_tfidf_filter_test, y_test)

print classification_report(
    y_test, y_tfidf_filter_pred, target_names=news.target_names)
