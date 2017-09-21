#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 从sklearn.datasets里导入手写体数字加载器
from sklearn.datasets import load_digits
digits = load_digits()
digits.shape

# 从sklearn.cross_validation中导入train_test_split用于训练/测试数据分割
from sklearn.cross_validation import train_test_split
# 随机选取75%的数据作为训练样本，其余25%作为测试样本
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=33)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

# 从sklearn中导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.svm中导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC

# 仍然需要对训练和测试的特征数据进行标准化处理
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化线性假设支持向量机分类器LinearSVC
lsvc = LinearSVC()
# 模型训练，即拟合fit
lsvc.fit(X_train, y_train)
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中
y_predict = lsvc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评
# score函数传入参数测试样本特征X_test和真实标记y_test的原因：
# 根据训练样本特征X_train和样本标记y_train训练学习得到模型
# 计算模型在测试样本上的分类准确性时，传入测试样本特征X_test，让模型知道该预测哪些样本，从而计算对应的预测标记y_predict
# 传入测试样本真实标记y_test后，将真实标记和预测标记进行比较得出准确性score
print 'The Accuracy of Linear SVC is ', lsvc.score(X_test, y_test)

# 使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print classification_report(
    y_test, y_predict, target_names=digits.target_names.astype(str))
