#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 从sklearn.datasets中导入iris数据加载器
from sklearn.datasets import load_iris
iris = load_iris()

iris.data.shape

# 查看数据说明
print iris.DESCR

# 从sklearn.cross_validation里选择导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split
# 从使用train_test_split，利用随机种子random_state采样25%的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=33)

# 从sklearn.preprocessing里选择导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier，即K近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 对训练、测试的特征数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 模型拟合、预测
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = kvc.predict(X_test)

print 'The Accuracy of K-Nearest Neighbor Classfier is ', knc.score(X_test,
                                                                    y_test)
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=iris.target_names)