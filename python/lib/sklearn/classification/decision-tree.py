#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file

# 导入pandas用于数据分析
import pandas as pd
# 利用pandas的read_csv模块直接从互联网上收集泰坦尼克号乘客数据
titanic = pd.read_csv(
    'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 观察前几行数据，发现数据各异，数值型、类别性，甚至还有缺失值missing value
titanic.head()

# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），使用info()查看数据统计特性
titanic.info()

# 机器学习中，特征选择是十分重要的一环
# 这很可能需要一些背景知识、领域知识
# 根据我们对这场事故的了解，sex、age、pclass这些特征都很有可能是决定幸免于否的关键因素
X = titanic[['sex', 'age', 'pclass']]
y = titanic['survived']

# 对当前选择的特征进行探查
X.info()

# 控制台输出如下：
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 3 columns):
# sex       1313 non-null object
# age       633 non-null float64
# pclass    1313 non-null object
# dtypes: float64(1), object(2)
# memory usage: 30.8+ KB

# 根据以上输出，设计如下几个数据处理的任务
#（1）age这个数据列，只有633个，需要补完
#（2）sex与pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替
# 首先我们补充age里面的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略

X['age'].fillna(X['age'].mean(), inplace=True)

# 对补完整的列重新检查
X.info()

# 控制台输出如下：可见age特征列得到了补全
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1313 entries, 0 to 1312
# Data columns (total 3 columns):
# sex       1313 non-null object
# age       1313 non-null float64
# pclass    1313 non-null object
# dtypes: float64(1), object(2)
# memory usage: 30.8+ KB

# 数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33)

# 使用sklearn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

# 转换特征后，但凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print vec.feature_names_

X_test = vec.transform(X_test.to_dict(orient='record'))

# 从sklearn.tree中导入决策树分类器，初始化->拟合->预测
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)

# 性能评估
from sklearn.metrics import classification_report
print 'The Accuracy of Decision Tree Classfier is ', dtc.score(X_test, y_test)
print classification_report(
    y_test, y_predict, target_names=['died', 'survived'])
