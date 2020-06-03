# *-utf-8-*
# GBDT algrithom implementation
# Date: 2020.06.01
# Author: Tsai
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor  ##regressor

# get training data
train_feature = np.genfromtxt('D:\\uidq1860\\Work-CJW\\LearningFiles\\GitHub\\DataSets\\GBDT_train_feat.txt')
num_feature = len(train_feature[0])
train_feature = pd.DataFrame(train_feature)

train_label = train_feature.iloc[:,num_feature-1]
train_feature = train_feature.iloc[:,0:num_feature-1]
# print(train_feature)  ## use to show the train features

# get testing data
test_feature = np.genfromtxt('D:\\uidq1860\\Work-CJW\\LearningFiles\\GitHub\\DataSets\\GBDT_test_feat.txt')
num_feature = len(test_feature[0])
test_feature = pd.DataFrame(test_feature)

test_label = test_feature.iloc[:,num_feature-1]
test_feature = test_feature.iloc[:,0:num_feature-1]
# print(test_label)  ## debug to show 

# establish GBDT model
gbdt = GradientBoostingRegressor(
    loss = 'ls',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False
)

gbdt.fit(train_feature,train_label)
pred = gbdt.predict(test_feature)
total_error = 0

for i in range(pred.shape[0]):
    print('pred:',pred[i],'label:',test_label[i])
print('average_squared_error:',np.sqrt(((pred-test_label)**2).mean()))