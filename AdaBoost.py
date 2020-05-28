# *-utf-8-*
# AdaBoost algrithom implementation
# Date: 2020.05.28
# Author: Tsai
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#matplotlib inline

# creat data
def creat_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width','label']
    data = np.array(df.iloc[:100,[0,1,-1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    return data[:,:2], data[:,-1]

X, y = creat_data()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
