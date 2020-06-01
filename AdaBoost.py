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
plt.scatter(X[:50,0],X[:50,1],label='0')
plt.scatter(X[50:,0],X[50:,1],label='1')
plt.legend()
plt.show()  # use plt.show() to watch the image classify result

class AdaBoost:
    def __init__(self, n_estimators=50,learning_rate=0.1):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate

    def init_args(self,datasets, labels):
        self.X = datasets
        self.Y = labels
        self.M, self.N = datasets.shape

        # soft classifiers's number and sets
        self.clf_sets = []
        # init weights
        self.weights = [1.0/self.M]*self.M
        # G(x) weight alpha
        self.alpha = []

    def _G(self,features,labels,weights):
        m = len(features)
        error = 100000.0 # much large
        best_v = 0.0
        # only 1 dimension features
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate)//self.learning_rate
        # print('n_step:{}'.format(n_step))
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i
            if v not in features:
                # wrong classified calcu.
                compare_array_positive = np.array(
                    [1 if features[k]> v else -1 for k in range(m)]
                )
                weight_error_positive = sum([
                    weights[k] for k in range(m)
                    if compare_array_positive[k] != labels[k]
                ])

                compare_array_nagetive = np.array(
                    [-1 if features[k] > v else 1 for k in range(m)]
                )
                weight_error_nagetive = sum([
                    weights[k] for k in range(m)
                    if compare_array_nagetive[k] != labels[k]
                ])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'

                # print('v:{} error:{}'.format(v,weight_error))
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array

    # calcu. alpha
    def _alpha(self, error):
        return 0.5*np.log((1 - error)/error)

    # normalize factor
    def _Z(self, weights, a, clf):
        return sum([
            weights[i] * np.exp(-1 * a * self.Y[i] * clf[i])
            for i in range(self.M)
        ])

    # weights modify
    def _w(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-1 * a * self.Y[i] * clf[i]) / Z

    # G(x) linear combine
    def _f(self, alpha, clf_sets):
        pass

    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, y):
        self.init_args(X, y)

        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # according to dimensions, to choose min. error
            for j in range(self.N):
                features = self.X[:, j]
                # classify threshold, classification error, classify results
                v, direct, error, compare_array = self._G(features, self.Y, self.weights)
                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                if best_clf_error == 0:
                    break
            # calcu. G(x) weight alpha
            a = self._alpha(best_clf_error)
            self.alpha.append(a)
            # record classifiers
            self.clf_sets.append((axis, best_v, final_direct))
            # normalized factors
            Z = self._Z(self.weights, a, clf_result)
            # weights modify
            self._w(a, clf_result, Z)

    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        # sign
        return 1 if result > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1

        return right_count / len(X_test)

X = np.arange(10).reshape(10,1)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
print(X)

