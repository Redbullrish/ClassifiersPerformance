import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


def read_data():
    X, y = load_iris(return_X_y=True)
    return X, y

def decision_tree(folds):
    res = []
    depths = [9,12,15,18,21]
    X, y = read_data()
    for d in depths:
        tree = DecisionTreeClassifier(criterion="entropy",max_depth=d)
        scores = cross_val_score(estimator=tree, X=X, y=y, cv=folds, scoring="neg_mean_absolute_error")
        res.append((d,1+scores.mean()))
    return res

def logistic_regression(folds):
    res = []
    alpha = [10**-6,10**5,10**-4,10**-2,1]
    X, y = read_data()

    for a in alpha:
        clf = SGDClassifier(loss="log",alpha=a,l1_ratio=0)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=folds, scoring="neg_mean_absolute_error")
        res.append((a,1+scores.mean()))

    return res

def kNN(folds):
    res = []
    neighbors = [3,5,7,9,11]
    X, y = read_data()

    for n in neighbors:
        clf = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=folds, scoring="neg_mean_absolute_error")
        res.append((n,1+scores.mean()))

    return res

def neural_network(folds):
    res = []
    alpha = [0.25,0.5,0.75,1,1.25]
    X, y = read_data()

    for a in alpha:
        clf = MLPClassifier(solver='lbfgs', alpha=a,hidden_layer_sizes=(5, 2), random_state=1)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=folds, scoring="neg_mean_absolute_error")
        res.append((a,1+scores.mean()))

    return res


def main():
    dtree_scores = decision_tree(5)
    log_scores = logistic_regression(5)
    kNN_scores = kNN(5)
    net_scores = neural_network(5)
    print(dtree_scores, log_scores, kNN_scores,net_scores)


if __name__ == '__main__':
    main()
