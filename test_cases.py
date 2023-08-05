import pandas as pd
import numpy as np
import pytest
from final_dt import DecisionTree, Node

#df preprocessing
df = pd.read_csv('Iris.csv', header=0)
df['Species'] = df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
y = df['Species']
x = df.drop(['Species', 'Id'], axis=1)
y = y.to_numpy()
x = x.to_numpy()

def test_bestsplit1():
    """
    Checks whether _findBestSplit function returns a feature number within the numbers of features which the data has
    """
    clf = DecisionTree()
    clf.fit(x, y)
    feature, threshold = clf._findBestSplit(x, y)
    assert feature in range(x.shape[1])

def test_bestsplit2():
    """
    Checks whether _findBestSplit function returns a threshold within that exists in the particular column of the data
    """
    clf = DecisionTree()
    clf.fit(x, y)
    feature, threshold = clf._findBestSplit(x, y)
    assert threshold in x[:,feature]


def test_bestsplit3():
    """
    Checks whether _findBestSplit function returns the feature as an instance of int
    """
    clf = DecisionTree()
    clf.fit(x, y)
    feature, threshold = clf._findBestSplit(x, y)
    assert isinstance(feature, int)


def test_bestsplit4():
    """
    Checks whether _findBestSplit function returns the threshold as a similar instance as the original data
    """
    clf = DecisionTree()
    clf.fit(x, y)
    feature, threshold = clf._findBestSplit(x, y)
    assert isinstance(threshold, type(x[1, feature]))

def test_split1():
    """
    Checks whether _splitTree function does not omit any rows from the original data when returning the left and right children
    """
    clf = DecisionTree()
    clf.fit(x,y)
    feature, threshold = clf._findBestSplit(x, y)
    left, right, y_left, y_right = clf._splitTree(x, y, feature, threshold)
    assert len(left) + len(right) == len(x)

def test_split2():
    """
    Checks whether _splitTree function does not omit any rows from the original data when returning the left and right labels
    """
    clf = DecisionTree()
    clf.fit(x, y)
    feature, threshold = clf._findBestSplit(x, y)
    left, right, y_left, y_right = clf._splitTree(x, y, feature, threshold)
    assert len(y_left) + len(y_right) == len(y)

def test_fit_inputs1():
    """
    Checks whether the fit function returns a ValueError when the lengths of labels and data are different
    """
    clf = DecisionTree()
    with pytest.raises(ValueError):
        clf.fit(x, y[0:100])

def test_fit_inputs2():
    """
    Checks whether the fit function returns a ValueError when the labels input is not a 1D array
    """
    clf = DecisionTree()
    with pytest.raises(ValueError):
        clf.fit(x, [[1,1,2,1], [1,2,2,1,2]])

def test_growtree1():
    """
    Checks whether the _growTree function returns an instance of class Node
    """
    clf = DecisionTree()
    clf.fit(x, y)
    tree = clf._growTree(x, y)
    assert isinstance(tree, Node)

def test_growtree2():
    """
    Checks whether the _growTree function returns a leaf node when all labels are the same
    """
    clf = DecisionTree()
    clf.fit(x, y)
    tree = clf._growTree(x, 150*[1])
    assert tree.value is not None

def test_growtree3():
    """
    Checks whether the _growTree function returns a leaf node when depth of tree > max_depth
    """
    clf = DecisionTree(max_depth=1)
    clf.fit(x, y)
    tree = clf._growTree(x, y, depth=2)
    assert tree.value is not None

def test_growtree4():
    """
    Checks whether the _growTree function returns a leaf node when samples to split < min_samples_split
    """
    clf = DecisionTree(min_samples_split=151)
    clf.fit(x, y)
    tree = clf._growTree(x, y)
    assert tree.value is not None

def test_growtree5():
    """
    Checks whether the _growTree function returns a leaf node when length of labels < min_samples_split
    """
    clf = DecisionTree(min_samples_leaf=151)
    clf.fit(x, y)
    tree = clf._growTree(x, y)
    assert tree.value is not None

def test_predict1():
    """
    Checks whether the predict function returns as many predictions as input observations
    """
    clf = DecisionTree()
    clf.fit(x[0:100],y[0:100])
    predictions = clf.predict(x[100:])
    assert len(predictions) == x[100:].shape[0]

def test_predict2():
    """
    Checks whether the predict function returns the same number of classes as in the data
    """
    clf = DecisionTree()
    clf.fit(x, y)
    predictions = clf.predict(x)
    assert len(np.unique(predictions)) == len(np.unique(y))

def test_predict3():
    """
    Checks whether the predict function raises a ValueError when the input data does not match the characteristics of the data the tree was trained on
    """
    clf = DecisionTree()
    clf.fit(x, y)
    with pytest.raises(ValueError):
        clf.predict(x[:,0:2])

def test_valid_parameters1():
    """
    Checks whether creating an instance of the DecisionTree class raises a ValueError when max_depth value is < 1
    """
    with pytest.raises(ValueError):
        clf = DecisionTree(max_depth=-1)

def test_valid_parameters2():
    """
    Checks whether creating an instance of the DecisionTree class raises a TypeError when max_depth value is not of type int
    """
    with pytest.raises(TypeError):
        clf = DecisionTree(max_depth=1.2)

def test_valid_parameters3():
    """
    Checks whether creating an instance of the DecisionTree class raises a ValueError when min_samples_split value is < 2
    """
    with pytest.raises(ValueError):
        clf = DecisionTree(min_samples_split=0)

def test_valid_parameters4():
    """
    Checks whether creating an instance of the DecisionTree class raises a TypeError when criterion is not of type str
    """
    with pytest.raises(TypeError):
        clf = DecisionTree(criterion=x)


list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        2, 2, 2, 2,2 ,2,2,2,2,2,2,2, 2, 2, 2,2 ,2,2,2,2,2,2,2, 2, 2, 2,2 ,2,2,2,2,2,2,2]

def test_entropy1():
    """
    Checks whether _entropy function calculates entropy correctly
    """
    clf = DecisionTree()
    assert clf._entropy([1, 1, 1, 1]) == 0

def test_entropy2():
    """
    Checks whether _entropy function calculates entropy correctly
    """
    clf = DecisionTree()
    assert round(clf._entropy([1, 1, 1, 2, 2, 2, 2, 2, 2, 2]), 2) == 0.88

def test_gini1():
    """
    Checks whether _gini function calculates gini index correctly
    """
    clf = DecisionTree()
    assert clf._gini(list) == 0.6666

def test_gini2():
    """
    Checks whether _gini function calculates gini index correctly
    """
    clf = DecisionTree()
    assert clf._gini([1,1,1,1]) == 0

def test_gini3():
    """
    Checks whether _gini function calculates gini index correctly
    """
    clf = DecisionTree()
    assert clf._gini([1,1,1,2,2,2]) == 0.5

def test_impurity1():
    """
    Checks whether _impurity function raises ValueError when criteria is not gini or entropy
    """
    clf = DecisionTree(criterion='abc')
    with pytest.raises(ValueError):
        clf._impurity([1, 1, 1, 2], [3, 1, 1])

def test_impurity2():
    """
    Checks whether _impurity function calculates impurity correctly for gini
    """
    clf = DecisionTree()
    assert round(clf._impurity(list, [1,1,1,2,2,2]),2) == 0.66

def test_impurity3():
    """
    Checks whether _impurity function calculates impurity correctly for entropy
    """
    clf = DecisionTree(criterion='entropy')
    assert round(clf._impurity([1,1,1,1], [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]), 2) == 0.63

def test_impurity4():
    """
    Checks whether _impurity function produces the right output when one of the lists is empty
    """
    clf = DecisionTree(criterion='entropy')
    assert clf._impurity([], [1, 2, 1, 2]) == float('inf')

def test_nodeImpurity1():
    """
    Checks whether _nodeImpurity function raises ValueError when criteria is not gini or entropy
    """
    clf = DecisionTree(criterion='xyz')
    with pytest.raises(ValueError):
        clf._nodeImpurity([1, 1, 1, 2, 2, 1, 1])

def test_nodeImpurity2():
    """
    Checks whether _nodeImpurity function returns an instance of float
    """
    clf = DecisionTree()
    assert isinstance(clf._nodeImpurity([1,2,2,1,2,1,1,1,1]), float)

def test_common_label1():
    """
    Checks whether _common_label function returns the right output
    """
    clf = DecisionTree()
    assert clf._common_label([1,1,1,1,1,1,2]) == 1

def test_common_label2():
    """
    Checks whether _common_label function returns the first element when all labels appear the same number of times
    """
    clf = DecisionTree()
    assert clf._common_label([2,1,2,1,1,1,2,2]) == 2

