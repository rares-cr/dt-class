import numpy as np
from collections import Counter

class Node:
    """
    This class is used to help create the tree-like structure of a decision tree.
    It takes several parameters: feature (the best feature to split by), threshold (the best threshold to split by),
    left (left child of the node), right (right child of the node), value (value of the node in the case of leaf nodes)

    The method leaf_node is used to check whether a node is a leaf node
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def leaf_node(self):
        return self.value is not None


class DecisionTree:
    """
    This code defines a class for a decision tree algorithm
    """
    def __init__(
        self,
        min_samples_split=2,
        max_depth=100,
        criterion="gini",
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
    ):
        """
        This is the constructor of the DecisionTree class. It allows the user to create an instance of class DecisionTree with default values of user-specified values
        The constructor also initializes the variable tree to None, and calls the _validate_inputs method to ensure the input parameters are valid
        :param min_samples_split: minimum number of samples in order to perform a split
        :param max_depth: maximum depth of the tree
        :param criterion: the splitting criterion: 'gini' or 'entropy'
        :param min_impurity_decrease: minimum decrease in impurity for a new split
        :param min_samples_leaf: minimum number of samples in a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

        # Validate the input values to make sure they meet the constraints
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Checks whether the input values meet the constraints
        :return: raises ValueErrors and TypeErrors when conditions are not met
        """
        # checks that max_depth is an integer >= 1
        if self.max_depth < 1:
            raise ValueError('max_depth value must be >= 1')
        if not isinstance(self.max_depth, int):
            raise TypeError('max_depth must be of type int')

        # checks that min_samples_split is an integer >= 2
        if self.min_samples_split < 2:
            raise ValueError('min_samples_split must be >= 2')
        if not isinstance(self.min_samples_split, int):
            raise TypeError('min_samples_split must be of type int')

        # checks that min_impurity_decrease is a float >= 0
        if self.min_impurity_decrease < 0:
            raise ValueError('min_impurity_decrease must be >= 0')
        if not isinstance(self.min_impurity_decrease, float):
            raise TypeError('min_impurity_decrease must be of type float')

        # checks that min_samples_split is an integer >= 1
        if self.min_samples_leaf < 1:
            raise ValueError('min_samples_leaf must be >= 1')
        if not isinstance(self.min_samples_leaf, int):
            raise TypeError('min_samples_leaf must be of type int')

        # checks that criterion is a string
        if not isinstance(self.criterion, str):
            raise TypeError('criterion must be of type str')

    def fit(self, data, y):
        """
        This method allows the user to grow the decision tree. The function calls the _growTree function which builds the tree
        :param data: the actual observations used in training a decision tree
        :param y: the respective labels used in training
        """
        # checks whether there are as many labels as observations, returns an error otherwise
        if data.shape[0] != len(y):
            raise ValueError('X and y need to be the same size')
        # checks whether y is a 1-D array, returns an error otherwise
        if y.ndim != 1:
            raise ValueError('y must be a 1D array')

        self.n_features = data.shape[1]
        self.tree = self._growTree(data, y)

    def _growTree(self, data, y, depth=0):
        """
        Recursive function that grows the tree by finding the best split, splitting the tree and doing the same for each branch until some conditions are met
        :param data: the actual observations used in training a decision tree
        :param y: the respective labels used in training
        :param depth: the depth of the decision tree. Starts at 0 and increments after every iteration
        :return: an instance of class Node
        """
        n_samples = data.shape[0]

        # checks whether max_depth has been reached, or all labels are the same, or number of samples < min_samples_split,
        # or length of y < min_samples_leaf. Return a leaf node with value equal to the most common label in y
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or n_samples < self.min_samples_split
            or len(y) < self.min_samples_leaf
        ):
            leaf_value = self._common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._findBestSplit(data, y) # retrieves the best feature and threshold to split by

        # the data is split based on feature and threshold
        left, right, y_left, y_right = self._splitTree(
            data, y, best_feature, best_threshold
        )

        # checks whether min_impurity_decrease has been user-specified
        if self.min_impurity_decrease != 0:
            # calculates the change in impurity
            impurity = self._nodeImpurity(y)
            impurity_decrease = impurity - self._impurity(y_left, y_right)
            # returns a leaf node if change in impurity < min_impurity_decrease
            if impurity_decrease < self.min_impurity_decrease:
                leaf_value = self._common_label(y)
                return Node(value=leaf_value)

        # recursively grows left and right branches, depth is incremented
        left_tree = self._growTree(left, y_left, depth + 1)
        right_tree = self._growTree(right, y_right, depth + 1)

        # returns an instance of class Node with inputs for feature, threshold, left and right
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_tree,
            right=right_tree,
        )

    def _common_label(self, y):
        """
        This function returns the most common label
        :param y: the array of labels
        :return: the most common label
        """
        return Counter(y).most_common(1)[0][0]

    def _findBestSplit(self, data, y):
        """
        This function iterates through the entire dataset column by column and finds the best split.
        The best feature and best threshold are stored and checked against new values at every iteration.
        If new best parameters are found, these are stored in the best feature and best threshold variables
        :param data: the actual data
        :param y: the class labels
        :return: best_feature, best_threshold
        """

        # best impurity, feature and threshold are initialized
        best_impurity = float("inf")
        best_feature = None
        best_threshold = None

        # iterate through all column
        for feature in range(self.n_features):
            col = data[:, feature]
            values = np.unique(col) # checks the number of unique values in the column

            if len(values) == 1:
                # go to next feature if all values are the same
                continue
            if len(values) == 2:
                # calculate the impurity when value are binary
                mask = col == values[1]
                y_left = y[mask]
                y_right = y[~mask]
                impurity = self._impurity(y_left, y_right)

                # checks whether new impurity is smaller than previous best. If True, save the parameters as best
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = values[1]

            else:
                # if data is not binary or single value, iterate through all unique values in column
                for threshold in np.unique(col):
                    # splits the tree using _splitTree method
                    left, right, y_left, y_right = self._splitTree(
                        data, y, feature, threshold
                    )
                    # new impurity is calculated
                    impurity = self._impurity(y_left, y_right)

                    # checks whether new impurity is smaller than previous best. If True, save the parameters as best
                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def _splitTree(self, data, y, feature, threshold):
        """
        This function is used to split the tree into left and right sets.
        It can handle categorical data by checking the instance of the data.
        A different splitting method is used in the case of categorical data
        :param data: actual observations
        :param y: corresponding class labels
        :param feature: feature to split by
        :param threshold: threshold to split by
        :return: left and right datasets, left and right labels
        """
        values = data[:, feature]

        # if values are numerical, mask of values < threshold is created and the data is split using the mask
        if isinstance(values[1], (int, float)):
            mask = values < threshold

            left = data[mask]
            right = data[~mask]
            y_left = y[mask]
            y_right = y[~mask]

        # if values are categorical, mask of values == threshold is created and the data is split using the mask
        else:
            mask = values == threshold

            left = data[mask]
            right = data[~mask]
            y_left = y[mask]
            y_right = y[~mask]

        return left, right, y_left, y_right

    def _impurity(self, y_left, y_right):
        """
        This function calculates the impurity created by a split
        :param y_left: left branch labels
        :param y_right: right branch labels
        :return: value of gini or entropy
        """
        len_left = len(y_left)
        len_right = len(y_right)

        # checks if any of the labels sets have no values, return infinity if so
        if len_left == 0 or len_right == 0:
            return float("inf")

        # calculates gini index by calling the _gini function on the left and right labels sets
        # returns the value of the gini index
        if self.criterion == "gini":
            gini_left = self._gini(y_left)
            gini_right = self._gini(y_right)

            # uses the formula to calculate the resulting impurity at a split
            gini = (len_left / (len_left + len_right)) * gini_left + (
                len_right / (len_left + len_right)
            ) * gini_right

            return gini

        # calculates entropy by calling the _entropy function on the left and right labels sets
        # returns the value of the entropy
        elif self.criterion == "entropy":
            entropy_left = self._entropy(y_left)
            entropy_right = self._entropy(y_right)

            # uses the formula to calculate the resulting impurity at a split
            entropy = (len_left / (len_left + len_right)) * entropy_left + (
                len_right / (len_left + len_right)
            ) * entropy_right

            return entropy

        else:
            # raise ValueError if user-specified criterion is not supported
            raise ValueError(f"Criterion {self.criterion} not supported.")

    def _nodeImpurity(self, y):
        """
        Calculates the impurity of a node based on labels
        :param y: class labels
        :return: impurity in a node; depends on used criterion
        """
        if self.criterion == "gini":
            gini = self._gini(y)
            return gini

        elif self.criterion == "entropy":
            entropy = self._entropy(y)
            return entropy

        else:
            # raise ValueError if user-specified criterion is not supported
            raise ValueError(f"Criterion {self.criterion} not supported.")

    def _entropy(self, y):
        """
        Calculates the entropy of a set of labels
        :param y: class labels
        :return: entropy value
        """
        value, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _gini(self, y):
        """
        Calculates the gini index of a set of labels
        :param y: class labels
        :return: gini value
        """
        gini = 1

        for label in np.unique(y):
            probs = (y == label).sum() / len(y)
            gini -= probs**2

        return gini

    def predict(self, X):
        """
        This method produces the predictions for a given set of data
        :param X: observations that need to be predicted
        :return: an array containing class labels predictions
        """
        # checks whether input data has the same number of features as data used to train the tree, raise a ValueError otherwise
        if X.shape[1] != self.n_features:
            raise ValueError('Input array does not match characteristics of training data')
        # initialise empty list with length of input data
        num_predictions = len(X)
        y_pred = np.empty(num_predictions)
        # iterate for every observation
        for i in range(num_predictions):
            node = self.tree
            # while loop that iterates until a leaf node is reached
            while not node.leaf_node():
                # checks whether the corresponding value of X at row i and the best feature stored in the Node is <= threshold stored in the node
                if X[i, node.feature] <= node.threshold:
                    # go to left branch if True
                    node = node.left
                else:
                    # right branch if False
                    node = node.right
            # value of leaf node is stored in the predictions array at position i
            y_pred[i] = int(node.value)
        return y_pred

