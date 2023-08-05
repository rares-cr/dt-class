import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from final_dt import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import time
import resource
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import category_encoders as ce


# iris preprocessing
iris = pd.read_csv("Iris.csv", header=0)
iris["Species"] = iris["Species"].map(
    {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
)
iris_y = iris["Species"]
iris_x = iris.drop(["Species", "Id"], axis=1)
iris_y = iris_y.to_numpy()
iris_x = iris_x.to_numpy()

# split data using sklearn
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
    iris_x, iris_y, test_size=0.3, random_state=14
)

# heart preprocessing
heart = pd.read_csv("heart.csv", header=0)
encoder = ce.OrdinalEncoder(
    cols=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
)
heart1 = encoder.fit_transform(heart)

h_y = heart["HeartDisease"]
h_x = heart.drop(["HeartDisease"], axis=1)
h_y = h_y.to_numpy()
h_x = h_x.to_numpy()


h1_y = heart1["HeartDisease"]
h1_x = heart1.drop(["HeartDisease"], axis=1)
h1_y = h1_y.to_numpy()
h1_x = h1_x.to_numpy()


# split into train and test using sklearn
heart_X_train, heart_X_test, heart_y_train, heart_y_test = train_test_split(
    h_x, h_y, test_size=0.3, random_state=14
)
heart1_X_train, heart1_X_test, heart1_y_train, heart1_y_test = train_test_split(
    h1_x, h1_y, test_size=0.3, random_state=14
)

# check model performance - IRIS

results = []

# MY IMPLEMENTATION


def get_results(
    clf,
    dataset_name,
    implementation,
    range1,
    range2,
    range3,
    step3,
    X_train,
    y_train,
    X_test,
    y_test,
):
    for criterion in ["gini", "entropy"]:
        for min_sample_split in range(2, range1, 5):
            for max_depth in range(1, range2):
                for min_impurity_decrease in np.linspace(0, 1, 6):
                    for min_samples_leaf in range(1, range3, step3):
                        time_start = time.perf_counter()
                        dt = clf(
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_sample_split,
                            min_impurity_decrease=min_impurity_decrease,
                            min_samples_leaf=min_samples_leaf,
                        )
                        tree = dt.fit(X_train, y_train)
                        time_elapsed = round((time.perf_counter() - time_start), 5)
                        memMb = (
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 1024.0
                            / 1024.0
                        )

                        y_pred = dt.predict(X_test)

                        acc = round(accuracy_score(y_test, y_pred), 2)
                        pre = round(
                            precision_score(
                                y_test,
                                y_pred,
                                average="weighted",
                                labels=np.unique(y_pred),
                            ),
                            2,
                        )
                        f1 = round(
                            f1_score(
                                y_test,
                                y_pred,
                                average="weighted",
                                labels=np.unique(y_pred),
                            ),
                            2,
                        )
                        rec = round(
                            recall_score(
                                y_test,
                                y_pred,
                                average="weighted",
                                labels=np.unique(y_pred),
                            ),
                            2,
                        )

                        row = [
                            dataset_name,
                            implementation,
                            criterion,
                            min_sample_split,
                            max_depth,
                            min_impurity_decrease,
                            min_samples_leaf,
                            acc,
                            pre,
                            f1,
                            rec,
                            time_elapsed,
                            memMb,
                        ]
                        results.append(row)


get_results(
    DecisionTree,
    "iris_dataset",
    "own_implementation",
    21,
    5,
    21,
    5,
    iris_X_train,
    iris_y_train,
    iris_X_test,
    iris_y_test,
)
get_results(
    DecisionTreeClassifier,
    "iris_dataset",
    "sklearn",
    21,
    5,
    21,
    5,
    iris_X_train,
    iris_y_train,
    iris_X_test,
    iris_y_test,
)
get_results(
    DecisionTree,
    "heart_dataset",
    "own_implementation",
    21,
    8,
    101,
    10,
    heart_X_train,
    heart_y_train,
    heart_X_test,
    heart_y_test,
)
get_results(
    DecisionTreeClassifier,
    "heart_dataset",
    "sklearn",
    21,
    8,
    101,
    10,
    heart1_X_train,
    heart1_y_train,
    heart1_X_test,
    heart1_y_test,
)

results = pd.DataFrame(
    results,
    columns=[
        "Dataset",
        "Implementation",
        "criterion",
        "min_sample_split",
        "max_depth",
        "min_impurity_decrease",
        "min_samples_leaf",
        "Accuracy",
        "Precision",
        "F1 Score",
        "Recall",
        "Runtime",
        "Memory Usage",
    ],
)

results.to_csv("DT_results.csv", index=False)
