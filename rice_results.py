import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from final_dt import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import time
import resource
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# df preprocessing
df = pd.read_csv("rice.csv", header=0)
df["Class"] = df["Class"].map({"Cammeo": 0, "": 1, "Osmancik": 2})
y = df["Class"]
x = df.drop(["Class"], axis=1)
y = y.to_numpy()
x = x.to_numpy()
np.set_printoptions(suppress=True)


# split data using sklearn
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=14
)

# check model performance
results = []


def get_results(
    clf,
    dataset_name,
    implementation,
    range1,
    range2,
    range3,
    X_train,
    y_train,
    X_test,
    y_test,
):
    for criterion in ["gini", "entropy"]:
        for min_sample_split in range(2, range1, 400):
            for max_depth in range(1, range2):
                for min_impurity_decrease in np.linspace(0, 0.5, 4):
                    for min_samples_leaf in range(1, range3, 400):
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
    "rice_dataset",
    "own_implementation",
    1000,
    8,
    1000,
    X_train,
    y_train,
    X_test,
    y_test,
)
get_results(
    DecisionTreeClassifier,
    "rice_dataset",
    "sklearn",
    1000,
    8,
    1000,
    X_train,
    y_train,
    X_test,
    y_test,
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

results.to_csv("DT1_results.csv", index=False)
