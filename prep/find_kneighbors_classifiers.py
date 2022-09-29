from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from find_classifiers import create_base, save_classifier_if_diverse


def generate() -> List[str]:
    nr_items = 200
    X_test, X_train, Y_train, Y_test = create_base(nr_items)

    predicts: List[int] = []
    params: List[str] = []
    for n_neighbors in range(5, 31):
        for weights in ["uniform", "distance"]:
            for algorithm in ["ball_tree", "kd_tree", "brute"]:
                for p in [1, 2, 3, 4, 5]:
                    knn = KNeighborsClassifier(
                        n_neighbors=n_neighbors,
                        weights=weights,
                        algorithm=algorithm,
                        p=p,
                    )
                    knn.fit(X_train, Y_train)
                    a_predict = knn.predict(X_test)
                    a_score = knn.score(X_test, Y_test)

                    save_classifier_if_diverse(
                        classif_score=a_score,
                        classif_predict=a_predict,
                        predicts=predicts,
                        params=params,
                        nr_items=nr_items,
                        string_parameters=str(
                            str(n_neighbors)
                            + ","
                            + weights
                            + ","
                            + algorithm
                            + ","
                            + str(p),
                        ),
                    )
        print(n_neighbors)
    return params


result: List[str] = []
for i in range(0, 5):
    result = result + generate()
counter = Counter(result)
print(counter)
result_most_common = [c for c in counter if counter[c] > 1]
filepath = Path("prep/data/knn.csv")
pd.DataFrame(result_most_common).to_csv(filepath, index=False)
