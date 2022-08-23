import random
from typing import Dict, List, Tuple

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ResultsMetrics = Dict[str, str | List[int] | float]


class Classifier:
    def __init__(
        self,
        name: str,
        classif: BaseEstimator,
    ) -> None:

        self.clf = classif
        self.name = name
        self.params = classif.get_params()
        self.classifier_trained = False

        self.train_features = List[List[float]]
        self.train_classes = List[int]
        self.test_features = List[List[float]]
        self.test_classes = List[int]

        self.prediction: List[int] = []
        self.score: float = 0.0
        self.accuracy: float = 0.0
        self.recall: float = 0.0
        self.precision: float = 0.0
        self.f1: float = 0.0

    def fit(
        self,
        features_values: List[List[float]],
        classes_values: List[int],
    ) -> None:
        self.train_features = features_values
        self.train_classes = classes_values
        self.clf.fit(features_values, classes_values)  # type: ignore
        self.classifier_trained = True
        print("Clasificador " + self.name + " treinado.")

    def test(
        self,
        features_values: List[List[float]],
        classes_values: List[int],
    ) -> ResultsMetrics | None:
        if not self.classifier_trained:
            print("This classifier is not trained!")
            return None
        else:
            self.test_features = features_values
            self.test_classes = classes_values

            self.prediction = self.predict(features_values)
            self.score = self.clf.score(  # type: ignore
                features_values,
                classes_values,
            )
            self.accuracy = accuracy_score(  # type: ignore
                classes_values,
                self.prediction,
            )
            self.recall = recall_score(classes_values, self.prediction)
            self.precision = precision_score(classes_values, self.prediction)
            self.f1 = f1_score(classes_values, self.prediction)
            return self.get_results()

    def get_results(
        self,
    ) -> ResultsMetrics:
        return {
            "prediction": self.prediction,
            "score": self.score,
            "accuracy": self.accuracy,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
        }

    def predict(
        self,
        features_values: List[List[float]],
    ) -> List[int]:
        return self.clf.predict(features_values)  # type: ignore


class Pool:
    def __init__(self, classifiers: List[Classifier]) -> None:
        self.classifiers = classifiers

    def get_classifiers(self) -> List[Classifier]:
        return self.classifiers

    def fit_all(
        self,
        features_values: List[List[float]],
        classes_values: List[int],
    ) -> None:
        for classifier in self.classifiers:
            classifier.fit(features_values, classes_values)

    def test_all(
        self,
        features_values: List[List[float]],
        classes_values: List[int],
    ) -> None:
        for classifier in self.classifiers:
            classifier.test(features_values, classes_values)

    def get_results(self) -> List[ResultsMetrics]:
        list_results_classifiers = []
        for classifier in self.classifiers:
            result: ResultsMetrics = {}
            result = classifier.get_results()
            result["name"] = classifier.name
            list_results_classifiers.append(result)
        return list_results_classifiers

    def remove_classifier_one_class(self) -> List[Classifier]:
        self.classifiers = [
            classifier for classifier in self.classifiers if classifier.f1 > 0
        ]
        return self.classifiers

    def shuffle_classifiers(self) -> List[Classifier]:
        random.shuffle(self.classifiers)
        return self.classifiers

    def pop(self) -> Classifier:
        return self.classifiers.pop(0)


class Cell:
    def __init__(
        self,
        classifier: Classifier,
        init_energy: float,
        localization: Tuple[int, int],
        neighbors: List[Tuple[int, int]],
    ) -> None:
        self.classifier: Classifier = classifier
        self.energy: float = init_energy
        self.localization: Tuple[int, int] = localization
        self.neighbors_list: List[Tuple[int, int]] = neighbors

    def get_neighbors(self) -> List[Tuple[int, int]]:
        return self.neighbors_list


class Matrix:
    def __init__(
        self,
        size: int,
        pool: Pool,
        init_enery: float,
        distance_neighborhood: int,
    ) -> None:
        self.matrix: List[List[Cell]] = []
        self.size = size
        self._init_matrix(
            pool=pool,
            init_enery=init_enery,
            distance=distance_neighborhood,
            size=size,
        )

    def get(self) -> List[List[Cell]]:
        return self.matrix

    def get_size(self) -> int:
        return self.size

    def _init_matrix(
        self, pool: Pool, init_enery: float, distance: int, size: int
    ) -> None:
        for x in range(self.size):
            line: List[Cell] = []
            for y in range(self.size):
                neighbors_list = self._gen_neighborhood(
                    cell_local=(x, y), distance=distance, size=size
                )
                c = Cell(
                    classifier=pool.pop(),
                    init_energy=init_enery,
                    localization=(x, y),
                    neighbors=neighbors_list,
                )
                line.append(c)
            self.matrix.append(line)

    def _gen_neighborhood(
        self, cell_local: Tuple[int, int], distance: int, size: int
    ) -> List[Tuple[int, int]]:
        neighbors_list: List[Tuple[int, int]] = []
        for i in range(cell_local[0] - distance, cell_local[0] + distance + 1):
            if i < 0 or i >= size:
                continue
            for j in range(
                cell_local[1] - distance, cell_local[1] + distance + 1
            ):
                if j < 0 or j >= size:
                    continue
                elif i == cell_local[0] and j == cell_local[1]:
                    continue
                neighbors_list.append((i, j))
        return neighbors_list
