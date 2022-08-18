import random
from typing import Dict, List

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class Classifiers:
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
        features_values: List[float],
        classes_values: List[int],
    ) -> None:
        self.train_features = features_values
        self.train_classes = classes_values
        self.clf.fit(features_values, classes_values)
        self.classifier_trained = True
        print("Clasificador " + self.name + " treinado.")

    def test(
        self,
        features_values: List[float],
        classes_values: List[int],
    ) -> Dict[str, float] | None:
        if not self.classifier_trained:
            print("This classifier is not trained!")
            return None
        else:
            self.test_features = features_values
            self.test_classes = classes_values

            self.prediction = self.predict(features_values)
            self.score = self.clf.score(features_values, classes_values)
            self.accuracy = accuracy_score(classes_values, self.prediction)
            self.recall = recall_score(classes_values, self.prediction)
            self.precision = precision_score(classes_values, self.prediction)
            self.f1 = f1_score(classes_values, self.prediction)
            return self.get_results()

    def get_results(self) -> Dict[str, float]:
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
        features_values: List[float],
    ) -> List[int]:
        return self.clf.predict(features_values)


class Pool:
    def __init__(self, classifiers: List[Classifiers]) -> None:
        self.classifiers = classifiers

    def fit_all(
        self,
        features_values: List[float],
        classes_values: List[int],
    ) -> None:
        for classifier in self.classifiers:
            classifier.fit(features_values, classes_values)

    def test_all(
        self,
        features_values: List[float],
        classes_values: List[int],
    ) -> None:
        for classifier in self.classifiers:
            classifier.test(features_values, classes_values)

    def get_results(self) -> List[Dict[str, float]]:
        list_results_classifiers = []
        for classifier in self.classifiers:
            result: Dict[str, float] = {}
            result = classifier.get_results()
            result["name"] = classifier.name
            list_results_classifiers.append(result)
        return list_results_classifiers

    def remove_classifier_one_class(self) -> List[Classifiers]:
        self.classifiers = [
            classifier for classifier in self.classifiers if classifier.f1 > 0
        ]
        return self.classifiers

    def shuffle_classifiers(self) -> List[Classifiers]:
        random.shuffle(self.classifiers)
        return self.classifiers
