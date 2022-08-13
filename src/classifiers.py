from typing import List

from sklearn.base import BaseEstimator


class Classifiers:
    def __init__(
        self,
        name: str,
        classif: BaseEstimator,
    ) -> None:

        self.clf = classif
        self.name = name
        self.params = classif.get_params()
        self.prediction: List[int] = []

    def fit(
        self,
        features_values: List[float],
        class_values: List[int],
    ) -> None:
        self.clf.fit(features_values, class_values)
        print("Clasificador " + self.name + " treinado.")

    def predict(
        self,
        features_values: List[float],
    ) -> List[int]:
        self.prediction = self.clf.predict(features_values)
        return self.prediction
