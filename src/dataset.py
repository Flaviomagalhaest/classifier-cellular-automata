import csv
import random
from pathlib import Path
from typing import List, Literal, Tuple


class Dataset:
    DATASET = Literal[
        "cm1",
        "jm1",
        "kc1",
        "pc1",
    ]

    def __init__(
        self,
        name: DATASET,
        train_clf_amount: int,
        train_cca_amount: int,
        test_amount: int,
    ) -> None:
        first_gap = train_clf_amount
        second_gap = first_gap + train_cca_amount
        third_gap = second_gap + test_amount
        filepath = Path("src/datasets/" + name + ".csv")
        with open(filepath, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            samples = [row for nr, row in enumerate(spamreader)]
            random.shuffle(samples)
            train_clf_samples = samples[:first_gap]
            train_cca_samples = samples[first_gap:second_gap]
            test_samples = samples[second_gap:third_gap]

            self.train_clf_features, self.train_clf_result = self._data_prep(
                train_clf_samples,
            )
            self.train_cca_features, self.train_cca_result = self._data_prep(
                train_cca_samples,
            )
            self.test_features, self.test_result = self._data_prep(
                test_samples,
            )

    def get_train_clf_samples(self) -> Tuple[List[List[float]], List[int]]:
        return self.train_clf_features, self.train_clf_result

    def get_train_cca_samples(self) -> Tuple[List[List[float]], List[int]]:
        return self.train_cca_features, self.train_cca_result

    def get_test_samples(self) -> Tuple[List[List[float]], List[int]]:
        return self.test_features, self.test_result

    def _data_prep(
        self, samples: List[List[str]]
    ) -> Tuple[List[List[float]], List[int]]:
        result: List[int]
        features: List[List[float]]

        result = list(map(lambda x: 1 if x.pop(-1) == "true" else 0, samples))
        features = list(map(lambda y: [float(s) for s in y], samples))

        return features, result
