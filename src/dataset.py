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
        train_amount: int,
        test_amount: int,
    ) -> None:
        filepath = Path("src/datasets/" + name + ".csv")
        with open(filepath, newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            samples = [row for nr, row in enumerate(spamreader)]
            random.shuffle(samples)
            train_samples = samples[:train_amount]
            test_samples = samples[train_amount : train_amount + test_amount]

            self.train_features, self.train_result = self._data_prep(
                train_samples,
            )
            self.test_features, self.test_result = self._data_prep(
                test_samples,
            )

    def get_train_samples(self) -> Tuple[List[List[float]], List[int]]:
        return self.train_features, self.train_result

    def get_test_samples(self) -> Tuple[List[List[float]], List[int]]:
        return self.test_features, self.test_result

    def _data_prep(
        self, samples: List[str]
    ) -> Tuple[List[List[float]], List[int]]:
        result: List[int]
        features: List[List[float]]

        result = list(map(lambda x: 1 if x.pop(-1) == "true" else 0, samples))
        features = list(map(lambda y: [float(s) for s in y], samples))

        return features, result
