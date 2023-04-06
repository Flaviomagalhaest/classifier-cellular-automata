import copy
import csv
import random
from typing import List, Tuple


def create_base(
    trainSamples: int,
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    X_test = []
    X_train = []
    with open("src/datasets/jm1.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
        jm1 = [row for nr, row in enumerate(spamreader)]
        jm1_true = [j for j in jm1 if j[21] == "true"]
        jm1_false = [j for j in jm1 if j[21] == "false"]
        random.shuffle(jm1_true)
        random.shuffle(jm1_false)
        trainPart = int(
            trainSamples / 2
        )  # Test sample divided between true answers and false answers
        jm1_train = jm1_true[:trainPart]
        jm1_train = jm1_train + jm1_false[:trainPart]
        # jm1_test = jm1_true[trainPart:trainSamples]
        # jm1_test = jm1_test + jm1_false[trainPart:trainSamples]
        jm1_test = jm1_true[trainPart:]
        jm1_test = jm1_test + jm1_false[trainPart:]
        random.shuffle(jm1_train)
        random.shuffle(jm1_test)
        jm1_test = jm1_test[:trainSamples]

    Y_train_str = [j.pop(-1) for j in jm1_train]
    Y_train = [1 if x == "true" else 0 for x in Y_train_str]

    for jt in jm1_train:
        X_train.append([float(j) for j in jt])

    Y_test_str = [j.pop(-1) for j in jm1_test]
    Y_test = [1 if x == "true" else 0 for x in Y_test_str]
    for jt in jm1_test:
        X_test.append([float(j) for j in jt])

    return X_test, X_train, Y_train, Y_test


def save_classifier_if_diverse(
    classif_score: float,
    classif_predict: List[int],
    predicts: List[List[int]],
    params: List[str],
    nr_items: int,
    string_parameters: str,
) -> None:
    if classif_score >= 0.6:
        if len(predicts) > 0:
            flag = False
            for pred in predicts:
                s = sum(
                    x != y
                    for x, y in zip(
                        classif_predict,
                        pred,
                    )
                )
                if s < (0.05 * nr_items):
                    flag = True
                    break

            if not flag:
                predicts.append(copy.deepcopy(classif_predict))
                params.append(string_parameters)
        else:
            predicts.append(copy.deepcopy(classif_predict))
            params.append(string_parameters)


def save_quality_classifier(
    f1_score: float,
    params: List[str],
    string_parameters: str,
):
    if f1_score > 0.3:
        params.append(string_parameters)
