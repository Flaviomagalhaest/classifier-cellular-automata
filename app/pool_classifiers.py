import csv
import re
from typing import List

from src.classifiers import Classifier


def _get_neighbors(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:

    from sklearn.neighbors import KNeighborsClassifier

    def _k_neighbors_classifier() -> None:
        with open("prep/data/knn.csv", newline="") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
            list_csv_classifier = [row for nr, row in enumerate(spamreader)]
            for knn in list_csv_classifier:
                n_neighbors = int(re.findall(r"\d+", knn[0])[0])
                weights = knn[1]
                algorithm = knn[2]
                p = int(re.findall(r"\d+", knn[3])[0])

                classifier = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    p=p,
                )

                list_classifiers.append(
                    Classifier(
                        "KNeighborsClassifier",
                        classifier,
                    )
                )

    _k_neighbors_classifier()
    return list_classifiers


def _get_discriminant_analysis(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )

    def _linear_discriminant_aanalysis() -> None:
        classifier = LinearDiscriminantAnalysis()

        list_classifiers.append(
            Classifier(
                "LinearDiscriminantAnalysis",
                classifier,
            )
        )

    def _quadratic_discriminant_aanalysis() -> None:
        classifier = QuadraticDiscriminantAnalysis()

        list_classifiers.append(
            Classifier(
                "QuadraticDiscriminantAnalysis",
                classifier,
            )
        )

    _linear_discriminant_aanalysis()
    _quadratic_discriminant_aanalysis()
    return list_classifiers


def _gaussian_process(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF

    def _gaussian_process_classifier() -> None:
        classifier = GaussianProcessClassifier(1.0 * RBF(1.0))

        list_classifiers.append(
            Classifier(
                "GaussianProcessClassifier",
                classifier,
            )
        )

    _gaussian_process_classifier()
    return list_classifiers


def _linear_model(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.linear_model import (
        LogisticRegression,
        PassiveAggressiveClassifier,
        RidgeClassifier,
        SGDClassifier,
    )

    def _logistic_regression() -> None:
        list_logistic_regression = [
            LogisticRegression(random_state=0),
            LogisticRegression(C=0.5),
            LogisticRegression(C=0.1),
            LogisticRegression(C=0.05),
            LogisticRegression(solver="newton-cg", random_state=0),
            LogisticRegression(solver="newton-cg", C=0.5),
            LogisticRegression(solver="newton-cg", C=0.1),
            LogisticRegression(solver="newton-cg", C=0.05),
            LogisticRegression(
                penalty="none",
                solver="newton-cg",
                random_state=0,
            ),
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                random_state=0,
            ),
            LogisticRegression(penalty="l2", solver="liblinear", C=0.5),
            LogisticRegression(penalty="l2", solver="liblinear", C=0.1),
            LogisticRegression(penalty="l2", solver="liblinear", C=0.05),
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                random_state=0,
            ),
            LogisticRegression(penalty="l1", solver="liblinear", C=0.5),
            LogisticRegression(penalty="l1", solver="liblinear", C=0.1),
            LogisticRegression(penalty="l1", solver="liblinear", C=0.05),
        ]

        for logistic_regression in list_logistic_regression:
            list_classifiers.append(
                Classifier(
                    "LogisticRegression",
                    logistic_regression,
                )
            )

    def _sgd_classifier() -> None:
        list_sgd_classifier = [
            SGDClassifier(loss="hinge", penalty="l2"),
            SGDClassifier(loss="log"),
            SGDClassifier(loss="modified_huber"),
            SGDClassifier(loss="squared_hinge"),
            SGDClassifier(loss="perceptron"),
            SGDClassifier(loss="huber"),
            SGDClassifier(loss="epsilon_insensitive"),
            SGDClassifier(loss="squared_loss"),
        ]

        for sgd_classifier in list_sgd_classifier:
            list_classifiers.append(
                Classifier(
                    "SGDClassifier",
                    sgd_classifier,
                )
            )

    def _ridge_classifier() -> None:
        list_ridge_classifier = [
            RidgeClassifier(solver="svd"),
            RidgeClassifier(alpha=2.5, solver="svd"),
            RidgeClassifier(alpha=5, solver="svd"),
            RidgeClassifier(alpha=0.5, solver="svd"),
            RidgeClassifier(fit_intercept=False, solver="svd"),
            RidgeClassifier(alpha=2.5, fit_intercept=False, solver="svd"),
            RidgeClassifier(alpha=5, fit_intercept=False, solver="svd"),
            RidgeClassifier(alpha=0.5, fit_intercept=False, solver="svd"),
            RidgeClassifier(solver="sparse_cg"),
            RidgeClassifier(solver="lsqr"),
            RidgeClassifier(solver="sag"),
        ]

        for ridge_classifier in list_ridge_classifier:
            list_classifiers.append(
                Classifier(
                    "RidgeClassifier",
                    ridge_classifier,
                )
            )

    def _passive_aggressive_classifier() -> None:
        list_passive_aggressive_classifier = [
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
            PassiveAggressiveClassifier(),
        ]

        for passive_aggressive_classif in list_passive_aggressive_classifier:
            list_classifiers.append(
                Classifier(
                    "PassiveAggressiveClassifier",
                    passive_aggressive_classif,
                )
            )

    _logistic_regression()
    _sgd_classifier()
    _ridge_classifier()
    _passive_aggressive_classifier()
    return list_classifiers


def get_all_classifiers() -> List[Classifier]:
    list_classifiers: List[Classifier] = []

    _get_neighbors(list_classifiers)
    _get_discriminant_analysis(list_classifiers)
    _gaussian_process(list_classifiers)
    _linear_model(list_classifiers)
    return list_classifiers
