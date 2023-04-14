import csv
import re
from typing import List

from sklearn.model_selection import ParameterGrid

from src.elements import Classifier


def _get_svm(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.svm import LinearSVC

    def _get_svc() -> None:
        from sklearn.svm import SVC

        svm_params = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.1, 1],
            "shrinking": [True, False],
            "class_weight": [None, "balanced"],
        }
        svm_combinations = list(ParameterGrid(svm_params))
        n_classifiers = min(len(svm_combinations), 100)
        for params in svm_combinations[:n_classifiers]:
            clf = SVC(**params)
            list_classifiers.append(
                Classifier(
                    "SVC",
                    clf,
                )
            )

    def _get_linear_svc() -> None:
        svc_params = {
            "C": [0.1, 1.0, 10.0],
            "loss": ["hinge", "squared_hinge"],
            "max_iter": [1000, 5000],
        }

        svc_combinations = list(ParameterGrid(svc_params))

        n_classifiers = min(len(svc_combinations), 100)

        for params in svc_combinations[:n_classifiers]:
            clf = LinearSVC(**params)
            list_classifiers.append(
                Classifier(
                    "LinearSVC",
                    clf,
                )
            )

    # _get_svc()
    _get_linear_svc()
    return list_classifiers


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
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)

    def _linear_discriminant_aanalysis() -> None:

        lda_params = {
            "solver": ["svd", "lsqr", "eigen"],
            # "shrinkage": [None, "auto", 0.1],
            "tol": [1e-4, 1e-3, 1e-2],
        }

        lda_combinations = list(ParameterGrid(lda_params))

        n_classifiers = min(len(lda_combinations), 100)

        for params in lda_combinations[:n_classifiers]:
            clf = LinearDiscriminantAnalysis(**params)
            list_classifiers.append(
                Classifier(
                    "LinearDiscriminantAnalysis",
                    clf,
                )
            )

    def _quadratic_discriminant_aanalysis() -> None:
        qda_params = {
            "priors": [None, [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]],
            "reg_param": [0, 0.1, 0.2, 0.5],
            "store_covariance": [True, False],
        }

        qda_combinations = list(ParameterGrid(qda_params))

        n_classifiers = min(len(qda_combinations), 100)

        for params in qda_combinations[:n_classifiers]:
            clf = QuadraticDiscriminantAnalysis(**params)
            list_classifiers.append(
                Classifier(
                    "QuadraticDiscriminantAnalysis",
                    clf,
                )
            )

    _linear_discriminant_aanalysis()
    _quadratic_discriminant_aanalysis()
    return list_classifiers


def _get_gaussian_process(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

    def _gaussian_process_classifier() -> None:
        gpc_params = {
            "kernel": [
                1.0 * RBF(length_scale=1.0),
                ConstantKernel(1.0) * Matern(length_scale=1.0),
                Matern(length_scale=1.0),
            ],
            "optimizer": [None, "fmin_l_bfgs_b"],
            "n_restarts_optimizer": [0, 1, 2, 3],
            "multi_class": ["one_vs_rest", "one_vs_one"],
        }

        gpc_combinations = list(ParameterGrid(gpc_params))

        n_classifiers = min(len(gpc_combinations), 50)

        for params in gpc_combinations[:n_classifiers]:
            clf = GaussianProcessClassifier(**params)
            list_classifiers.append(
                Classifier(
                    "GaussianProcessClassifier",
                    clf,
                )
            )

    # _gaussian_process_classifier()
    return list_classifiers


def _get_linear_model(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.linear_model import (LogisticRegression,
                                      PassiveAggressiveClassifier,
                                      RidgeClassifier, SGDClassifier)

    def _logistic_regression() -> None:
        logreg_params = {
            "penalty": ["l1", "l2"],
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "saga"],
            "max_iter": [100, 200],
            "class_weight": [None, "balanced"],
        }

        logreg_combinations = list(ParameterGrid(logreg_params))

        n_classifiers = min(len(logreg_combinations), 100)

        for params in logreg_combinations[:n_classifiers]:
            clf = LogisticRegression(**params)
            list_classifiers.append(
                Classifier(
                    "LogisticRegression",
                    clf,
                )
            )

    def _sgd_classifier() -> None:
        sgd_params = {
            "loss": ["hinge", "log", "modified_huber"],
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
            "max_iter": [100, 200, 500],
            "class_weight": [None, "balanced"],
        }

        sgd_combinations = list(ParameterGrid(sgd_params))

        n_classifiers = min(len(sgd_combinations), 100)

        for params in sgd_combinations[:n_classifiers]:
            clf = SGDClassifier(**params)
            list_classifiers.append(
                Classifier(
                    "SGDClassifier",
                    clf,
                )
            )

    def _ridge_classifier() -> None:
        ridge_params = {
            "alpha": [0.01, 0.1, 1.0],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "solver": [
                "auto",
                "svd",
                "cholesky",
                "lsqr",
                "sparse_cg",
                "sag",
                "saga",
            ],
        }

        ridge_combinations = list(ParameterGrid(ridge_params))

        n_classifiers = min(len(ridge_combinations), 100)

        for params in ridge_combinations[:n_classifiers]:
            clf = RidgeClassifier(**params)
            list_classifiers.append(
                Classifier(
                    "RidgeClassifier",
                    clf,
                )
            )

    def _passive_aggressive_classifier() -> None:
        pa_params = {
            "C": [0.1, 1.0, 10.0],
            "fit_intercept": [True, False],
            "loss": ["hinge", "squared_hinge"],
            "max_iter": [1000, 5000],
        }

        pa_combinations = list(ParameterGrid(pa_params))

        n_classifiers = min(len(pa_combinations), 100)

        for params in pa_combinations[:n_classifiers]:
            clf = PassiveAggressiveClassifier(**params)
            list_classifiers.append(
                Classifier(
                    "PassiveAggressiveClassifier",
                    clf,
                )
            )
        # list_passive_aggressive_classifier = [
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        #     PassiveAggressiveClassifier(),
        # ]

        # for passive_aggressive_classif in list_passive_aggressive_classifier:
        #     list_classifiers.append(
        #         Classifier(
        #             "PassiveAggressiveClassifier",
        #             passive_aggressive_classif,
        #         )
        #     )

    _logistic_regression()
    _sgd_classifier()
    _ridge_classifier()
    _passive_aggressive_classifier()
    return list_classifiers


def _get_naive_bayes(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.naive_bayes import GaussianNB

    def _gaussian_nb() -> None:
        gnb_params = {
            "priors": [None, [0.1, 0.9], [0.25, 0.75], [0.4, 0.6], [0.5, 0.5]],
        }

        gnb_combinations = list(ParameterGrid(gnb_params))

        n_classifiers = min(len(gnb_combinations), 100)

        for params in gnb_combinations[:n_classifiers]:
            clf = GaussianNB(**params)
            list_classifiers.append(
                Classifier(
                    "GaussianNB",
                    clf,
                )
            )

    _gaussian_nb()
    return list_classifiers


def _get_neural_network(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.neural_network import MLPClassifier

    def _mlpc() -> None:
        mlp_params = {
            "hidden_layer_sizes": [
                (10,),
                (50,),
                (100,),
                (10, 10),
                (50, 50),
                (100, 100),
            ],
            "activation": ["relu", "logistic", "tanh"],
            "solver": ["sgd", "adam"],
            "alpha": [0.0001, 0.001, 0.01],
        }

        mlp_combinations = list(ParameterGrid(mlp_params))

        n_classifiers = min(len(mlp_combinations), 50)

        for params in mlp_combinations[:n_classifiers]:
            clf = MLPClassifier(**params)
            list_classifiers.append(
                Classifier(
                    "MLPClassifier",
                    clf,
                )
            )

    _mlpc()
    return list_classifiers


def _get_tree(
    list_classifiers: List[Classifier] = [],
) -> List[Classifier]:
    from sklearn.tree import DecisionTreeClassifier

    def _decision_tree() -> None:
        params = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [2, 4, 6, 8],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 3, 4, 5],
        }

        tree_combinations = list(ParameterGrid(params))

        n_classifiers = min(len(tree_combinations), 100)

        for params in tree_combinations[:n_classifiers]:
            clf = DecisionTreeClassifier(**params)
            list_classifiers.append(
                Classifier(
                    "DecisionTreeClassifier",
                    clf,
                )
            )

    _decision_tree()
    return list_classifiers


def get_all_classifiers() -> List[Classifier]:
    list_classifiers: List[Classifier] = []

    _get_neighbors(list_classifiers)
    _get_discriminant_analysis(list_classifiers)
    _get_gaussian_process(list_classifiers)
    _get_linear_model(list_classifiers)
    _get_naive_bayes(list_classifiers)
    _get_neural_network(list_classifiers)
    _get_tree(list_classifiers)
    _get_svm(list_classifiers)
    return list_classifiers
