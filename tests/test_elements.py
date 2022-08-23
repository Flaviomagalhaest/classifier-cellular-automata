import copy

from src.elements import Classifier, Matrix, Pool


def test_creating_matrix_3x3_object_getting_classifier_from_pool():
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        algorithm="ball_tree",
        p=1,
    )
    list = []
    for i in range(12):
        list.append(Classifier("clf_" + str(i), classifier))
    pool = Pool(list)

    matrix = Matrix(size=3, pool=pool, init_enery=100, distance_neighborhood=1)

    assert matrix is not None
    assert len(matrix.get()) == 3
    assert len(pool.get_classifiers()) == 3
    assert pool.get_classifiers()[0].name == "clf_9"
    assert pool.get_classifiers()[1].name == "clf_10"
    assert pool.get_classifiers()[2].name == "clf_11"


def test_creating_matrix_4x4_object_getting_classifier_from_pool():
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        algorithm="ball_tree",
        p=1,
    )
    list = []
    for i in range(20):
        list.append(Classifier("clf_" + str(i), classifier))
    pool = Pool(list)

    matrix = Matrix(size=4, pool=pool, init_enery=100, distance_neighborhood=1)

    assert matrix is not None
    assert len(matrix.get()) == 4
    assert len(pool.get_classifiers()) == 4
    assert pool.get_classifiers()[0].name == "clf_16"
    assert pool.get_classifiers()[1].name == "clf_17"
    assert pool.get_classifiers()[2].name == "clf_18"


def test_matrix_gen_neighborhood_correctly():
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        algorithm="ball_tree",
        p=1,
    )
    list = []
    for i in range(65):
        list.append(Classifier("clf_" + str(i), classifier))
    pool_orig = Pool(list)

    def generate_answer_neighbors(x_cell, y_cell, list_possibilities):
        resp_tuple = []
        for i in list_possibilities:
            for j in list_possibilities:
                if not (x_cell == i and y_cell == j):
                    resp_tuple.append((i, j))
        return resp_tuple

    pool = copy.deepcopy(pool_orig)
    m1 = Matrix(size=8, pool=pool, init_enery=100, distance_neighborhood=1)

    assert m1.get()[0][0].get_neighbors() == generate_answer_neighbors(
        0, 0, [0, 1]
    )
    assert m1.get()[1][1].get_neighbors() == generate_answer_neighbors(
        1, 1, [0, 1, 2]
    )
    assert m1.get()[2][2].get_neighbors() == generate_answer_neighbors(
        2, 2, [1, 2, 3]
    )
    assert m1.get()[3][3].get_neighbors() == generate_answer_neighbors(
        3, 3, [2, 3, 4]
    )
    assert m1.get()[4][4].get_neighbors() == generate_answer_neighbors(
        4, 4, [3, 4, 5]
    )
    assert m1.get()[5][5].get_neighbors() == generate_answer_neighbors(
        5, 5, [4, 5, 6]
    )
    assert m1.get()[6][6].get_neighbors() == generate_answer_neighbors(
        6, 6, [5, 6, 7]
    )
    assert m1.get()[7][7].get_neighbors() == generate_answer_neighbors(
        7, 7, [6, 7]
    )

    pool = copy.deepcopy(pool_orig)
    m2 = Matrix(size=8, pool=pool, init_enery=100, distance_neighborhood=2)
    assert m2.get()[0][0].get_neighbors() == generate_answer_neighbors(
        0, 0, [0, 1, 2]
    )
    assert m2.get()[1][1].get_neighbors() == generate_answer_neighbors(
        1, 1, [0, 1, 2, 3]
    )
    assert m2.get()[2][2].get_neighbors() == generate_answer_neighbors(
        2, 2, [0, 1, 2, 3, 4]
    )
    assert m2.get()[3][3].get_neighbors() == generate_answer_neighbors(
        3, 3, [1, 2, 3, 4, 5]
    )
    assert m2.get()[4][4].get_neighbors() == generate_answer_neighbors(
        4, 4, [2, 3, 4, 5, 6]
    )
    assert m2.get()[5][5].get_neighbors() == generate_answer_neighbors(
        5, 5, [3, 4, 5, 6, 7]
    )
    assert m2.get()[6][6].get_neighbors() == generate_answer_neighbors(
        6, 6, [4, 5, 6, 7]
    )
    assert m2.get()[7][7].get_neighbors() == generate_answer_neighbors(
        7, 7, [5, 6, 7]
    )

    pool = copy.deepcopy(pool_orig)
    m3 = Matrix(size=8, pool=pool, init_enery=100, distance_neighborhood=3)
    assert m3.get()[0][0].get_neighbors() == generate_answer_neighbors(
        0, 0, [0, 1, 2, 3]
    )
    assert m3.get()[1][1].get_neighbors() == generate_answer_neighbors(
        1, 1, [0, 1, 2, 3, 4]
    )
    assert m3.get()[2][2].get_neighbors() == generate_answer_neighbors(
        2, 2, [0, 1, 2, 3, 4, 5]
    )
    assert m3.get()[3][3].get_neighbors() == generate_answer_neighbors(
        3, 3, [0, 1, 2, 3, 4, 5, 6]
    )
    assert m3.get()[4][4].get_neighbors() == generate_answer_neighbors(
        4, 4, [1, 2, 3, 4, 5, 6, 7]
    )
    assert m3.get()[5][5].get_neighbors() == generate_answer_neighbors(
        5, 5, [2, 3, 4, 5, 6, 7]
    )
    assert m3.get()[6][6].get_neighbors() == generate_answer_neighbors(
        6, 6, [3, 4, 5, 6, 7]
    )
    assert m3.get()[7][7].get_neighbors() == generate_answer_neighbors(
        7, 7, [4, 5, 6, 7]
    )


# def test():
#     from sklearn.linear_model import LogisticRegression
#     from src.dataset import Dataset

#     dataset = Dataset("jm1", 1000, 1000)
#     x_train, y_train = dataset.get_train_samples()
#     x_test, y_test = dataset.get_test_samples()
#     lr = LogisticRegression(C=0.5)
#     c = Classifier("RegressaoLogistica", lr)
#     c.fit(x_train, y_train)
#     c.test(x_test, y_test)

#     from sklearn.metrics import roc_auc_score

#     # prob = c.clf.predict_proba(x_test)[:, 1]  # type: ignore
#     prob = c.clf.decision_function(x_test)  # type: ignore
#     roc = roc_auc_score(y_test, prob)
#     ...
