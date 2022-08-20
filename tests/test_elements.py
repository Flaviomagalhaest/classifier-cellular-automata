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

    matrix = Matrix(size=3, pool=pool, init_enery=100)

    assert matrix is not None
    assert len(matrix.matrix) == 3
    assert len(pool.classifiers) == 3
    assert pool.classifiers[0].name == "clf_9"
    assert pool.classifiers[1].name == "clf_10"
    assert pool.classifiers[2].name == "clf_11"


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

    matrix = Matrix(size=4, pool=pool, init_enery=100)

    assert matrix is not None
    assert len(matrix.matrix) == 4
    assert len(pool.classifiers) == 4
    assert pool.classifiers[0].name == "clf_16"
    assert pool.classifiers[1].name == "clf_17"
    assert pool.classifiers[2].name == "clf_18"
