from src.dataset import Dataset


def test_initialize_dataset():
    dataset = Dataset("jm1", 1000, 1000, 1000)
    x_train_clf, y_train_clf = dataset.get_train_clf_samples()
    x_train_cca, y_train_cca = dataset.get_train_cca_samples()
    x_test, y_test = dataset.get_test_samples()

    assert len(x_train_clf) == 1000
    assert len(x_train_cca) == 1000
    assert len(x_test) == 1000
    ...
