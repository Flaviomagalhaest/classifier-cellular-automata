import yaml
from sklearn.metrics import classification_report

from app.pool_classifiers import get_all_classifiers
from src.cca import learning_algorithm
from src.dataset import Dataset
from src.elements import Matrix, Pool
from src.helpers import dict_results_to_dataframe

with open("src/cca_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

all_classifiers = get_all_classifiers()
dataset = Dataset(
    name="pc1",
    train_clf_amount=300,
    train_cca_amount=300,
    test_amount=400,
)
x_train_clf, y_train_clf = dataset.get_train_clf_samples()
x_train_cca, y_train_cca = dataset.get_train_cca_samples()
x_test, y_test = dataset.get_test_samples()
pool = Pool(all_classifiers)
pool.fit_all(x_train_clf, y_train_clf)
pool.test_all(x_test, y_test)
pool.remove_classifier_one_class()
results = pool.get_results()

results_df = dict_results_to_dataframe(results)

pool.shuffle_classifiers()
matrix = Matrix(
    size=config["matrix_size"],
    pool=pool,
    init_enery=config["init_energy"],
    distance_neighborhood=config["distance"],
)

learning_algorithm(
    matrix=matrix,
    pool=pool,
    sample_features=x_train_cca,
    sample_class=y_train_cca,
    distance=config["distance"],
    interactions=config["interactions"],
    init_energy=config["init_energy"],
)
matrix_class = matrix.predict_matrix(x_test)
results_df_matrix = matrix.get_results()
print(classification_report(y_test, matrix_class, digits=3))
...
