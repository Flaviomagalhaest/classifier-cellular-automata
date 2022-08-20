import yaml

from app.pool_classifiers import get_all_classifiers
from src.dataset import Dataset
from src.elements import Matrix, Pool
from src.helpers import dict_results_to_dataframe

with open("src/cca_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

all_classifiers = get_all_classifiers()
dataset = Dataset("jm1", 1000, 1000)
x_train, y_train = dataset.get_train_samples()
x_test, y_test = dataset.get_test_samples()
pool = Pool(all_classifiers)
pool.fit_all(x_train, y_train)
pool.test_all(x_test, y_test)
pool.remove_classifier_one_class()
results = pool.get_results()

results_df = dict_results_to_dataframe(results)

pool.shuffle_classifiers()
matrix = Matrix(
    size=config["matrix_size"],
    pool=pool,
    init_enery=config["init_energy"],
)
...
