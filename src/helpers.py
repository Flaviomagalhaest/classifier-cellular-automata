import copy
import math
from collections import Counter
from typing import Dict, List, Literal, Tuple

from pandas import DataFrame

PREDICTIONS = Literal[0, 1]
ResultsMetrics = Dict[str, str | List[int] | float]


def compare_two_predicts(l1, l2):
    return list(map(lambda p, q: p == q, l1, l2)).count(True)


def count_items_from_predict(
    prediction: List[int], predict: PREDICTIONS = 1
) -> int:
    c = Counter([r for r in prediction])
    return c[predict]


def dict_results_to_dataframe(results: List[ResultsMetrics]) -> DataFrame:
    def remove_predicionts(result: Dict) -> Dict:
        del result["prediction"]
        return result

    new_results = copy.deepcopy(results)
    results_without_predictions = list(map(remove_predicionts, new_results))
    df = DataFrame.from_dict(results_without_predictions)  # type: ignore
    return df.sort_values("f1", ascending=False)


def distance_two_points(
    cell_a: Tuple[int, int], cell_b: Tuple[int, int]
) -> float:
    distance_one = (cell_a[0] - cell_b[0]) ** 2
    distance_two = (cell_a[1] - cell_b[1]) ** 2
    return round(math.sqrt(distance_one + distance_two), 4)
