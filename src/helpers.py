import copy
from collections import Counter
from typing import Dict, List, Literal

from pandas import DataFrame

PREDICTIONS = Literal[0, 1]


def compare_two_predicts(l1, l2):
    return list(map(lambda p, q: p == q, l1, l2)).count(True)


def count_items_from_predict(
    prediction: List[int], predict: PREDICTIONS = 1
) -> int:
    c = Counter([r for r in prediction["prediction"]])
    return c[predict]


def dict_results_to_dataframe(results: List[Dict[str, float]]) -> DataFrame:
    def remove_predicionts(result: Dict) -> Dict:
        del result["prediction"]
        return result

    new_results = copy.deepcopy(results)
    results_without_predictions = list(map(remove_predicionts, new_results))
    df = DataFrame.from_dict(results_without_predictions)
    return df.sort_values("f1", ascending=False)
