from enum import Enum
from typing import Callable, Dict, List, Tuple

from src.elements import Matrix, Pool


class Module(Enum):
    with_defect = 1
    no_defect = 0


def _transaction_rule(
    score_equal_classified: float, score_different_classified: float
):
    total = score_equal_classified + score_different_classified
    pct = (score_equal_classified / total) * 100
    return 100 - pct


def _count_scores_neighbors(
    matrix: Matrix, neighbors: List[Dict[str, Tuple[int, int] | float]]
) -> Tuple[float, float]:
    score_votes_defect = 0.0
    score_votes_no_defect = 0.0
    for neighbor in neighbors:
        line = neighbor["local"][0]  # type: ignore
        column = neighbor["local"][1]  # type: ignore
        cell_neighbor = matrix.get()[line][column]
        if cell_neighbor.get_predicion() == Module.with_defect.value:
            score_votes_defect += 1 / neighbor["distance"]  # type: ignore
        elif cell_neighbor.get_predicion() == Module.no_defect.value:
            score_votes_no_defect += 1 / neighbor["distance"]  # type: ignore
    return score_votes_defect, score_votes_no_defect


def _calc_energy(
    cell_predict: int | None,
    score_defect: float,
    score_no_defect: float,
    class_correct: int,
) -> float:
    energy = 0.0
    # find energy gain
    if cell_predict == Module.with_defect.value:
        energy = _transaction_rule(score_defect, score_no_defect)
    if cell_predict == Module.no_defect.value:
        energy = _transaction_rule(score_no_defect, score_defect)

    # if cell predict is wrong, the gain of energy is negative
    if cell_predict != class_correct:
        energy = -energy
    return energy


def learning_algorithm(
    matrix: Matrix,
    pool: Pool,
    sample_features: List[List[float]],
    sample_class: List[int],
    distance: int,
    interactions: int = 100,
    init_energy: int = 1000,
    transaction_rule: Callable = _transaction_rule,
) -> Matrix:
    for interaction in range(interactions):
        print("Interacao: " + str(interaction))
        nr_dead_cells = 0
        # iterate index of samples
        for index in range(len(sample_features)):
            # iterate all cells in matrix
            matrix.predict_all_cells(sample_features[index])
            for line_matrix in matrix.get():
                for cell in line_matrix:
                    # find neighborhoods scores
                    score_defect, score_no_defect = _count_scores_neighbors(
                        matrix=matrix, neighbors=cell.get_neighbors()
                    )

                    # calc energy
                    energy = _calc_energy(
                        cell_predict=cell.get_predicion(),
                        score_defect=score_defect,
                        score_no_defect=score_no_defect,
                        class_correct=sample_class[index],
                    )

                    # update energy
                    cell.add_energy(sample_class[index], energy)

                    # replace dead cell
                    cells_energy = cell.get_energy()
                    if cells_energy[0] <= 0 or cells_energy[1] <= 0:
                        nr_dead_cells += 1
                        cell.reset_classifier(
                            pool=pool, init_energy=init_energy
                        )
        print("Total de mortes: ", nr_dead_cells)
    return matrix
