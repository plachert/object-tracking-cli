from typing import Callable, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

Assignment = Dict[int, int]  # new bbox idx to registered bbox idx
AssignmentFunction = Callable[[np.ndarray], Assignment]

AVAILABLE_ASSIGNMENT_FUNCS = {}


def register_func(cls):
    AVAILABLE_ASSIGNMENT_FUNCS[cls.__name__] = cls
    return cls


@register_func
def greedy_assignnment(cost_matrix) -> Assignment:
    # https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ approach
    rows = cost_matrix.min(axis=1).argsort()
    cols = cost_matrix.argmin(axis=1)[rows]
    used_rows = set()
    used_cols = set()
    assignment = {}
    for row, col in zip(rows, cols):
        if row in used_rows or col in used_cols:
            continue
        assignment[col] = row
        used_rows.add(row)
        used_cols.add(col)
    return assignment


@register_func
def hungarian_assignment(cost_matrix, th: float = 0.0) -> Assignment:
    def make_matrix_square(matrix, pad_value=0):
        rows, cols = matrix.shape
        max_dim = max(rows, cols)
        pad_rows = max_dim - rows
        pad_cols = max_dim - cols
        if pad_rows > 0:
            padding = np.full((pad_rows, cols), pad_value)
            matrix = np.vstack((matrix, padding))

        if pad_cols > 0:
            padding = np.full((rows, pad_cols), pad_value)
            matrix = np.hstack((matrix, padding))
        return matrix

    cost_matrix_square = make_matrix_square(cost_matrix, pad_value=th)
    row_indices, col_indices = linear_sum_assignment(cost_matrix_square)
    org_rows, org_cols = cost_matrix.shape
    assignment = {}
    for row, col in zip(row_indices, col_indices):
        if row < org_rows and col < org_cols:
            assignment[col] = row
    return assignment
