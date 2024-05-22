import numpy as np
import pytest

from object_tracking_cli.object_tracking.assignment import (
    greedy_assignment,
    hungarian_assignment,
)


@pytest.fixture(scope="module")
def nonglobal_problem():
    return np.array([[1, 20], [30, 40]])


@pytest.fixture(scope="module")
def global_problem():
    return np.array([[1, 2], [4, 3]])


def test_greedy_on_nonglobal_problem(nonglobal_problem):
    assignment = greedy_assignment(nonglobal_problem)
    assert assignment == {0: 0}


def test_greedy_on_global_problem(global_problem):
    assignment = greedy_assignment(global_problem)
    assert assignment == {0: 0, 1: 1}


def test_hungarian_on_nonglobal_problem(nonglobal_problem):
    assignment = hungarian_assignment(nonglobal_problem, th=100)
    assert assignment == {0: 0, 1: 1}


def test_hungarian_on_global_problem(global_problem):
    assignment = hungarian_assignment(global_problem, th=100)
    assert assignment == {0: 0, 1: 1}


def test_hungarian_on_missing_col():
    cost_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    assignment = hungarian_assignment(cost_matrix)
    assert assignment == {0: 0}


def test_hungarian_on_missing_row():
    cost_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    assignment = hungarian_assignment(cost_matrix)
    assert assignment == {0: 0}
