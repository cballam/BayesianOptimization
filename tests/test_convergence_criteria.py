from __future__ import annotations

import pytest
from datetime import datetime

from bayes_opt import BayesianOptimization

import numpy as np


@pytest.fixture
def target_func_trivial():
    # Max at 0, 1
    return lambda x, y: -(x**2) - ((y - 1) ** 2)


# Test each condition individually
def test_convergence_criteria(target_func_trivial):
    termination_criteria = {"iterations": 10}
    pbounds = {"x": [-10.0, 10.0], "y": [-10.0, 10.0]}
    opt = BayesianOptimization(
        f=target_func_trivial, pbounds=pbounds, termination_criteria=termination_criteria
    )

    # Ensure no initial points are specified.
    opt.maximize(init_points=0, n_iter=10)

    assert len(opt.res) == termination_criteria["iterations"]

    # Provide reasonable target value for objective fn
    termination_criteria = {"value": -0.05}
    opt = BayesianOptimization(
        f=target_func_trivial, pbounds=pbounds, termination_criteria=termination_criteria
    )

    # Call with large number of iterations, so that this is not the termination criteria
    opt.maximize(init_points=5, n_iter=1_000)

    assert opt.max["target"] > termination_criteria["value"]

    # 3 seconds of maximizing before termination
    termination_criteria = {"time": {"seconds": 3}}
    opt = BayesianOptimization(
        f=target_func_trivial, pbounds=pbounds, termination_criteria=termination_criteria
    )

    start = datetime.now()
    # Call with large number of iterations, so that this is not the termination criteria
    opt.maximize(n_iter=1_000, init_points=1)

    # Allow ~200ms tolerance on timing
    assert abs((datetime.now() - start).total_seconds() - termination_criteria["time"]["seconds"]) < 0.2

    # Terminate if no improvement in last 3 iterations
    termination_criteria = {"convergence_tol": {"n_iters": 3, "abs_tol": 0}}

    opt = BayesianOptimization(
        f=target_func_trivial, pbounds=pbounds, termination_criteria=termination_criteria
    )
    # Call with number of iterations which will not lead to termination criteria on iterations
    opt.maximize(n_iter=1_000, init_points=5)

    # Check that none of the last 3 values are the maximum
    no_improvement_in_3 = all([value < opt._space.max()["target"] for value in opt._space.target[-3:]])
    assert no_improvement_in_3

    # Converged if minimum improvement below 1 in last 10 iterations
    termination_criteria = {"convergence_tol": {"n_iters": 10, "abs_tol": 1}}

    opt = BayesianOptimization(
        f=target_func_trivial, pbounds=pbounds, termination_criteria=termination_criteria
    )
    opt.maximize(n_iter=1_000, init_points=5)

    improvement_below_tol = np.max(opt._space.target[-10:] - opt._space.max()["target"]) < 1

    assert improvement_below_tol
