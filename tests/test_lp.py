import cvxpy as cp
from algorithms.linear_programming import simplex

def generate_random_lp(n_vars=3, n_constraints=5, seed=None):
    rng = np.random.default_rng(seed)

    A = rng.uniform(-10, 10, size=(n_constraints, n_vars))
    x_true = rng.uniform(0, 10, size=n_vars)  # feasible solution
    b = A @ x_true + rng.uniform(1, 5, size=n_constraints)  # ensures feasibility
    c = rng.uniform(-5, 5, size=n_vars)  # random objective

    return A, b, c, x_true

def solve_with_cvxpy(A, b, c):
    n_vars = A.shape[1]
    x = cp.Variable(n_vars)
    objective = cp.Minimize(c @ x)
    constraints = [A @ x <= b, x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return x.value, problem.value  # return the optimal solution and the optimal value


def test_simplex():
    # Generate a random LP problem  
    A, b, c, x_true = generate_random_lp(seed=42)

    x_cvx, val_cvx = solve_with_cvxpy(A, b, c)  # CVXPY's solution
    x_my, val_my = simplex(c, A, b)  # My implementation's solution

    assert np.allclose(x_cvx, x_my)
    assert np.isclose(val_cvx, val_my)