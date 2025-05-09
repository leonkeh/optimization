import numpy as np


def simplex(c_, A_, b_):
    """solve problems of the standard form: min c^T x; s.t. Ax<=b and x>=0"""
    # do some assertions
    assert np.linalg.matrix_rank(A_) == A_.shape[0], "A_ should be full rank"
    assert A_.shape[0] == b_.shape[0], "A_ and b_ should have the same number of rows"
    assert A_.shape[1] == c_.shape[0], "A_ and c_ should have the same number of columns"

    # the simplex algorithm
    n = A_.shape[1]  # number of decision variables in the original problem
    m = A_.shape[0]  # number of constraints
    x_B = np.ones((m, 1))  # basic variables (slack variables to make ineq. constraints to eq. constraints)
    x_N = np.ones((n, 1))
    B = np.eye(m)
    N = A_
    c_N = c_.reshape((-1, 1))
    c_B = np.zeros((m, 1))

    p = - np.inf * np.ones(x_N.shape)
    while (p < 0).all():
        x_B = np.linalg.inv(B) @ (b_ - N @ x_N)
        z_0 = c_B.T @ np.linalg.inv(B) @ b_
        p = (c_N.T - c_B.T @ np.linalg.inv(B) @ N).T
        i = np.argmin([p[i] for i in range(n)]) # actually p_i should be less than 0 accoridng to slides
        y = (np.linalg.inv(B) @ N[:, i]).reshape((-1, 1))
        eps = 1e-5
        j = np.argmin([np.nanmin([float(x_B[j]/y[j]), eps]) for j in range(m)])
        N_i = N[:, i].copy()
        B_j = B[:, j].copy()
        N[:, i] = B_j
        B[:, j] = N_i
    print(f"The optimal solution is {x_B}")


# testing with some dummy problem
c_ = np.array([1., 0]).reshape((-1, 1))
A_ = np.array([[-1., -1.],
              [-1, 0.]])
b_ = np.array([-3., -1.]).reshape((-1, 1))
simplex(c_, A_, b_)


