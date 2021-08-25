import numpy as np


def create_hessian(df_dx1_dx1, df_dx1_dx2, df_dx2_dx1, df_dx2_dx2):
    """
    Create hessian from the second partial derivatives.

    Args:
        df_dxi_dxj:
            partial derivatives wrt xi, xj. size (k,)

    Returns:
        Hessian:
            size: (k, 2, 2)
    """
    assert df_dx1_dx1.shape == df_dx1_dx2.shape
    assert df_dx2_dx1.shape == df_dx1_dx2.shape
    assert df_dx2_dx2.shape == df_dx2_dx1.shape

    Hess = np.array([[df_dx1_dx1, df_dx1_dx2],
                     [df_dx2_dx1, df_dx2_dx2]]).transpose(2, 3, 0, 1)
    assert Hess.shape == (*df_dx1_dx1.shape, 2, 2)

    return Hess


def get_finite_second_diff_Hessian(grid, evaluate):
    """
    Get hessian from finite second differences of the function values
    on a grid.

    Args:
        grid:
            Grid instance (see grid.py).
        evaluate:
            evaluation functional that hessian is computed for.

    Returns:
        Hess:
            size: (k, 2, 2)
    """
    assert grid.x.dtype == np.float64
    z = evaluate(grid.x)
    z_2D = z.reshape(grid.meshgrid_size)

    # indexes (i,j) = (y,x)
    df_dx1 = np.diff(z_2D, axis=1)[:-1, :] / grid.h
    df_dx2 = np.diff(z_2D, axis=0)[:, :-1] / grid.h

    # second-order derivatives up to h^2
    df_dx1_dx1 = np.diff(df_dx1, axis=1)[:-1, :] / grid.h
    df_dx1_dx2 = np.diff(df_dx2, axis=1)[:-1, :] / grid.h
    df_dx2_dx1 = np.diff(df_dx1, axis=0)[:, :-1] / grid.h
    df_dx2_dx2 = np.diff(df_dx2, axis=0)[:, :-1] / grid.h

    # Hessian up to h^2
    Hess = create_hessian(df_dx1_dx1, df_dx1_dx2, df_dx2_dx1, df_dx2_dx2)

    # Check symmetry of Hessian
    Hess_transpose = Hess.transpose(0, 1, 3, 2)
    assert np.all(np.allclose(Hess, Hess_transpose))

    return Hess


def get_exact_grad_Hessian(grid, grad_eval):
    """
    Get hessian from finite first differences of the exact gradient
    on a grid.

    Args:
        grid:
            Grid instance (see grid.py).
        grad_eval:
            function to evaluate gradient.

    Returns:
        Hess:
            size: (k, 2, 2)
    """
    assert grid.x.dtype == np.float64
    input = grid.x
    x_grad = grad_eval(input)
    x_grad_2D = x_grad.reshape((*grid.meshgrid_size, 2))

    df_dx1 = x_grad_2D[:, :, 0]
    df_dx2 = x_grad_2D[:, :, 1]

    # indexes (i,j) = (y,x)
    # second-order derivatives up to h^2
    df_dx1_dx1 = np.diff(df_dx1, axis=1)[:-1, :] / grid.h
    df_dx1_dx2 = np.diff(df_dx2, axis=1)[:-1, :] / grid.h
    df_dx2_dx1 = np.diff(df_dx1, axis=0)[:, :-1] / grid.h
    df_dx2_dx2 = np.diff(df_dx2, axis=0)[:, :-1] / grid.h

    # Hessian up to h^2
    Hess = create_hessian(df_dx1_dx1, df_dx1_dx2, df_dx2_dx1, df_dx2_dx2)

    # Check non-symmetry of Hessian
    Hess_transpose = Hess.transpose(0, 1, 3, 2)
    assert not np.all(np.allclose(Hess, Hess_transpose))

    return Hess


def get_exact_Hessian(grid, hessian_eval):
    """
    Get hessian from 'exact' partial second derivatives computation
    on a grid.

    Args:
        grid:
            Grid instance (see grid.py).
        hessian_eval:
            function to evaluate hessian.

    Returns:
        Hess:
            size: (k, 2, 2)
    """
    input = grid.x
    x_hessian = hessian_eval(input)
    # Hessian up to h^2
    Hess = x_hessian.reshape((*grid.meshgrid_size, 2, 2))

    # Check symmetry of Hessian
    Hess_transpose = Hess.transpose(0, 1, 3, 2)
    assert np.all(np.allclose(Hess, Hess_transpose))

    return Hess
