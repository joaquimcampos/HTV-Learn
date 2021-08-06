import numpy as np


def create_hessian(df_dx1_dx1, df_dx1_dx2, df_dx2_dx1, df_dx2_dx2):
    """ """
    assert df_dx1_dx1.shape == df_dx1_dx2.shape
    assert df_dx2_dx1.shape == df_dx1_dx2.shape
    assert df_dx2_dx2.shape == df_dx2_dx1.shape

    Hess = np.array([[df_dx1_dx1, df_dx1_dx2],
                     [df_dx2_dx1, df_dx2_dx2]]).transpose(2, 3, 0, 1)
    assert Hess.shape == (*df_dx1_dx1.shape, 2, 2)

    return Hess


def get_finite_second_diff_Hessian(grid, evaluate):
    """
    Args:
        grid: grid object
        evaluate: evaluation functional
    """
    assert grid.x.dtype == np.float64
    z = evaluate(grid.x)
    z_2D = z.reshape(grid.meshgrid_size)

    # indexes (i,j) = (y,x)
    df_dx1 = np.diff(z_2D, axis=1)[:-1, :] / grid.h
    df_dx2 = np.diff(z_2D, axis=0)[:, :-1] / grid.h
    # df_dx2, df_dx1 = np.gradient(z_2D, grid.h)

    # df_dx1 = np.clip(df_dx1, -1, 1)
    # df_dx2 = np.clip(df_dx2, -1, 1)

    print('Lefki grad')
    print(df_dx1.min(), df_dx1.max())
    print(df_dx2.min(), df_dx2.max())

    # second-order derivatives up to h^2
    df_dx1_dx1 = np.diff(df_dx1, axis=1)[:-1, :] / grid.h
    df_dx1_dx2 = np.diff(df_dx2, axis=1)[:-1, :] / grid.h
    df_dx2_dx1 = np.diff(df_dx1, axis=0)[:, :-1] / grid.h
    df_dx2_dx2 = np.diff(df_dx2, axis=0)[:, :-1] / grid.h
    # df_dx2_dx1, df_dx1_dx1 = np.gradient(df_dx1, grid.h)
    # df_dx2_dx2, df_dx1_dx2 = np.gradient(df_dx2, grid.h)

    # Hessian up to h^2
    Hess = create_hessian(df_dx1_dx1, df_dx1_dx2, df_dx2_dx1, df_dx2_dx2)

    # Check symmetry of Hessian
    Hess_transpose = Hess.transpose(0, 1, 3, 2)
    assert np.all(np.allclose(Hess, Hess_transpose))

    return Hess


def get_exact_grad_Hessian(grid, grad_eval):
    """
    Args:
        grid: grid object
        grad_eval: gradient evaluation function
    """
    assert grid.x.dtype == np.float64
    input = grid.x
    x_grad = grad_eval(input)
    x_grad_2D = x_grad.reshape((*grid.meshgrid_size, 2))

    df_dx1 = x_grad_2D[:, :, 0]
    df_dx2 = x_grad_2D[:, :, 1]

    print('Exact grad')
    print(df_dx1.min(), df_dx1.max())
    print(df_dx2.min(), df_dx2.max())

    # indexes (i,j) = (y,x)
    # second-order derivatives up to h^2
    df_dx1_dx1 = np.diff(df_dx1, axis=1)[:-1, :] / grid.h
    df_dx1_dx2 = np.diff(df_dx2, axis=1)[:-1, :] / grid.h
    df_dx2_dx1 = np.diff(df_dx1, axis=0)[:, :-1] / grid.h
    df_dx2_dx2 = np.diff(df_dx2, axis=0)[:, :-1] / grid.h
    # df_dx2_dx1, df_dx1_dx1 = np.gradient(df_dx1, grid.h)
    # df_dx2_dx2, df_dx1_dx2 = np.gradient(df_dx2, grid.h)

    # Hessian up to h^2
    Hess = create_hessian(df_dx1_dx1, df_dx1_dx2, df_dx2_dx1, df_dx2_dx2)

    # Check non-symmetry of Hessian
    Hess_transpose = Hess.transpose(0, 1, 3, 2)
    assert not np.all(np.allclose(Hess, Hess_transpose))

    return Hess


def get_exact_Hessian(grid, hessian_eval):
    """ """
    # # convert to float32 for possible feed to network
    # input = grid.x.astype('float32')
    input = grid.x
    x_hessian = hessian_eval(input)
    # Hessian up to h^2
    Hess = x_hessian.reshape((*grid.meshgrid_size, 2, 2))

    # Check symmetry of Hessian
    Hess_transpose = Hess.transpose(0, 1, 3, 2)
    assert np.all(np.allclose(Hess, Hess_transpose))

    return Hess
