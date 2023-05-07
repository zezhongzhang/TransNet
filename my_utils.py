import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_2d_tri(x1, x2, value, title='title', figsize=(8,6), shape=False, equal_ratio=True):
    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x1, x2)

    # Mask off unwanted triangles.
    if shape == 'annulus':
        tol = 1e-7 + 1.0
        if np.max(np.abs(x1))>tol or np.max(np.abs(x2)) > tol:
            raise ValueError('Only support [-1,1] range for annulus.')
        # Mask off unwanted triangles.
        min_radius = 0.5
        triang.set_mask(np.hypot(x1[triang.triangles].mean(axis=1),
                                 x2[triang.triangles].mean(axis=1))
                        < min_radius)
    if shape == 'L-shape':
        tol = 1e-7 + 1.0
        if np.max(np.abs(x1)) > tol or np.max(np.abs(x2)) > tol:
            raise ValueError('Only support [-1,1] range for annulus.')

        dx = x2[1] - x2[0]
        v1 = x1[triang.triangles]
        v2 = x2[triang.triangles]

        x1_diff = np.stack([v1[:, 0] - v1[:, 1], v1[:, 1] - v1[:, 2], v1[:, 2] - v1[:, 0]], axis=1)
        x2_diff = np.stack([v2[:, 0] - v2[:, 1], v2[:, 1] - v2[:, 2], v2[:, 2] - v2[:, 0]], axis=1)

        tol = dx * 1.2
        exceed_ct = np.sum((x1_diff > tol) * 1 + (x2_diff > tol) * 1, axis=1)
        bd_mask = exceed_ct > 0
        triang.set_mask(bd_mask)


    fig1, ax1 = plt.subplots(figsize=figsize)
    if equal_ratio:
        ax1.set_aspect('equal')

    tpc = ax1.tricontourf(triang, value, levels=20, cmap='jet')
    fig1.colorbar(tpc)
    ax1.set_title(title)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.grid(which='both')
    plt.show()

def plot_2d_mesh(x1, x2, value, title='title', figsize=(8,6), equal_ratio=True, z_min=None, z_max=None):
    #  plot
    fig, (ax, cax) = plt.subplots(ncols=2,figsize=figsize,
                      gridspec_kw={"width_ratios": [1, 0.05]})
    fig.subplots_adjust(wspace=0.1)
    im = ax.pcolormesh(x1, x2, value, cmap='jet', shading='auto', vmin=z_min, vmax=z_max)
    # ax.plot(0,0,'r*')
    fig.colorbar(im, cax=cax)
    if equal_ratio:
        ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def plot_domain_2d(x_pde=None, x_bd=None, x_ic=None, x_test=None, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    if x_test is not None:
        plt.plot(x_test[:, 0], x_test[:, 1], 'g.', alpha=0.5)

    if x_pde is not None:
        plt.plot(x_pde[:, 0], x_pde[:, 1], 'b.')

    if x_bd is not None:
        plt.plot(x_bd[:, 0], x_bd[:, 1], 'r.')

    if x_ic is not None:
        plt.plot(x_ic[:, 0], x_ic[:, 1], 'y.')

    plt.grid(which='both')
    plt.title('problem domain (train)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

