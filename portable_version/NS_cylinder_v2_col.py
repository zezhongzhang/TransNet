import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import gc
from scipy.stats import qmc
import time


class TransNet(object):
    def __init__(self, x_dim, basis_num, include_const=True, nlin_type='tanh'):
        self.x_dim = x_dim
        self.include_const = include_const
        self.basis_num = basis_num
        self.nlin_type = nlin_type

        # basis evaluate range
        self.eval_range = np.array([[-1, 1]] * x_dim)

        # coef in [-1,1]^n
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        self.weight_0 = None
        self.bias_0 = None

        # coef in problem domain
        self.weight = None
        self.bias = None

        # info container
        self.info = {}

    def __repr__(self):
        text = f'basis num={self.basis_num} \t nlin_type={self.nlin_type} \t evaluate range='
        for i in range(self.x_dim):
            text += f'{self.eval_range[i]}' if i == 0 else f'*{self.eval_range[i]}'
        return text

    def set_eval_range(self, eval_range):
        assert eval_range.shape == self.eval_range.shape, f'eval_range dose not match shape of current eval range={self.eval_range.shape}'
        weight_shifted, bias_shifted = self.shift_coef(weight=self.weight_0, bias=self.bias_0, eval_range=eval_range)
        self.eval_range = eval_range
        self.weight = weight_shifted
        self.bias = bias_shifted

    def print_eval_range(self):
        print(self.eval_range)

    def set_W0_b0(self, W0, b0):
        # set W0,b0 and shift them to eval_range
        self.weight_0 = W0
        self.bias_0 = b0
        self.set_eval_range(self.eval_range)

    def init_pde_basis(self, shape, radius):
        """
        generate uniform random basis
        :param shape: the shape parameter in paper, which determine the bandwidth for each basis
        :param radius: the radius for uniform L2 ball with uniform cutting plane density
        :return: None
        """
        # initialize basis in [-1,1]^n
        x_dim = self.x_dim
        basis_num = self.basis_num

        # nd random unit vectors
        weight = np.random.randn(basis_num, x_dim)
        weight = weight / np.sqrt(np.sum(weight ** 2, axis=1, keepdims=True))

        # random loc in [-1,1]^n
        # b = np.linspace(0, 1, basis_num)*radius
        b = np.random.rand(basis_num) * radius
        self.set_W0_b0(W0=weight * shape, b0=b * shape)

    def init_gaussian(self, sd=1):
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        basis_num = self.basis_num
        x_dim = self.x_dim
        W0 = np.random.randn(basis_num, x_dim) * sd
        b0 = np.random.randn(basis_num) * sd
        self.set_W0_b0(W0=W0, b0=b0)

    def init_unif(self, a=1, b=1):
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        basis_num = self.basis_num
        x_dim = self.x_dim
        W0 = (np.random.rand(basis_num, x_dim) * 2 - 1) * a
        b0 = (np.random.rand(basis_num) * 2 - 1) * b
        self.set_W0_b0(W0=W0, b0=b0)

    def init_dnn(self, init_type='default'):
        if init_type == 'default':
            temp = nn.Linear(in_features=self.x_dim, out_features=self.basis_num)
            W0 = temp.weight.data.numpy().astype('float64')
            b0 = temp.bias.data.numpy().astype('float64')
            self.set_W0_b0(W0=W0, b0=b0)

    def eval_basis(self, x_in, eval_list=('u',)):
        """
        evaluate the basis and derivatives
        :param x_in: points of evaluation
        :param eval_list: a list of str indicating evaluation variables.
        e.g. 'u': just evaluate the basis; 'u0' evaluate d_basis/d_x0; 'u00' evaluate d^2_basis/d_x0^2
        :return: a dictionary of basis evaluation
        """
        if self.weight is None:
            raise NotImplementedError('Basis not initialized')
        # get max order and check
        max_order = 0
        for eval_item in eval_list:
            item_order = len(eval_item)
            max_order = np.maximum(max_order, item_order)
        max_order -= 1
        assert max_order <= 2, f'{max_order} not implemented.'

        # evaluate basis and derivatives
        if self.include_const:
            # weight:   (basis_num+1, x_dim)
            # bias:     (basis_num+1)
            weight = np.concatenate([self.weight, np.zeros((1, self.x_dim))], axis=0)
            bias = np.concatenate([self.bias, np.ones(1)], axis=0)
        else:
            # weight:   (basis_num, x_dim)
            # bias:     (basis_num)
            weight = self.weight
            bias = self.bias
        z_pre = np.matmul(x_in, weight.T) + bias
        z_eval = self.nonlinear(z_pre, max_order=max_order)
        # evaluate all basis and derivatives
        eval_all = {}
        for eval_item in eval_list:
            item_order = len(eval_item)
            if item_order == 1:
                eval_all[eval_item] = z_eval[item_order - 1]
            elif item_order == 2:
                diff_1 = int(eval_item[1])
                eval_all[eval_item] = z_eval[item_order - 1] * weight[:, diff_1]
            elif item_order == 3:
                diff_1 = int(eval_item[1])
                diff_2 = int(eval_item[2])
                # eval_all[eval_item] = z_eval[item_order-1]*weight[:, diff_1]*weight[:, diff_2]
                eval_all[eval_item] = z_eval[item_order - 1] * (weight[:, diff_1] * weight[:, diff_2])
        return eval_all

    def nonlinear(self, x_in, max_order=0):
        if self.nlin_type == 'tanh':
            return self.nonlinear_tanh(x_in=x_in, order=max_order)
        elif self.nlin_type == 'sin':
            return self.nonlinear_sin(x_in=x_in, order=max_order)
        else:
            raise ValueError(f'Invalid nlin_type.')

    @staticmethod
    def shift_coef(weight, bias, eval_range):
        # shift coef defined in [-1,1] in a new range
        lower = eval_range[:, 0]
        upper = eval_range[:, 1]
        scale_w = (upper - lower) / 2
        scale_b = (upper + lower) / 2

        weight_shifted = weight / scale_w[None, :]
        bias_shifted = bias - np.matmul(weight, scale_b[:, None] / scale_w[:, None])[:, 0]

        # weight_shifted = weight*scale_w[None, :]
        # bias_shifted = bias + np.matmul(weight, scale_b[:, None])[:,0]
        return weight_shifted, bias_shifted

    @staticmethod
    def nonlinear_tanh(x_in, order=0):
        if order == 0:
            z_0 = np.tanh(x_in)
            return [z_0]
        elif order == 1:
            z_0 = np.tanh(x_in)
            z_1 = 1 - z_0 ** 2
            return [z_0, z_1]
        elif order == 2:
            z_0 = np.tanh(x_in)
            z_1 = 1 - z_0 ** 2
            z_2 = -2 * z_0 * z_1
            return [z_0, z_1, z_2]
        else:
            raise NotImplementedError(f'order {order} not implemented.')

    @staticmethod
    def nonlinear_sin(x_in, order=0):
        if order == 0:
            z_0 = np.sin(x_in)
            return [z_0]
        elif order == 1:
            z_0 = np.sin(x_in)
            z_1 = np.cos(x_in)
            return [z_0, z_1]
        elif order == 2:
            z_0 = np.sin(x_in)
            z_1 = np.cos(x_in)
            z_2 = -np.sin(x_in)
            return [z_0, z_1, z_2]
        else:
            raise NotImplementedError(f'order {order} not implemented.')


def sample_LHS(n, upper, lower):
    d = upper.shape[0]
    sampler = qmc.LatinHypercube(d=d)
    z = sampler.random(n=n)
    sample = z * (upper - lower)[None, :] + lower[None, :]
    return sample


if __name__ == '__main__':
    # variables: [t,x,y]
    # set domain
    t_min, t_max = 0, 0.1  # set t_max here
    x_min, x_max = 0, 2.2
    y_min, y_max = 0, 0.41

    # cylinder location and radius
    cld_r = 0.05
    cld_c = np.array([0.2, 0.2])

    # NS parameter
    nu = 0.001

    #########################################
    # collocation points (interior)
    n_pde_background = 5000  # uniform on domain
    n_pde_reinforce = 1000  # close to cylinder

    # cylinder boundary points
    n_cld = 50
    t_mesh = 5

    # number of boundary point on each side
    n_bd_x = 1000
    n_bd_y = 1000
    n_bd_ic = 3000

    # basis number
    basis_num = 1000
    ####################################

    # x_pde_background: background points that cover all space equally
    txy_min = np.array([t_min, x_min, y_min])
    txy_max = np.array([t_max, x_max, y_max])
    txy_pde_background = sample_LHS(n=n_pde_background, upper=txy_max, lower=txy_min)

    # x_pde_reinforce: points close to cylinder
    txy_rein_min = np.array([t_min, 0.1, 0.05])
    txy_rein_max = np.array([t_max, 2, 0.35])
    txy_pde_reinforce = sample_LHS(n=n_pde_reinforce, upper=txy_rein_max, lower=txy_rein_min)

    # remove points inside cylinder
    txy_pde_background = txy_pde_background[
        (txy_pde_background[:, 1] - cld_c[0]) ** 2 + (txy_pde_background[:, 2] - cld_c[1]) ** 2 > cld_r ** 2]
    txy_pde_reinforce = txy_pde_reinforce[
        (txy_pde_reinforce[:, 1] - cld_c[0]) ** 2 + (txy_pde_reinforce[:, 2] - cld_c[1]) ** 2 > cld_r ** 2]

    # points on cylinder boundary
    angles = np.linspace(0, 2 * np.pi, n_cld)
    xy_cld = np.stack([np.cos(angles), np.sin(angles)]) * cld_r
    xy_cld = xy_cld.T + cld_c
    t_1d = np.linspace(t_min, t_max, t_mesh)
    txy_cld = np.concatenate([np.concatenate([np.ones((n_cld, 1)) * t, xy_cld], axis=1) for t in t_1d], axis=0)

    # combine all points
    # add cylinder boundary points to collocation
    txy_pde = np.concatenate([txy_pde_background, txy_pde_reinforce, txy_cld], axis=0)

    print('Collocation points: txy_pde ', txy_pde.shape)
    print('\ttxy_pde_background: ', txy_pde_background.shape)
    print('\ttxy_pde_reinforce: ', txy_pde_reinforce.shape)
    print('\ttxy_cld: ', txy_cld.shape)

    # domain boundary sampling
    # x_min/x_max
    txy_bd_left = sample_LHS(n=n_bd_x, upper=txy_max, lower=txy_min)
    txy_bd_right = sample_LHS(n=n_bd_x, upper=txy_max, lower=txy_min)
    txy_bd_left[:, 1] = x_min
    txy_bd_right[:, 1] = x_max

    # y_min/y_max
    txy_bd_top = sample_LHS(n=n_bd_y, upper=txy_max, lower=txy_min)
    txy_bd_down = sample_LHS(n=n_bd_y, upper=txy_max, lower=txy_min)
    txy_bd_top[:, 2] = y_max
    txy_bd_down[:, 2] = y_min

    # initial: t=t_min
    txy_ic = sample_LHS(n=n_bd_ic, upper=txy_max, lower=txy_min)
    txy_ic[:, 0] = t_min

    txy_bd = np.concatenate([txy_bd_left, txy_bd_right, txy_bd_top, txy_bd_down, txy_ic, txy_cld], axis=0)
    print('Boundary points: txy_bd', txy_bd.shape)
    print('\ttxy_bd_left', txy_bd_left.shape)
    print('\ttxy_bd_right', txy_bd_right.shape)
    print('\ttxy_bd_top', txy_bd_top.shape)
    print('\ttxy_bd_down', txy_bd_down.shape)
    print('\ttxy_ic', txy_ic.shape)
    print('\ttxy_cld', txy_cld.shape)


    # construct fitting target for LS
    def target_bd_fun(txy):
        # boundary condition on left and right
        t = txy[:, 0]
        x = txy[:, 1]
        y = txy[:, 2]
        target = 6.0 / 0.41 ** 2 * np.sin(np.pi * t / 8.) * y * (0.41 - y)
        return target


    # boundary target
    # u condition
    target_bd_u_left = target_bd_fun(txy_bd_left)[:, None]
    target_bd_u_right = target_bd_fun(txy_bd_right)[:, None]
    target_bd_u_top = np.zeros((txy_bd_top.shape[0], 1))
    target_bd_u_down = np.zeros((txy_bd_down.shape[0], 1))
    target_ic = np.zeros((txy_ic.shape[0], 1))
    target_cld = np.zeros((txy_cld.shape[0], 1))
    target_bd_u = np.concatenate(
        [target_bd_u_left, target_bd_u_right, target_bd_u_top, target_bd_u_down, target_ic, target_cld], axis=0)

    # v,p condition
    target_bd_v = np.zeros(txy_bd.shape[0])[:, None]
    target_bd_p = np.zeros(txy_bd.shape[0])[:, None]
    # print('\nboundary targets:')
    # print('\tbd_u:', txy_bd.shape, target_bd_u.shape)
    # print('\tbd_v:', txy_bd.shape, target_bd_v.shape)
    # print('\tbd_p:', txy_bd.shape, target_bd_p.shape)

    # set up basis
    basis = TransNet(x_dim=3, basis_num=basis_num)
    basis.init_pde_basis(shape=2, radius=1.7)
    basis.set_eval_range(np.array([[t_min, t_max], [x_min, x_max], [y_min, y_max]]))

    # initial solution
    coef_u = np.zeros((basis_num + 1, 1))
    coef_v = np.zeros((basis_num + 1, 1))
    coef_p = np.zeros((basis_num + 1, 1))

    # get LS features:
    # coef: [ceof_u, coef_v, coef_p]
    col_num = (basis_num + 1) * 3
    row_bd = txy_bd.shape[0]
    row_pde = txy_pde.shape[0]
    row_num = row_bd * 3 + row_pde * 3
    fea_mem_GB = row_num * col_num * 8 / 1e+9
    tar_mem_GB = row_num * 1 * 8 / 1e+9
    print(f'\nExpected LS size: ({row_num}, {col_num})/({row_num}, 1)')
    print(f'\tfeature memory size: {fea_mem_GB:.4f} GB')
    print(f'\ttarget memory size: {tar_mem_GB:.4f} GB')

    # initialize
    feature_all = np.zeros((row_num, col_num))
    target_all = np.zeros((row_num, 1))

    # get all feature indices
    ind_bd_u = np.ones(row_bd, dtype=np.int32) * 1
    ind_bd_v = np.ones(row_bd, dtype=np.int32) * 2
    ind_bd_p = np.ones(row_bd, dtype=np.int32) * 3
    ind_div = np.ones(row_pde, dtype=np.int32) * 4
    ind_pde_u = np.ones(row_pde, dtype=np.int32) * 5
    ind_pde_v = np.ones(row_pde, dtype=np.int32) * 6
    ind_all = np.concatenate([ind_bd_u, ind_bd_v, ind_bd_p, ind_div, ind_pde_u, ind_pde_v], axis=0)

    ind_bd_u = ind_all == 1
    ind_bd_v = ind_all == 2
    ind_bd_p = ind_all == 3
    ind_div = ind_all == 4
    ind_pde_u = ind_all == 5
    ind_pde_v = ind_all == 6

    # boundary features
    basis_eval = basis.eval_basis(x_in=txy_bd, eval_list=['u'])
    basis_bd = basis_eval['u']
    # bd_u
    # feature_bd_u = np.concatenate([basis_bd, np.zeros_like(basis_bd), np.zeros_like(basis_bd)], axis=1)
    feature_all[ind_bd_u, 0:(basis_num + 1)] = basis_bd
    target_all[ind_bd_u, :] = target_bd_u
    # bd_v
    # feature_bd_v = np.concatenate([np.zeros_like(basis_bd), basis_bd, np.zeros_like(basis_bd)], axis=1)
    feature_all[ind_bd_v, (basis_num + 1):2 * (basis_num + 1)] = basis_bd
    target_all[ind_bd_v, :] = target_bd_v
    # bd_p
    # feature_bd_p = np.concatenate([np.zeros_like(basis_bd), np.zeros_like(basis_bd), basis_bd], axis=1)
    feature_all[ind_bd_p, 2 * (basis_num + 1):3 * (basis_num + 1)] = basis_bd
    target_all[ind_bd_p, :] = target_bd_p
    del basis_eval, basis_bd
    gc.collect()

    # divergence free
    basis_eval = basis.eval_basis(x_in=txy_pde, eval_list=['u1', 'u2', ])
    z_x = basis_eval['u1']
    z_y = basis_eval['u2']
    # divergence free condition
    # feature_div = np.concatenate([z_x, z_y, np.zeros_like(z_x)], axis=1)
    feature_all[ind_div, 0:(basis_num + 1)] = z_x
    feature_all[ind_div, (basis_num + 1):2 * (basis_num + 1)] = z_y
    del basis_eval, z_x, z_y
    gc.collect()

    ####################################################
    # NS PDE features ##################################
    ####################################################
    basis_eval = basis.eval_basis(txy_pde, eval_list=['u', 'u0', 'u1', 'u2', 'u11', 'u22'])
    z = basis_eval['u']
    z_t = basis_eval['u0']
    z_x = basis_eval['u1']
    z_xx = basis_eval['u11']
    z_y = basis_eval['u2']
    z_yy = basis_eval['u22']

    u_eval = np.matmul(z, coef_u)
    v_eval = np.matmul(z, coef_v)

    del z, basis_eval
    gc.collect()

    # pde_u: u*u_x + v*u_y - (u_xx + u_yy)*nu + p_x + u_t
    # feature_pde_u = np.concatenate([u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy + z_t, np.zeros_like(z_x), z_x], axis=1)
    feature_all[ind_pde_u, :(basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy + z_t
    feature_all[ind_pde_u, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_x

    # pde_v: u*v_x + v*v_y - (v_xx + v_yy)*nu + p_y + v_t
    # feature_pde_v = np.concatenate([np.zeros_like(z_x), u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy + z_t, z_y], axis=1)
    feature_all[ind_pde_v,
    1 * (basis_num + 1):2 * (basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy + z_t
    feature_all[ind_pde_v, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_y

    del z_t, z_x, z_y, z_xx, z_yy
    gc.collect()
    ####################################################

    ################################################
    # Solving LS
    print('\nSolving Least Square:')
    # solve LS
    t1 = time.time()
    coef, _, r, _ = np.linalg.lstsq(feature_all, target_all, rcond=None)
    t2 = time.time()
    print('\ttime:', t2 - t1)
    print('\tfeature rank:', r)
    fitted = np.matmul(feature_all, coef)
    res = fitted - target_all
    ls_mse = np.mean(res ** 2)
    print('\tLS MSE:', ls_mse)
    ################################################

    ################################################
    # Picard iteration
    iter_picard = 1
    print('Begin Picard iteration:')
    for i in range(iter_picard):
        print('iter:', i)
        # coef: [ceof_u, coef_v, coef_p]
        coef_u = coef[: (basis_num + 1)]
        coef_v = coef[(basis_num + 1): (basis_num + 1) * 2]
        # coef_p = coef[(basis_num + 1)*2 : (basis_num + 1)*3]

        # (update feature matrix)
        ####################################################
        # NS PDE features ##################################
        ####################################################
        basis_eval = basis.eval_basis(txy_pde, eval_list=['u', 'u0', 'u1', 'u2', 'u11', 'u22'])
        z = basis_eval['u']
        z_t = basis_eval['u0']
        z_x = basis_eval['u1']
        z_xx = basis_eval['u11']
        z_y = basis_eval['u2']
        z_yy = basis_eval['u22']

        u_eval = np.matmul(z, coef_u)
        v_eval = np.matmul(z, coef_v)

        del z, basis_eval
        gc.collect()

        # pde_u: u*u_x + v*u_y - (u_xx + u_yy)*nu + p_x + u_t
        # feature_pde_u = np.concatenate([u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy + z_t, np.zeros_like(z_x), z_x], axis=1)
        feature_all[ind_pde_u, :(basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy + z_t
        feature_all[ind_pde_u, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_x
        # pde_v: u*v_x + v*v_y - (v_xx + v_yy)*nu + p_y + v_t
        # feature_pde_v = np.concatenate([np.zeros_like(z_x), u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy + z_t, z_y], axis=1)
        feature_all[ind_pde_v,
        1 * (basis_num + 1):2 * (basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy + z_t
        feature_all[ind_pde_v, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_y
        del z_t, z_x, z_y, z_xx, z_yy
        gc.collect()
        ####################################################

        ################################################
        # Solving LS
        print('Solving Least Square...')
        # solve LS
        t1 = time.time()
        coef, _, r, _ = np.linalg.lstsq(feature_all, target_all, rcond=None)
        t2 = time.time()
        print('\ttime:', t2 - t1)
        print('\tfeature rank:', r)
        fitted = np.matmul(feature_all, coef)
        res = fitted - target_all
        ls_mse = np.mean(res ** 2)
        print('\tLS MSE:', ls_mse)
        ################################################
        print()

    # plot vorticity
    x_test_mesh = 200
    y_test_mesh = 50
    t_eval = t_max

    x_2d, y_2d = np.meshgrid(np.linspace(x_min, x_max, x_test_mesh), np.linspace(y_min, y_max, y_test_mesh),
                             indexing='ij')
    x_test = np.concatenate(
        [np.ones((x_test_mesh * y_test_mesh, 1)) * t_eval, x_2d.flatten()[:, None], y_2d.flatten()[:, None]], axis=1)

    coef_u = coef[: (basis_num + 1)]
    coef_v = coef[(basis_num + 1): (basis_num + 1) * 2]
    coef_p = coef[(basis_num + 1) * 2: (basis_num + 1) * 3]

    basis_eval = basis.eval_basis(x_in=x_test, eval_list=['u', 'u1', 'u2'])
    z = basis_eval['u']
    est_u = np.matmul(z, coef_u).reshape((x_test_mesh, y_test_mesh))
    est_v = np.matmul(z, coef_v).reshape((x_test_mesh, y_test_mesh))
    est_p = np.matmul(z, coef_p).reshape((x_test_mesh, y_test_mesh))

    # vorticity
    z_x = basis_eval['u1']
    z_y = basis_eval['u2']
    est_u_y = np.matmul(z_y, coef_u).reshape((x_test_mesh, y_test_mesh))
    est_v_x = np.matmul(z_x, coef_u).reshape((x_test_mesh, y_test_mesh))
    vor = est_v_x - est_u_y

    # plt.figure(figsize=(12, 2.4))
    # plt.streamplot(x_2d.T, y_2d.T, est_u.T, est_v.T, density=1, color='b')
    # plt.pcolormesh(x_2d, y_2d, est_p)
    # plt.plot(*xy_cld.T, 'r.')
    # plt.title(f'solution at {t_eval}')
    # plt.show()

    # plot vorticity
    z_max = np.abs(vor).max()
    z_min = -z_max
    fig, (ax, cax) = plt.subplots(ncols=2, figsize=(12, 2.4),
                                  gridspec_kw={"width_ratios": [1, 0.05]})
    fig.subplots_adjust(wspace=0.1)
    im = ax.pcolormesh(x_2d, y_2d, vor, cmap='jet', shading='auto', vmin=z_min, vmax=z_max)
    ax.plot(*xy_cld.T, 'r.')
    fig.colorbar(im, cax=cax)
    ax.set_aspect('equal')
    ax.set_title(f'vorticity at {t_eval}')
    ax.grid(True)
    plt.show()