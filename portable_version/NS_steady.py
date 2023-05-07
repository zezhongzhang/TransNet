import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import gc
import time


class TransNet(object):
    def __init__(self, x_dim, basis_num, include_const=True, nlin_type='tanh'):
        self.x_dim = x_dim
        self.include_const = include_const
        self.basis_num = basis_num
        self.nlin_type = nlin_type

        # basis evaluate range
        self.eval_range = np.array([[-1, 1]]*x_dim)

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
        weight = weight/np.sqrt(np.sum(weight**2, axis=1, keepdims=True))

        # random loc in [-1,1]^n
        # b = np.linspace(0, 1, basis_num)*radius
        b = np.random.rand(basis_num)*radius
        self.set_W0_b0(W0=weight*shape, b0=b*shape)

    def init_gaussian(self, sd=1):
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        basis_num = self.basis_num
        x_dim = self.x_dim
        W0 = np.random.randn(basis_num, x_dim)*sd
        b0 = np.random.randn(basis_num)*sd
        self.set_W0_b0(W0=W0, b0=b0)

    def init_unif(self, a=1, b=1):
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        basis_num = self.basis_num
        x_dim = self.x_dim
        W0 = (np.random.rand(basis_num, x_dim)*2-1)*a
        b0 = (np.random.rand(basis_num)*2-1)*b
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
                eval_all[eval_item] = z_eval[item_order-1]
            elif item_order == 2:
                diff_1 = int(eval_item[1])
                eval_all[eval_item] = z_eval[item_order-1]*weight[:, diff_1]
            elif item_order == 3:
                diff_1 = int(eval_item[1])
                diff_2 = int(eval_item[2])
                # eval_all[eval_item] = z_eval[item_order-1]*weight[:, diff_1]*weight[:, diff_2]
                eval_all[eval_item] = z_eval[item_order-1]*(weight[:, diff_1]*weight[:, diff_2])
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
        lower = eval_range[:,0]
        upper = eval_range[:,1]
        scale_w = (upper - lower)/2
        scale_b = (upper + lower)/2

        weight_shifted = weight/scale_w[None, :]
        bias_shifted = bias - np.matmul(weight, scale_b[:, None]/ scale_w[:, None])[:, 0]

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
            z_1 = 1 - z_0**2
            return [z_0, z_1]
        elif order == 2:
            z_0 = np.tanh(x_in)
            z_1 = 1 - z_0**2
            z_2 = -2*z_0*z_1
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

def sample_2d_mesh(x_mesh, y_mesh, x_min=-1., x_max=1., y_min=-1., y_max=1.):
    # sample
    x_1d = np.linspace(x_min, x_max, x_mesh)
    y_1d = np.linspace(y_min, y_max, y_mesh)

    x_2d, y_2d = np.meshgrid(x_1d, y_1d)

    x_sample = np.stack([x_2d.flatten(), y_2d.flatten()], axis=1)
    return x_sample


def u_exact(x_in, re):
    x = x_in[:, [0]]
    y = x_in[:, [1]]
    nu = 1.0/re
    lamb = 1.0/(2.0*nu) - np.sqrt(1.0/(4.0*nu**2) + 4*np.pi**2)

    u = 1 - np.exp(lamb*x)*np.cos(2*np.pi*y)
    v = lamb/(2*np.pi) * np.exp(lamb*x) * np.sin(2*np.pi*y)
    p = 0.5*(1 - np.exp(2*lamb*x))
    return {'u': u, 'v': v, 'p': p}

def check_sol(x_test, basis, coef_u, coef_v, coef_p):
    basis_test = basis.eval_basis(x_in=x_test, eval_list=['u'])['u']
    u_hat = np.matmul(basis_test, coef_u)
    v_hat = np.matmul(basis_test, coef_v)
    p_hat = np.matmul(basis_test, coef_p)

    sol_ture_eval = u_exact(x_in=x_test, re=re)

    mse_u = np.mean((u_hat - sol_ture_eval['u']) ** 2)
    mse_v = np.mean((v_hat - sol_ture_eval['v']) ** 2)
    mse_p = np.mean((p_hat - sol_ture_eval['p']) ** 2)
    mse = (mse_u + mse_v + mse_p)/3
    return mse


if __name__ == '__main__':
    # variable order: [x,y]
    # set domain
    x_min, x_max = -0.5, 1
    y_min, y_max = -0.5, 1.5

    # sampling mesh
    x_mesh = 50
    y_mesh = 50

    # NS parameter
    re = 40

    # basis setting
    basis_num = 1000
    shape = 2
    radius = 1.5
    ##################################################
    # x_pde/x_bd
    print('Sampling:')
    x_pde = sample_2d_mesh(x_mesh, y_mesh, x_min, x_max, y_min, y_max)
    x_bd = np.concatenate([x_pde[x_pde[:, 0] == x_min], x_pde[x_pde[:, 0] == x_max], x_pde[x_pde[:, 1] == y_min],
                           x_pde[x_pde[:, 1] == y_max]], axis=0)
    print('\tCollocation points: x_pde ', x_pde.shape)
    print('\tBoundary points: x_bd', x_bd.shape)

    # setup basis
    basis = TransNet(x_dim=2, basis_num=basis_num)
    basis.init_pde_basis(shape=shape, radius=radius)
    basis.set_eval_range(np.array([[x_min, x_max], [y_min, y_max]]))

    # initial solution
    coef_u = np.zeros((basis_num + 1, 1))
    coef_v = np.zeros((basis_num + 1, 1))
    coef_p = np.zeros((basis_num + 1, 1))

    # get LS features:
    # coef: [ceof_u, coef_v, coef_p]
    col_num = (basis_num + 1) * 3
    row_bd = x_bd.shape[0]
    row_pde = x_pde.shape[0]
    row_num = row_bd * 3 + row_pde * 3
    fea_mem_GB = row_num * col_num * 8 / 1e+9
    tar_mem_GB = row_num * 1 * 8 / 1e+9
    print(f'\nExpected LS size: ({row_num}, {col_num})/({row_num}, 1)')
    print(f'\tfeature memory size: {fea_mem_GB:.4f} GB')
    print(f'\ttarget memory size: {tar_mem_GB:.4f} GB')

    # initialize LS matrix
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
    #

    # setup boundary features/targets
    target_bd = u_exact(x_in=x_bd, re=re)
    basis_eval = basis.eval_basis(x_in=x_bd, eval_list=['u'])
    basis_bd = basis_eval['u']
    # bd_u
    # feature_bd_u = np.concatenate([basis_bd, np.zeros_like(basis_bd), np.zeros_like(basis_bd)], axis=1)
    feature_all[ind_bd_u, 0:(basis_num + 1)] = basis_bd
    target_all[ind_bd_u, :] = target_bd['u']
    # bd_v
    # feature_bd_v = np.concatenate([np.zeros_like(basis_bd), basis_bd, np.zeros_like(basis_bd)], axis=1)
    feature_all[ind_bd_v, (basis_num + 1):2 * (basis_num + 1)] = basis_bd
    target_all[ind_bd_v, :] = target_bd['v']
    # bd_p
    # feature_bd_p = np.concatenate([np.zeros_like(basis_bd), np.zeros_like(basis_bd), basis_bd], axis=1)
    feature_all[ind_bd_p, 2 * (basis_num + 1):3 * (basis_num + 1)] = basis_bd
    target_all[ind_bd_p, :] = target_bd['p']
    del basis_eval, basis_bd, target_bd
    gc.collect()

    # divergence free
    basis_eval = basis.eval_basis(x_in=x_pde, eval_list=['u0', 'u1'])
    z_x = basis_eval['u0']
    z_y = basis_eval['u1']
    # divergence free condition
    # feature_div = np.concatenate([z_x, z_y, np.zeros_like(z_x)], axis=1)
    feature_all[ind_div, 0:(basis_num + 1)] = z_x
    feature_all[ind_div, (basis_num + 1):2 * (basis_num + 1)] = z_y
    del basis_eval, z_x, z_y
    gc.collect()

    ####################################################
    # NS PDE features ##################################
    ####################################################
    nu = 1.0 / re
    basis_eval = basis.eval_basis(x_pde, eval_list=['u', 'u0', 'u1', 'u00', 'u11'])
    z = basis_eval['u']
    z_x = basis_eval['u0']
    z_xx = basis_eval['u00']
    z_y = basis_eval['u1']
    z_yy = basis_eval['u11']

    u_eval = np.matmul(z, coef_u)
    v_eval = np.matmul(z, coef_v)

    del z, basis_eval
    gc.collect()

    # pde_u: u*u_x + v*u_y - (u_xx + u_yy)*nu + p_x
    # feature_pde_u = np.concatenate([u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy, np.zeros_like(z_x), z_x], axis=1)
    feature_all[ind_pde_u, :(basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy
    feature_all[ind_pde_u, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_x

    # pde_v: u*v_x + v*v_y - (v_xx + v_yy)*nu + p_y
    # feature_pde_v = np.concatenate([np.zeros_like(z_x), u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy, z_y], axis=1)
    feature_all[ind_pde_v,
    1 * (basis_num + 1):2 * (basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy
    feature_all[ind_pde_v, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_y

    del z_x, z_y, z_xx, z_yy
    gc.collect()
    ####################################################

    # setup test points
    x_test = sample_2d_mesh(x_mesh=100, y_mesh=100, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    print('x_test:', x_test.shape)

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

    # get variable coef: [ceof_u, coef_v, coef_p]
    coef_u = coef[: (basis_num + 1)]
    coef_v = coef[(basis_num + 1): (basis_num + 1) * 2]
    coef_p = coef[(basis_num + 1) * 2: (basis_num + 1) * 3]
    ################################################


    # test MSE
    mse_test = check_sol(x_test, basis, coef_u, coef_v, coef_p)
    print('\tTest MSE: ', mse_test)


    ################################################
    # Picard iteration
    iter_picard = 20
    print('\nBegin Picard iteration:')
    for i in range(iter_picard):
        print('iter:', i)
        # (update feature matrix)
        ####################################################
        # NS PDE features ##################################
        ####################################################
        nu = 1.0 / re
        basis_eval = basis.eval_basis(x_pde, eval_list=['u', 'u0', 'u1', 'u00', 'u11'])
        z = basis_eval['u']
        z_x = basis_eval['u0']
        z_xx = basis_eval['u00']
        z_y = basis_eval['u1']
        z_yy = basis_eval['u11']

        u_eval = np.matmul(z, coef_u)
        v_eval = np.matmul(z, coef_v)

        del z, basis_eval
        gc.collect()

        # pde_u: u*u_x + v*u_y - (u_xx + u_yy)*nu + p_x
        # feature_pde_u = np.concatenate([u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy, np.zeros_like(z_x), z_x], axis=1)
        feature_all[ind_pde_u, :(basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy
        feature_all[ind_pde_u, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_x

        # pde_v: u*v_x + v*v_y - (v_xx + v_yy)*nu + p_y
        # feature_pde_v = np.concatenate([np.zeros_like(z_x), u_eval*z_x + v_eval*z_y - nu*z_xx - nu*z_yy, z_y], axis=1)
        feature_all[ind_pde_v,
        1 * (basis_num + 1):2 * (basis_num + 1)] = u_eval * z_x + v_eval * z_y - nu * z_xx - nu * z_yy
        feature_all[ind_pde_v, 2 * (basis_num + 1):3 * (basis_num + 1)] = z_y

        del z_x, z_y, z_xx, z_yy
        gc.collect()
        ####################################################

        ################################################
        # Solving LS
        print('Solving Least Square:')
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

        # get variable coef: [ceof_u, coef_v, coef_p]
        coef_u = coef[: (basis_num + 1)]
        coef_v = coef[(basis_num + 1): (basis_num + 1) * 2]
        coef_p = coef[(basis_num + 1) * 2: (basis_num + 1) * 3]
        ################################################

        # test MSE
        mse_test = check_sol(x_test, basis, coef_u, coef_v, coef_p)
        print('\ttest_mse: ', mse_test)