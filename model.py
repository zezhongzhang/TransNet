import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class PDE_basis(object):
    def __init__(self, x_dim, basis_num, nlin_type='tanh', include_const=True):
        self.x_dim = x_dim
        self.nlin_type = nlin_type
        self.include_const = include_const
        self.basis_num = basis_num

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

    def set_nlin(self, nlin_type):
        self.nlin_type = nlin_type

    def set_Wb(self, W, b):
        self.weight_0 = W
        self.bias_0 = b
        self.set_eval_range(self.eval_range)

    def init_scale_loc(self, scale, loc0_range):
        # initialize basis in [-1,1]^n
        x_dim = self.x_dim
        basis_num = self.basis_num

        # nd Gaussian
        weight = np.random.randn(basis_num, x_dim)
        weight = weight/np.sqrt(np.sum(weight**2, axis=1, keepdims=True))

        # random loc in [-1,1]^n
        loc0 = np.random.rand(basis_num, x_dim)*2 - 1
        loc0 = loc0*loc0_range
        self.info['loc0'] = loc0

        # random scale
        # scale = np.ones(basis_num)*scale_gamma
        self.info['scale'] = scale

        b = -np.sum(weight*loc0, axis=1)

        self.set_Wb(W=weight * scale, b=b*scale)

    def init_pde_basis(self, shape, radius):
        # initialize basis in [-1,1]^n
        x_dim = self.x_dim
        basis_num = self.basis_num

        # nd random unit vectors
        weight = np.random.randn(basis_num, x_dim)
        weight = weight/np.sqrt(np.sum(weight**2, axis=1, keepdims=True))

        # random loc in [-1,1]^n
        # b = np.linspace(0, 1, basis_num)*radius
        b = np.random.rand(basis_num)*radius

        self.set_Wb(W=weight*shape, b=b*shape)

    def init_scale_range(self, scale_range):
        # initialize basis in [-1,1]^n
        x_dim = self.x_dim
        basis_num = self.basis_num

        # nd Gaussian
        weight = np.random.randn(basis_num, x_dim)
        weight = weight/np.sqrt(np.sum(weight**2, axis=1, keepdims=True))

        # random loc in [-1,1]^n
        loc0 = np.random.rand(basis_num, x_dim)*2 - 1
        self.info['loc0'] = loc0

        # random scale
        scale = np.random.rand(basis_num)*(scale_range[1] - scale_range[0]) + scale_range[0]
        self.info['scale'] = scale

        b = -np.sum(weight*loc0, axis=1)

        self.weight_0 = weight*scale[:, None]
        self.bias_0 = b*scale

        self.set_eval_range(self.eval_range)

    def init_gaussian(self, sd=1):
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        basis_num = self.basis_num
        x_dim = self.x_dim
        self.weight_0 = np.random.randn(basis_num, x_dim)*sd
        self.bias_0 = np.random.randn(basis_num)*sd
        self.set_eval_range(self.eval_range)

    def init_unif(self, a=1, b=1):
        # weight:   (basis_num, x_dim)
        # bias:     (basis_num)
        basis_num = self.basis_num
        x_dim = self.x_dim
        self.weight_0 = (np.random.rand(basis_num, x_dim)*2-1)*a
        self.bias_0 = (np.random.rand(basis_num)*2-1)*b

        self.set_eval_range(self.eval_range)

    def init_dnn(self, init_type='default'):
        if init_type == 'default':
            temp = nn.Linear(in_features=self.x_dim, out_features=self.basis_num)
            weight = temp.weight.data.numpy().astype('float64')
            bias = temp.bias.data.numpy().astype('float64')
            self.set_Wb(W=weight, b=bias)

    def eval_basis(self, x_in, eval_list=('u',)):
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
            bias = np.concatenate([self.bias, np.zeros(1)], axis=0)
        else:
            # weight:   (basis_num, x_dim)
            # bias:     (basis_num)
            weight = self.weight
            bias = self.bias
        z_pre = np.matmul(x_in, weight.T) + bias
        z_eval = self.nonlinear(z_pre, max_order=max_order)
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
                eval_all[eval_item] = z_eval[item_order-1]*weight[:, diff_1]*weight[:, diff_2]
        return eval_all

    def plot_basis_2d(self, basis_id=0, figsize=(8,6)):
        assert self.x_dim == 2, 'dim should be 2 for this plot'
        basis_x1_range = self.eval_range[0]
        basis_x2_range = self.eval_range[1]

        mesh_size = 50
        x1_1d = np.linspace(basis_x1_range[0], basis_x1_range[1], mesh_size)
        x2_1d = np.linspace(basis_x2_range[0], basis_x2_range[1], mesh_size)
        x1_2d, x2_2d = np.meshgrid(x1_1d, x2_1d, indexing='xy')

        # plot basis
        x_in = np.stack([x1_2d.flatten(), x2_2d.flatten()], axis=1)
        basis_eval_all = self.eval_basis(x_in=x_in, eval_list=['u'])['u']

        a = self.weight[basis_id, 0]
        b = self.weight[basis_id, 1]
        c = self.bias[basis_id]
        cut_line = -(a/b) * x1_1d - (c/b)

        value = basis_eval_all[:, basis_id].reshape(mesh_size, mesh_size)
        # z_min1, z_max1 = -np.abs(value).max(), np.abs(value).max()
        z_min1, z_max1 = -1, 1

        fig, (ax, cax) = plt.subplots(ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.05]})
        fig.subplots_adjust(wspace=0.1)
        im = ax.pcolormesh(x1_2d, x2_2d, value, cmap='RdBu', vmin=z_min1, vmax=z_max1)
        fig.colorbar(im, cax=cax)
        if 'scale' in self.info.keys():
            basis_scale = self.info['scale']
            scale = basis_scale[basis_id]
            ax.set_title(f'basis {basis_id}: scale={scale:.2f}')
        else:
            ax.set_title(f'basis {basis_id}')
        ax.plot(x1_1d, cut_line, 'y')

        if 'loc0' in self.info.keys():
            basis_loc0 = self.info['loc0']
            basis_loc0 = self.shift_x(basis_loc0, self.eval_range)
            loc0 = basis_loc0[basis_id]
            ax.plot(loc0[0], loc0[1], 'r*')

        ax.set_xlim(basis_x1_range[0], basis_x1_range[1])
        ax.set_ylim(basis_x2_range[0], basis_x2_range[1])
        ax.grid(True)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.show()

    def plot_cut_2d(self, show_loc=True, figsize=(8,6)):
        assert self.x_dim == 2, 'dim should be 2 for this plot'

        a = self.weight[:, 0]
        b = self.weight[:, 1]
        c = self.bias

        basis_x1_range = self.eval_range[0]
        basis_x2_range = self.eval_range[1]

        x1_mesh = 100
        x = np.linspace(basis_x1_range[0], basis_x1_range[1], x1_mesh)
        # y = -(a/b) * x - (c/b)

        # y(sample_size, x1_mesh)
        y = -(a[:,None]/b[:,None]) * x[None,:] - (c[:,None]/b[:,None])

        plt.figure(figsize=figsize)
        plt.plot(x, y.T)
        if show_loc:
            loc0 = self.info['loc0']
            loc0 = self.shift_x(loc0, self.eval_range)
            plt.plot(loc0[:,0], loc0[:,1], 'r.')
        plt.xlim(basis_x1_range[0], basis_x1_range[1])
        plt.ylim(basis_x2_range[0], basis_x2_range[1])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('cut lines')
        plt.show()

    def nonlinear(self, x_in, max_order=0):
        if self.nlin_type == 'tanh':
            return self.nonlinear_tanh(x_in=x_in, order=max_order)
        elif self.nlin_type == 'sin':
            return self.nonlinear_sin(x_in=x_in, order=max_order)
        else:
            raise ValueError(f'Invalid nlin_type.')

    @staticmethod
    def shift_x(x_in, eval_range):
        # x_in [-1,1]^n
        lower = eval_range[:, 0]
        upper = eval_range[:, 1]
        x_shift = (x_in + 1)/2*(upper - lower) + lower
        return x_shift

    @staticmethod
    def shift_coef(weight, bias, eval_range):
        # shift coef defined in [-1,1] in a new range
        lower = eval_range[:,0]
        upper = eval_range[:,1]
        scale_w = (upper - lower)/2
        scale_b = (upper + lower)/2

        weight_shifted = weight/scale_w[None, :]
        bias_shifted = bias - np.matmul(weight, scale_b[:, None]/ scale_w[:, None])[:,0]

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
    def nonlinear_sigmoid(x_in, order=0):
        pass

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


class LS_Base(object):
    def __init__(self):
        pass
    
    @staticmethod
    def ls_fit(data, weights=None, ls_mse=True, item_mse=False):
        # data: dictionary of feature and targets
        # each feature is [N,d] and target is [N,1]
        # weights must match keys in data

        # concatenate all LS features
        feature_all = []
        target_all = []
        for key, value in data.items():
            if weights is None:
                w = 1.0
            else:
                w = weights[key]
            feature_all.append(value[0]*w)
            target_all.append(value[1]*w)
        feature_all = np.concatenate(feature_all, axis=0)
        target_all = np.concatenate(target_all, axis=0)

        # lS solution
        coef, _, r, _ = np.linalg.lstsq(feature_all, target_all, rcond=None)


        info = {}
        # get feature shape
        info['feature_shape'] = [feature_all.shape]
        # get feature rank
        info['rank'] = r
        # get overall mse
        if ls_mse:
            fitted = np.matmul(feature_all, coef)
            res = fitted - target_all
            ls_mse = np.mean(res ** 2)
            info['ls_mse'] = ls_mse
            info['fitted'] = fitted
            info['target'] = target_all
        # get itemized LS mse
        if item_mse:
            for key, value in data.items():
                item_feature = value[0]
                item_target = value[1]
                item_res = np.matmul(item_feature, coef) - item_target
                item_rmse = np.mean(item_res ** 2)
                info[f'ls_mse_{key}'] = item_rmse
        return coef, info

    @staticmethod
    def get_mse_problem(x_in, basis, problem, coef_sol):
        # ceof: dict of out_var coef
        sol_true = problem.u_exact(x_in=x_in)
        out_var_list = problem.out_var
        num_var = len(out_var_list)

        # get test mse
        info = {'x_in': x_in}
        mse_all = 0
        for i in range(num_var):
            out_var = out_var_list[i]
            coef = coef_sol[out_var]
            target = sol_true[out_var]

            mse, info = LS_Base.get_mse_data(x_in=x_in, y_true=target, basis=basis, coef=coef)
            fitted = info['fitted']

            mse_all = mse_all + mse

            info[f'{out_var}_mse'] = mse
            info[f'{out_var}_fitted'] = fitted
            info[f'{out_var}_true'] = target

        mse_all = mse_all/num_var
        return mse_all, info

    @staticmethod
    def get_mse_data(x_in, y_true, basis, coef):
        # coef(N,1)
        basis_test = basis.eval_basis(x_in=x_in, eval_list=['u'])['u']
        y_hat = np.matmul(basis_test, coef)
        mse = np.mean((y_hat - y_true) ** 2)

        info={}
        info['fitted'] = y_hat
        return mse, info


class TrainLS(LS_Base):
    def __init__(self, problem, basis=None):
        super(TrainLS, self).__init__()
        self.problem = problem

        #
        self.basis = None

        if basis is not None:
            self.set_basis(basis=basis)
        #
        self.info = {}


    def set_basis(self, basis):
        self.basis = basis
        eval_range = np.stack([self.problem.x_pde.min(axis=0), self.problem.x_pde.max(axis=0)], axis=1)
        self.basis.set_eval_range(eval_range)
        # self.basis_x_pde = self.basis.eval_basis(self.problem.x_pde, eval_list=self.problem.eval_list_pde)
        # self.basis_x_test = self.basis.eval_basis(self.problem.x_test, eval_list=['u'])
        #
        # if self.problem.x_bd is not None:
        #     self.basis_x_bd = self.basis.eval_basis(self.problem.x_bd, eval_list=['u'])
        #
        # if self.problem.x_ic is not None:
        #     self.basis_x_ic = self.basis.eval_basis(self.problem.x_ic, eval_list=['u'])

    def ls_pde_picard(self, max_iter=20, weights=None, verbose=False):
        # picard
        current_ceof = None
        info = {}
        train_ls_mse = []
        basis_x_pde = self.basis.eval_basis(self.problem.x_pde, eval_list=['u'])
        for i in range(max_iter):
            coef_sol, info1 = self.ls_pde(current_ceof=current_ceof, weights=weights,
                                          ls_mse=True, item_mse=False, basis_x_pde=basis_x_pde)

            current_ceof = coef_sol
            ls_mse = info1['ls_mse']
            train_ls_mse.append(ls_mse)

            if verbose:
                print(f'iter {i}\tls_mse={ls_mse}')
            # rmse_test, info2 = train_ls.get_rmse_test(coef_sol=coef_sol)
            # print('\nTest RMSE:', rmse_test)
            # test_rmse.append(rmse_test)
        info['train_ls_mse'] = train_ls_mse
        return current_ceof, info

    def ls_pde(self, current_ceof=None, weights=None, ls_mse=True, item_mse=False, basis_x_pde=None):
        if current_ceof is None:
            feature_all = self.get_feature_all(current_sol=None)
        else:
            if basis_x_pde is None:
                basis_x_pde = self.basis.eval_basis(self.problem.x_pde, eval_list=['u'])
            current_sol = {}
            for var in self.problem.out_var:
                current_sol[var] = np.matmul(basis_x_pde['u'], current_ceof[var])
            feature_all = self.get_feature_all(current_sol=current_sol)

        coef, info = self.ls_fit(feature_all, weights=weights, ls_mse=ls_mse, item_mse=item_mse)
        out_var = self.problem.out_var
        basis_num = int(coef.shape[0]/len(out_var))

        # process coef
        coef_sol = {}
        for i in range(len(out_var)):
            coef_sol[out_var[i]] = coef[i*basis_num: (i+1)*basis_num]
        info['feature_all'] = feature_all
        return coef_sol, info

    def get_feature_all(self, current_sol=None):
        feature_all = {}
        # pde features
        for feature_name in self.problem.eq_names:
            feature_all[feature_name] = self.problem.ls_feature_pde(feature_name=feature_name,
                                                                    basis=self.basis,
                                                                    current_sol=current_sol)
        # value features
        value_feature = self.problem.ls_feature_value(basis=self.basis)
        for key, value in value_feature.items():
            feature_all[key] = value
        return feature_all


    def get_mse_test(self, coef_sol):
        # ceof: dict of out_var coef
        basis = self.basis
        problem = self.problem
        x_in = problem.x_test
        target_test = problem.target_test
        out_var_list = problem.out_var


        num_var = len(out_var_list)

        # get test mse
        info1 = {'x_in': x_in}
        mse_all = 0
        for i in range(num_var):
            out_var = out_var_list[i]
            coef = coef_sol[out_var]
            target = target_test[out_var]

            mse, info = LS_Base.get_mse_data(x_in=x_in, y_true=target, basis=basis, coef=coef)
            fitted = info['fitted']

            mse_all = mse_all + mse

            info1[f'{out_var}_mse'] = mse
            info1[f'{out_var}_fitted'] = fitted

        mse_all = mse_all/num_var
        return mse_all, info1


