import numpy as np
import matplotlib.pyplot as plt
import torch
# from my_utils import grad




class Domain(object):
    def __init__(self, domain_range, domain_shape='box', mesh_domain=None, mesh_boundary=None):
        domain_range = np.array(domain_range)
        domain_range = domain_range.reshape((domain_range.shape[0] // 2, 2))
        self.domain_range = domain_range
        self.domain_dim = domain_range.shape[0]

        # check shape
        self.valid_shapes = ['box', 'annulus', 'circle', 'L-shape']
        if domain_shape in self.valid_shapes:
            pass
        else:
            raise ValueError(f'shape must be in {self.valid_shapes}')

        # different shape only valid in 2d
        if self.domain_dim == 2 or domain_shape == 'box':
            self.domain_shape = domain_shape
        else:
            raise NotImplementedError(f'{domain_shape} not implemented in {self.domain_dim}d')

        if mesh_domain is not None:
            x_train_domain = self.sample_domain_uniform(mesh_size=mesh_domain)
            self.x_train_domain = x_train_domain
        else:
            self.x_train_domain = None

        if mesh_boundary is not None:
            x_train_bd = self.sample_boundary_uniform(sample_size=mesh_boundary)
            self.x_train_bd = x_train_bd
        else:
            self.x_train_bd = None

    def __repr__(self):
        text = f'{self.domain_dim}d {self.domain_shape} domain with range: '
        for i in range(self.domain_dim):
            text += f'{self.domain_range[i]}' if i == 0 else f'*{self.domain_range[i]}'
        return text

    def sample_domain_uniform(self, mesh_size):
        domain_dim = self.domain_dim

        if type(mesh_size) == int:
            mesh_size = [mesh_size] * domain_dim
        elif type(mesh_size) == list and len(mesh_size) == self.domain_dim:
            pass
        else:
            raise ValueError(f'mesh_vec must be list of length {self.domain_dim} or int')

        # generate samples on [-1,1]^n
        x_train_domain_standard = self.sample_nd_mesh(mesh_size)

        if self.domain_shape == 'circle':
            assert self.domain_dim == 2, 'Dim should be 2 for circle.'
            r = np.sqrt(np.sum(x_train_domain_standard ** 2, axis=1))
            index1 = r <= 1.0
            x_train_domain_standard = x_train_domain_standard[index1, :]
            print(x_train_domain_standard.shape)

        if self.domain_shape == 'annulus':
            assert self.domain_dim == 2, 'Dim should be 2 for annulus.'
            r = np.sqrt(np.sum(x_train_domain_standard ** 2, axis=1))
            index1 = r <= 1.0
            index2 = r >= 0.5
            x_train_domain_standard = x_train_domain_standard[index1 * index2, :]

        if self.domain_shape == 'L-shape':
            assert self.domain_dim == 2, 'Dim should be 2 for L-shape.'
            index1 = np.sum(x_train_domain_standard > 0, axis=1) <= 1
            x_train_domain_standard = x_train_domain_standard[index1, :]

        # shift and scale by domain range
        x_train_domain = self.shift2range(x_train_domain_standard)
        return x_train_domain

    def sample_boundary_uniform(self, sample_size):
        if self.domain_shape == 'circle':
            assert self.domain_dim == 2, 'Dim should be 2 for circle.'
            x_train_bd = self.sample_circle_uniform(r=1, sample_size=sample_size)

        if self.domain_shape == 'annulus':
            assert self.domain_dim == 2, 'Dim should be 2 for circle.'
            inner_size = int(sample_size / 3)
            outer_size = sample_size - inner_size
            x_train_boundary_out = self.sample_circle_uniform(r=1, sample_size=outer_size)
            x_train_boundary_in = self.sample_circle_uniform(r=0.5, sample_size=inner_size)
            x_train_bd = np.concatenate([x_train_boundary_out, x_train_boundary_in], axis=0)

        if self.domain_shape == 'L-shape':
            assert self.domain_dim == 2, 'Dim should be 2 for L-shape.'
            sample_1_side = int(np.ceil(sample_size / np.exp2(self.domain_dim))) + 1
            mesh_vec = [sample_1_side] * self.domain_dim
            x_train_bd = self.sample_nd_mesh_bd(mesh_vec=mesh_vec)
            # reshape to L-shape
            x_train_bd = x_train_bd - 1.0 * (x_train_bd == 1) * (np.min(x_train_bd, axis=1,
                                                                        keepdims=True) > 0)

        if self.domain_shape == 'box':
            sample_1_side = sample_size/self.domain_dim/2
            mesh_vec = [int(np.ceil(sample_1_side**(1.0/(self.domain_dim-1))))] * self.domain_dim
            x_train_bd = self.sample_nd_mesh_bd(mesh_vec=mesh_vec)

        # shift and scale by domain range
        x_train_bd = self.shift2range(x_train_bd)
        return x_train_bd

    def shift2range(self, x):
        lb = self.domain_range[:, 0]
        ub = self.domain_range[:, 1]
        x = (x * 0.5 + 0.5) * (ub - lb) + lb
        return x

    @staticmethod
    def sample_circle_uniform(r, sample_size):
        theta_vec = np.linspace(0, 2 * np.pi, sample_size + 1)[:-1]
        x_bd = np.stack([np.cos(theta_vec), np.sin(theta_vec)], axis=1) * r
        return x_bd

    @staticmethod
    def sample_nd_mesh_bd(mesh_vec):
        # generate uniform samples on boundary of [-1,1]^n
        # mesh_size: list of int mesh for each dim
        domain_dim = len(mesh_vec)
        if domain_dim >= 2:
            sample_all = []
            for i in range(domain_dim):
                bd_samples = Domain.sample_nd_mesh(np.delete(mesh_vec, i))
                side1 = np.insert(bd_samples, obj=i, values=1, axis=1)
                side2 = np.insert(bd_samples, obj=i, values=-1, axis=1)
                sample_all.append(side1)
                sample_all.append(side2)
            # generate samples on [-1,1]^n
            box_bd = np.concatenate(sample_all, axis=0)
        else:
            box_bd = np.array([[-1], [1]])
        return box_bd

    @staticmethod
    def sample_nd_mesh(mesh_vec):
        # generate uniform samples on [-1,1]^n
        # mesh_size: list of int mesh for each dim
        domain_dim = len(mesh_vec)
        x_domain_standard = []
        temp_ones = np.ones(mesh_vec)
        for i in range(domain_dim):
            mesh_1d = np.linspace(-1, 1, mesh_vec[i])
            dim_array = np.ones(domain_dim, int)
            dim_array[i] = -1
            mesh_nd = mesh_1d.reshape(dim_array) * temp_ones
            x_domain_standard.append(mesh_nd.flatten())
        x_domain_standard = np.stack(x_domain_standard, axis=1)
        return x_domain_standard


class Problem(object):
    def __init__(self, case=None, data=None):
        self.case = case

        self.x_pde = None
        self.x_bd = None
        self.x_ic = None
        self.x_test = None

        self.target_pde = None
        self.target_bd = None
        self.target_ic = None
        self.target_test = None

        if data is not None:
            self.from_data(data)

        #
        self.pde_name = None
        self.eval_list_pde = None
        self.operator_type = None
        self.eq_names = None
        self.out_var = None

    def from_data(self, data):
        if data is not None:
            if 'x_pde' in data.keys():
                self.x_pde = data['x_pde']

                if len(data['target_pde'].shape) == 0:
                    self.target_pde = data['target_pde'].item()
                else:
                    self.target_pde = data['target_pde']

            if 'x_test' in data.keys():
                self.x_test = data['x_test']
                if len(data['target_test'].shape) == 0:
                    self.target_test = data['target_test'].item()
                else:
                    self.target_test = data['target_test']

            if 'x_bd' in data.keys():
                self.x_bd = data['x_bd']
                if len(data['target_bd'].shape) == 0:
                    self.target_bd = data['target_bd'].item()
                else:
                    self.target_bd = data['target_bd']

            if 'x_ic' in data.keys():
                self.x_ic = data['x_ic']
                if len(data['target_ic'].shape) == 0:
                    self.target_ic = data['target_ic'].item()
                else:
                    self.target_ic = data['target_ic']

    def set_data(self, x_pde,  x_test, target_test, target_pde=None, x_bd=None, x_ic=None, target_bd=None, target_ic=None):
        self.x_pde = x_pde
        self.x_bd = x_bd
        self.x_ic = x_ic
        self.x_test = x_test

        self.target_pde = target_pde
        self.target_bd = target_bd
        self.target_ic = target_ic
        self.target_test = target_test

    def __repr__(self):
        sep = '*****************' * 3
        text_pde = f'{self.pde_name} (case={self.case}):'
        text_train = f'\tx_pde:   \t{None if self.x_pde is None else self.x_pde.shape}'
        text_test = f'\tx_test:    \t{None if self.x_test is None else self.x_test.shape}'
        text_train_bd = f'\tx_bd:    \t{None if self.x_bd is None else self.x_bd.shape}'
        text_train_ic = f'\tx_ic:    \t{None if self.x_ic is None else self.x_ic.shape}'
        return '\n'.join([sep, text_pde, text_train, text_train_bd, text_train_ic, text_test, sep])

    def u_exact(self, x_in):
        # u_exact are used to check solution error and problem
        raise NotImplementedError('Not Implemented')

    def rhs(self, x_in):
        # right hand side of pde (forcing terms)
        raise NotImplementedError('Not Implemented')

    def lhs(self, *args, **kwargs):
        # right hand side of pde (pde operators)
        raise NotImplementedError('Not Implemented')

    def check_solution(self, x_in):
        # right hand side of pde (pde operators)
        raise NotImplementedError('Not Implemented')

    @staticmethod
    def get_grad_auto(u, x):
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:,
              [0]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:,
               [0]]
        return u_x, u_xx

class Poisson_2d(Problem):
    def __init__(self, case=1, data=None):
        super(Poisson_2d, self).__init__(case, data)
        assert case in [1, 2, 3, 4], f'case {case} not implemented'

        self.pde_name = 'Poisson'
        self.eval_list_pde = ['u', 'u00', 'u11']
        self.eq_names = ['pde']
        self.out_var = 'u'

    def u_exact(self, x_in):
        x = x_in[:, [0]]
        y = x_in[:, [1]]
        if self.case == 1:
            if type(x) == torch.Tensor:
                return {'u': torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)}
            else:
                return {'u': np.sin(2*np.pi*x) * np.sin(2*np.pi*y)}

        if self.case == 2:
            if type(x) == torch.Tensor:
                return {'u': torch.cos(np.pi*x) * torch.cos(np.pi*y)}
            else:
                return {'u': np.cos(np.pi*x) * np.cos(np.pi*y)}

        if self.case == 3:
            if type(x) == torch.Tensor:
                return {'u': x * (1 - x) * y * (1 - y) * torch.exp(x - y)}
            else:
                return {'u': x * (1 - x) * y * (1 - y) * np.exp(x - y)}

        if self.case == 4:
            if type(x) == torch.Tensor:
                return {'u': torch.sin(4 * np.pi * x) * torch.sin(4 * np.pi * y)}
            else:
                return {'u': np.sin(4*np.pi*x) * np.sin(4*np.pi*y)}

    def rhs(self, x_in):
        x = x_in[:, [0]]
        y = x_in[:, [1]]
        if self.case == 1:
            return np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * (-8)*np.pi**2

        if self.case == 2:
            return np.cos(np.pi*x) * np.cos(np.pi*y)*(-2)*np.pi**2

        if self.case == 3:
            return 2 * x * (y - 1) * (y - 2*x + x*y + 2) * np.exp(x - y)

        if self.case == 4:
            return np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * (-32)*np.pi**2

    def lhs(self, u_xx, u_yy):
        # Poisson operator
        return u_xx + u_yy

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        x = x_in[:, [0]]
        y = x_in[:, [1]]

        # get auto grad
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        x_t.requires_grad_()
        y_t.requires_grad_()
        x_in_t = torch.cat([x_t, y_t], dim=1)
        u = self.u_exact(x_in_t)['u']
        # grad
        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)
        u_y, u_yy = self.get_grad_auto(u=u, x=y_t)

        # eval
        u_xx = u_xx.detach().numpy()
        u_yy = u_yy.detach().numpy()
        lhs_eval = self.lhs(u_xx, u_yy)
        rhs_eval = self.rhs(x_in)
        rmse = np.sqrt(np.mean((lhs_eval-rhs_eval)**2))
        print(f'pde rmse: {rmse}')
        return x_in, u.detach().numpy()

    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)

        feature_pde = basis_eval['u00'] + basis_eval['u11']
        target_pde = self.rhs(x_in)
        return [feature_pde, target_pde]

    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_bd = basis_eval['u']
        feature_bd_all = {'bd_u': [feature_bd, self.target_bd['u']]}
        return feature_bd_all


class Fokker_Planck_1d(Problem):
    def __init__(self, case=None, data=None):
        super(Fokker_Planck_1d, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'

        self.pde_name = 'Fokker_Planck'
        self.eval_list_pde = ['u', 'u0', 'u1', 'u11']
        self.eq_names = ['pde']
        self.out_var = 'u'

        # model param
        self.sigma0 = 0.4
        self.mu0 = 0.0
        self.sigma_SDE = 0.3

        # drift param
        self.mu_freq = 3
        self.mu_mag = 2

    def mu_sol(self, t):
        mag = self.mu_mag
        freq = self.mu_freq
        if type(t) == torch.Tensor:
            mu_t = mag * torch.sin(freq * t) / freq + self.mu0
        else:
            mu_t = mag * np.sin(freq * t) / freq + self.mu0
        return mu_t

    def mu_SDE(self, t, x):
        mag = self.mu_mag
        freq = self.mu_freq
        mu = mag * np.cos(freq * t)
        return mu

    def sigma_sol(self, t):
        sigma0 = self.sigma0
        sigma_SDE = self.sigma_SDE
        if type(t) == torch.Tensor:
            sigma_t = torch.sqrt(sigma0 ** 2 + t * sigma_SDE ** 2)
        else:
            sigma_t = np.sqrt(sigma0 ** 2 + t * sigma_SDE ** 2)
        return sigma_t


    @staticmethod
    def gaussian_1d(x, mu, sigma):
        if type(x) == torch.Tensor:
            p = torch.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))
        else:
            p = np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))
        return p

    def p_true(self, t, x):
        mu_true = self.mu_sol(t)
        sigma_true = self.sigma_sol(t)
        p = self.gaussian_1d(x, mu_true, sigma_true)
        return p

    def u_exact(self, x_in):
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        mu_true = self.mu_sol(t)
        sigma_true = self.sigma_sol(t)
        p = self.gaussian_1d(x, mu_true, sigma_true)
        return{'u': p}

    def rhs(self, x_in):
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        return np.zeros_like(x)


    def lhs(self, x, t, u_t, u_x, u_xx):
        # Fokker-Planck operator: u_t + [mu(t,x) u(t,x)]_x - [D(t,x) u(t,x)]_xx
        # Our case: u_t + mu(t,x) u_x(t,x) - sigma_SDE^2/2  u_xx(t,x)
        mu_eval = self.mu_SDE(t, x)
        D = self.sigma_SDE**2 / 2
        return u_t + mu_eval*u_x - D*u_xx

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        t = x_in[:, [0]]
        x = x_in[:, [1]]

        # get auto grad
        t_t = torch.from_numpy(t)
        x_t = torch.from_numpy(x)
        t_t.requires_grad_()
        x_t.requires_grad_()
        x_in_t = torch.cat([t_t, x_t], dim=1)
        u = self.u_exact(x_in_t)['u']

        u_t, u_tt = self.get_grad_auto(u=u, x=t_t)
        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)

        # eval
        u_xx = u_xx.detach().numpy()
        u_x = u_x.detach().numpy()
        u_t = u_t.detach().numpy()
        lhs_eval = self.lhs(x, t, u_t, u_x, u_xx)
        rhs_eval = self.rhs(x_in)
        rmse = np.sqrt(np.mean((lhs_eval-rhs_eval)**2))
        print(f'pde rmse: {rmse}')
        return x_in, u.detach().numpy()


    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # Fokker-Planck operator: u_t + [mu(t,x) u(t,x)]_x - [D(t,x) u(t,x)]_xx
        # Our case: u_t + mu(t,x) u_x(t,x) - sigma_SDE^2/2  u_xx(t,x)
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)
        t = x_in[:, [0]]
        x = x_in[:, [1]]

        mu_eval = self.mu_SDE(t, x)
        D = self.sigma_SDE ** 2 / 2

        feature_pde = basis_eval['u0'] + mu_eval*basis_eval['u1'] - D*basis_eval['u11']
        target_pde = self.rhs(x_in)
        return [feature_pde, target_pde]

    def ls_feature_value(self,basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_bd = basis_eval['u']
        feature_bd_all = {'bd_u': [feature_bd, self.target_bd['u']]}
        return feature_bd_all


class NS_steady_2d(Problem):
    def __init__(self,case=None, data=None):
        super(NS_steady_2d, self).__init__(case, data)
        # assert case in [1,2,3], f'case {case} not implemented'
        self.pde_name = 'NS_steady'
        self.eval_list_pde = ['u', 'u0', 'u1', 'u00', 'u11']
        self.eq_names = ['div', 'pde_u', 'pde_v']
        self.out_var = ['u', 'v', 'p']

        self.re = 40

    def set_re(self, re):
        self.re = re

    def u_exact(self, x_in):
        x = x_in[:, [0]]
        y = x_in[:, [1]]
        nu = 1.0/self.re
        lamb = 1.0/(2.0*nu) - np.sqrt(1.0/(4.0*nu**2) + 4*np.pi**2)
        if type(x) == torch.Tensor:
            u = 1 - torch.exp(lamb*x)*torch.cos(2*np.pi*y)
            v = lamb/(2*np.pi) * torch.exp(lamb*x) * torch.sin(2*np.pi*y)
            p = 0.5*(1 - torch.exp(2*lamb*x))
            return {'u': u, 'v': v, 'p': p}
        else:
            u = 1 - np.exp(lamb*x)*np.cos(2*np.pi*y)
            v = lamb/(2*np.pi) * np.exp(lamb*x) * np.sin(2*np.pi*y)
            p = 0.5*(1 - np.exp(2*lamb*x))
            return {'u': u, 'v': v, 'p': p}

    def rhs(self, x_in):
        x = x_in[:, [0]]
        y = x_in[:, [1]]
        return {'div':np.zeros_like(x), 'pde_u':np.zeros_like(x), 'pde_v':np.zeros_like(x)}

    def lhs(self, u,v,u_x,u_y,v_x,v_y,p_x,p_y,u_xx,u_yy,v_xx,v_yy):
        re = self.re

        div = u_x + v_y
        pde_u = u*u_x + v*u_y + p_x  - (u_xx + u_yy)/re
        pde_v = u*v_x + v*v_y + p_y  - (v_xx + v_yy)/re
        return {'div':div, 'pde_u':pde_u, 'pde_v':pde_v }

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        x = x_in[:, [0]]
        y = x_in[:, [1]]

        # get auto grad
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        x_t.requires_grad_()
        y_t.requires_grad_()
        x_in_t = torch.cat([x_t, y_t], dim=1)
        var_all = self.u_exact(x_in_t)
        u = var_all['u']
        v = var_all['v']
        p = var_all['p']

        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)
        u_y, u_yy = self.get_grad_auto(u=u, x=y_t)
        v_x, v_xx = self.get_grad_auto(u=v, x=x_t)
        v_y, v_yy = self.get_grad_auto(u=v, x=y_t)
        p_x, p_xx = self.get_grad_auto(u=p, x=x_t)
        p_y, p_yy = self.get_grad_auto(u=p, x=y_t)

        # eval
        u = u.detach().numpy()
        v = v.detach().numpy()
        p = p.detach().numpy()

        u_x = u_x.detach().numpy()
        u_y = u_y.detach().numpy()
        u_xx = u_xx.detach().numpy()
        u_yy = u_yy.detach().numpy()

        p_x = p_x.detach().numpy()
        p_y = p_y.detach().numpy()

        v_x = v_x.detach().numpy()
        v_y = v_y.detach().numpy()
        v_xx = v_xx.detach().numpy()
        v_yy = v_yy.detach().numpy()

        lhs_eval = self.lhs(u,v,u_x,u_y,v_x,v_y,p_x,p_y,u_xx,u_yy,v_xx,v_yy)
        rhs_eval = self.rhs(x_in)
        for eq_item in self.eq_names:
            lhs_item = lhs_eval[eq_item]
            rhs_item = rhs_eval[eq_item]
            rmse = np.sqrt(np.mean((lhs_item-rhs_item)**2))
            print(f'{eq_item} rmse:\t {rmse}')
        return x_in, {'u': u, 'v': v, 'p': p}

    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        re = self.re
        nu = 1.0/re
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)

        z_x = basis_eval['u0']
        z_xx = basis_eval['u00']
        z_y = basis_eval['u1']
        z_yy = basis_eval['u11']

        if current_sol is None:
            u_eval = np.zeros((z_x.shape[0],1))
            v_eval = np.zeros((z_x.shape[0],1))
        else:
            u_eval = current_sol['u']
            v_eval = current_sol['v']

        if feature_name == 'div':
            feature_div = np.concatenate([z_x, z_y, np.zeros_like(z_x)], axis=1)
            target_div = np.zeros((feature_div.shape[0],1))
            return [feature_div, target_div]

        if feature_name == 'pde_u':
            feature_pde_u = np.concatenate([u_eval*z_x + v_eval*z_y -nu*z_xx - nu*z_yy, np.zeros_like(z_x), z_x], axis=1)
            target_pde_u = np.zeros((feature_pde_u.shape[0],1))
            return [feature_pde_u, target_pde_u]

        if feature_name == 'pde_v':
            feature_pde_v = np.concatenate([np.zeros_like(z_x), u_eval*z_x + v_eval*z_y- nu*z_xx - nu*z_yy, z_y], axis=1)
            target_pde_v = np.zeros((feature_pde_v.shape[0],1))
            return [feature_pde_v, target_pde_v]

    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        basis_bd = basis_eval['u']

        # boundary feature
        feature_bd_u = np.concatenate([basis_bd, np.zeros_like(basis_bd), np.zeros_like(basis_bd)], axis=1)
        feature_bd_v = np.concatenate([np.zeros_like(basis_bd), basis_bd, np.zeros_like(basis_bd)], axis=1)
        feature_bd_p = np.concatenate([np.zeros_like(basis_bd), np.zeros_like(basis_bd), basis_bd], axis=1)

        # boundary target
        target_bd_u = self.target_bd['u']
        target_bd_v = self.target_bd['v']
        target_bd_p = self.target_bd['p']

        feature_bd_all = {'u': [feature_bd_u, target_bd_u],
                          'v': [feature_bd_v, target_bd_v],
                          'p': [feature_bd_p, target_bd_p]}
        return feature_bd_all



class Burgers_1d(Problem):
    def __init__(self, domain, case=1):
        super(Burgers_1d, self).__init__(domain, case)
        assert domain.domain_dim == 1, 'Domain dim should be 2 for this problem.'
        assert case in [1], f'case {case} not implemented'

        self.pde_name = 'Burgers_1d'
        self.eval_list_pde = ['u', 'u0', 'u00']
        self.operator_type = 'nlin'
        self.eq_names = ['pde']
        self.out_var = ['u']


        self.nu = 0.1
        self.A = 1
        self.z = 0.25


    def set_problem_param(self, A, nu, z):
        self.A = A
        self.nu = nu
        self.z = z


    def u_exact(self, x_in):
        # x_in: (N, 1)
        A = self.A
        nu = self.nu
        z = self.z
        if self.case == 1:
            if type(x_in) == torch.Tensor:
                u = -A*torch.tanh(A/2/nu*(x_in - z))
                return {'u': u}
            else:
                u = -A*np.tanh(A/2/nu*(x_in - z))
                return {'u': u}

    def rhs(self, x_in):
        # x_in: (N, 1)
        return {'pde':np.zeros_like(x_in)}

    def lhs(self, u,u_x,u_xx):
        pde = u*u_x - self.nu*u_xx
        return {'pde':pde}

    def check_solution(self):
        x_in = self.x_train

        x = x_in

        # get auto grad
        x_t = torch.from_numpy(x)
        x_t.requires_grad_()
        var_all = self.u_exact(x_t)
        u = var_all['u']

        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)


        # eval
        u = u.detach().numpy()

        u_x = u_x.detach().numpy()
        u_xx = u_xx.detach().numpy()


        lhs_eval = self.lhs(u,u_x,u_xx)
        rhs_eval = self.rhs(x_in)
        for eq_item in self.eq_names:
            lhs_item = lhs_eval[eq_item]
            rhs_item = rhs_eval[eq_item]
            rmse = np.sqrt(np.mean((lhs_item-rhs_item)**2))
            print(f'{eq_item} rmse:\t {rmse}')
        return [x_in, u]

    def ls_feature_pde(self, feature_name, x_in, basis_eval, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        nu = self.nu
        z_x = basis_eval['u0']
        z_xx = basis_eval['u00']

        if current_sol is None:
            u_eval = np.zeros((z_x.shape[0],1))
        else:
            u_eval = current_sol['u']

        if feature_name == 'pde':
            feature_pde = u_eval*z_x - nu*z_xx
            target_pde = np.zeros((feature_pde.shape[0],1))
            return [feature_pde, target_pde]

    def ls_feature_bd(self, x_in,  basis_eval):
        # assert value_name in self.out_var, 'Invalid value name.'
        target_eval = self.u_exact(x_in)
        basis_bd = basis_eval['u']

        # boundary feature
        feature_bd_u = basis_bd

        # boundary target
        target_bd_u = target_eval['u']

        feature_bd_all = {'u': [feature_bd_u, target_bd_u]}
        return feature_bd_all


class Poisson_3d(Problem):
    def __init__(self, case=None, data=None):
        super(Poisson_3d, self).__init__(case, data)
        # assert domain.domain_dim == 3, 'Domain dim should be 3 for this problem.'

        self.pde_name = 'Poisson'
        self.eval_list_pde = ['u', 'u00', 'u11', 'u22']
        self.eq_names = ['pde']
        self.out_var = 'u'

    def u_exact(self, x_in):
        x = x_in[:, [0]]
        y = x_in[:, [1]]
        z = x_in[:, [2]]

        if self.case == 1:
            if type(x) == torch.Tensor:
                return {'u': torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * z)}
            else:
                return {'u': np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)}

        if self.case == 2:
            if type(x) == torch.Tensor:
                return {'u': torch.sin(2*np.pi * x) * torch.sin(2*np.pi * y) * torch.sin(2*np.pi * z)}
            else:
                return {'u': np.sin(2*np.pi * x) * np.sin(2*np.pi * y) * np.sin(2*np.pi * z)}

        if self.case == 3:
            if type(x) == torch.Tensor:
                return {'u': torch.sin(np.pi * x/2) * torch.sin(np.pi * y/2) * torch.sin(np.pi * z/2)}
            else:
                return {'u': np.sin(np.pi * x/2) * np.sin(np.pi * y/2) * np.sin(np.pi * z/2)}

    def rhs(self, x_in):
        x = x_in[:, [0]]
        y = x_in[:, [1]]
        z = x_in[:, [2]]
        if self.case == 1:
            return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) * (-3)*np.pi**2

        if self.case == 2:
            return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.sin(2*np.pi*z) * (-12)*np.pi**2

        if self.case == 3:
            return np.sin(np.pi*x/2)*np.sin(np.pi*y/2)*np.sin(np.pi*z/2) * (-3.0/4.0)*np.pi**2

    def lhs(self, u_xx, u_yy, u_zz):
        # Poisson operator
        return u_xx + u_yy + u_zz

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        x = x_in[:, [0]]
        y = x_in[:, [1]]
        z = x_in[:, [2]]

        # get auto grad
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        z_t = torch.from_numpy(z)
        x_t.requires_grad_()
        y_t.requires_grad_()
        z_t.requires_grad_()

        x_in_t = torch.cat([x_t, y_t, z_t], dim=1)
        u = self.u_exact(x_in_t)['u']
        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)
        u_y, u_yy = self.get_grad_auto(u=u, x=y_t)
        u_z, u_zz = self.get_grad_auto(u=u, x=z_t)


        # eval
        u_xx = u_xx.detach().numpy()
        u_yy = u_yy.detach().numpy()
        u_zz = u_zz.detach().numpy()
        lhs_eval = self.lhs(u_xx, u_yy, u_zz)
        rhs_eval = self.rhs(x_in)
        rmse = np.sqrt(np.mean((lhs_eval-rhs_eval)**2))
        print(f'pde rmse: {rmse}')
        return x_in, u.detach().numpy()

    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)

        feature_pde = basis_eval['u00'] + basis_eval['u11'] + basis_eval['u22']
        target_pde = self.rhs(x_in)
        return [feature_pde, target_pde]

    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_bd = basis_eval['u']
        feature_bd_all = {'bd_u': [feature_bd, self.target_bd['u']]}
        return feature_bd_all


class Fokker_Planck_2d(Problem):
    def __init__(self, case=None, data=None):
        super(Fokker_Planck_2d, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Fokker_Planck'
        self.eval_list_pde = ['u', 'u0', 'u1', 'u2', 'u11', 'u22']
        self.eq_names = ['pde']
        self.out_var = 'u'

        # model param
        self.sigma0 = 0.4
        self.sigma_SDE = 0.3

        # drift param
        # self.mu_freq = 3
        # self.mu_mag = 2

    def mu_sol(self, t):
        if type(t) == torch.Tensor:
            mu1 = -(torch.cos(2 * np.pi * t) - 1) / 2 / np.pi
            mu2 = torch.sin(2 * np.pi * t) / 2 / np.pi
        else:
            mu1 = -(np.cos(2 * np.pi * t) - 1) / 2 / np.pi
            mu2 = np.sin(2 * np.pi * t) / 2 / np.pi
        return mu1, mu2

    def mu_SDE(self, t):
        if type(t) == torch.Tensor:
            mu1 = torch.sin(2 * np.pi * t)
            mu2 = torch.cos(2 * np.pi * t)
        else:
            mu1 = np.sin(2 * np.pi * t)
            mu2 = np.cos(2 * np.pi * t)
        return mu1, mu2

    def sigma_sol(self, t):
        sigma0 = self.sigma0
        sigma_SDE = self.sigma_SDE
        if type(t) == torch.Tensor:
            sigma_t = torch.sqrt(sigma0 ** 2 + t * sigma_SDE ** 2)
        else:
            sigma_t = np.sqrt(sigma0 ** 2 + t * sigma_SDE ** 2)
        return sigma_t


    @staticmethod
    def gaussian_1d(x, mu, sigma):
        if type(x) == torch.Tensor:
            p = torch.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))
        else:
            p = np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))
        return p

    def p_true(self, t, x, y):
        mu1, mu2 = self.mu_sol(t)
        sigma_true = self.sigma_sol(t)
        p1 = self.gaussian_1d(x, mu1, sigma_true)
        p2 = self.gaussian_1d(y, mu2, sigma_true)
        p = p1*p2
        return p

    def u_exact(self, x_in):
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        y = x_in[:, [2]]
        mu1, mu2 = self.mu_sol(t)
        sigma_true = self.sigma_sol(t)
        p1 = self.gaussian_1d(x, mu1, sigma_true)
        p2 = self.gaussian_1d(y, mu2, sigma_true)
        p = p1 * p2
        return{'u': p}

    def rhs(self, x_in):
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        return np.zeros_like(x)

    def lhs(self, x, t, u_t, u_x, u_y, u_xx, u_yy):
        # Fokker-Planck operator: u_t + [mu(t,x) u(t,x)]_x - [D(t,x) u(t,x)]_xx
        # Our case: u_t + mu(t,x) u_x(t,x) - sigma_SDE^2/2  u_xx(t,x)
        mu1, mu2 = self.mu_SDE(t)
        D = self.sigma_SDE**2 / 2
        return u_t + mu1*u_x + mu2*u_y - D*(u_xx+u_yy)

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        t = x_in[:, [0]]
        x = x_in[:, [1]]
        y = x_in[:, [2]]

        # get auto grad
        t_t = torch.from_numpy(t)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        t_t.requires_grad_()
        x_t.requires_grad_()
        y_t.requires_grad_()

        x_in_t = torch.cat([t_t, x_t, y_t], dim=1)
        u = self.u_exact(x_in_t)['u']

        u_t, u_tt = self.get_grad_auto(u=u, x=t_t)
        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)
        u_y, u_yy = self.get_grad_auto(u=u, x=y_t)

        # eval
        u_t = u_t.detach().numpy()
        u_xx = u_xx.detach().numpy()
        u_x = u_x.detach().numpy()

        u_yy = u_yy.detach().numpy()
        u_y = u_y.detach().numpy()

        lhs_eval = self.lhs(x, t, u_t, u_x, u_y, u_xx, u_yy)
        rhs_eval = self.rhs(x_in)
        rmse = np.sqrt(np.mean((lhs_eval-rhs_eval)**2))
        print(f'pde rmse: {rmse}')
        return x_in, u.detach().numpy()


    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # Fokker-Planck operator: u_t + [mu(t,x) u(t,x)]_x - [D(t,x) u(t,x)]_xx
        # Our case: u_t + mu(t,x) u_x(t,x) - sigma_SDE^2/2  u_xx(t,x)
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        y = x_in[:, [2]]

        mu1, mu2 = self.mu_SDE(t)
        D = self.sigma_SDE ** 2 / 2

        feature_pde = basis_eval['u0'] + mu1*basis_eval['u1'] + mu2*basis_eval['u2']\
                      - D*(basis_eval['u11'] + basis_eval['u22'])
        target_pde = self.rhs(x_in)
        return [feature_pde, target_pde]

    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_bd = basis_eval['u']
        feature_bd_all = {'bd_u': [feature_bd, self.target_bd['u']]}
        return feature_bd_all


class Allen_Cahn_2d(Problem):
    def __init__(self, case=None, data=None):
        super(Allen_Cahn_2d, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Allen_Cahn'
        self.eval_list_pde = ['u', 'u0', 'u11', 'u22']
        self.eq_names = ['pde']
        self.out_var = 'u'

        # model param
        self.epsilon = 0.05
        self.R0 = 0.4


    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # u_t - u_xx - u_yy + (u^3 - u)/eps^2 = 0
        # u_t - u_xx - u_yy - u/eps^2 = -u^3/eps^2 (ver 1)
        # u_t - u_xx - u_yy = (u-u^3)/eps^2 (ver 2)
        # u_t - u_xx - u_yy - u(1-u^2)/eps^2 = 0 (ver 3)

        epsilon = self.epsilon
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)


        if current_sol is None:
            u_eval = np.zeros((x_in.shape[0], 1))
        else:
            u_eval = current_sol['u']

        # feature_pde = basis_eval['u0'] - basis_eval['u11'] - basis_eval['u22'] - basis_eval['u']/epsilon**2
        # target_pde = -u_eval**3/epsilon**2


        # feature_pde = basis_eval['u0'] - basis_eval['u11'] - basis_eval['u22']
        # target_pde = (u_eval-u_eval ** 3) / epsilon ** 2

        feature_pde = basis_eval['u0'] - basis_eval['u11'] - basis_eval['u22'] - basis_eval['u']*(1-u_eval**2)/epsilon**2
        target_pde = np.zeros_like(basis_eval['u'][:, [0]])

        # target_pde[target_pde > 1] = 1
        # target_pde[target_pde < -1] = -1

        return [feature_pde, target_pde]

    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_all = {}

        x_in = self.x_ic
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_ic = basis_eval['u']
        feature_all['ic_u'] = [feature_ic, self.target_ic]

        x_bd_all = self.x_bd.item()
        x_bd_l = x_bd_all['x_bd_l']
        x_bd_r = x_bd_all['x_bd_r']
        feature_bd_lr = basis.eval_basis(x_bd_l)['u'] - basis.eval_basis(x_bd_r)['u']
        target_bd = np.zeros_like(x_bd_l[:, [0]])
        feature_all['u_bd_lr'] = [feature_bd_lr, target_bd]

        x_bd_u = x_bd_all['x_bd_u']
        x_bd_d = x_bd_all['x_bd_d']
        feature_bd_ud = basis.eval_basis(x_bd_u)['u'] - basis.eval_basis(x_bd_d)['u']
        target_bd = np.zeros_like(x_bd_u[:, [0]])
        feature_all['u_bd_ud'] = [feature_bd_ud, target_bd]


        return feature_all


class Allen_Cahn_1d(Problem):
    def __init__(self, case=None, data=None):
        super(Allen_Cahn_1d, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Allen_Cahn'
        self.eval_list_pde = ['u', 'u0', 'u11']
        self.eq_names = ['pde']
        self.out_var = 'u'

        # model param
        self.epsilon = 0.05


    def u_exact(self, x_in):
        epsilon = self.epsilon
        s = 3.0/np.sqrt(2)/epsilon

        t = x_in[:, [0]]
        x = x_in[:, [1]]
        if type(x) == torch.Tensor:
            u = 0.5*(1-torch.tanh((x - s*t)/(2*np.sqrt(2)*epsilon)))
        else:
            u = 0.5*(1-np.tanh((x - s*t)/(2*np.sqrt(2)*epsilon)))
        return{'u': u}

    def rhs(self, x_in):
        t = x_in[:, [0]]
        return np.zeros_like(t)


    def lhs(self, u, u_t, u_xx):
        epsilon = self.epsilon
        op = u_t - u_xx + (u**3 - u)/epsilon**2
        return op

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        t = x_in[:, [0]]
        x = x_in[:, [1]]

        # get auto grad
        t_t = torch.from_numpy(t)
        x_t = torch.from_numpy(x)
        t_t.requires_grad_()
        x_t.requires_grad_()
        x_in_t = torch.cat([t_t, x_t], dim=1)
        u = self.u_exact(x_in_t)['u']

        u_t, u_tt = self.get_grad_auto(u=u, x=t_t)
        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)

        # eval
        u = u.detach().numpy()
        u_xx = u_xx.detach().numpy()
        u_t = u_t.detach().numpy()
        lhs_eval = self.lhs(u, u_t, u_xx)
        rhs_eval = self.rhs(x_in)
        rmse = np.sqrt(np.mean((lhs_eval-rhs_eval)**2))
        print(f'pde rmse: {rmse}')
        return x_in, u


    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # u_t - u_xx  + (u^3 - u)/eps^2 = 0
        # u_t - u_xx  - u/eps^2 = -u^3/eps^2 (ver 1)
        # u_t - u_xx  = (u-u^3)/eps^2 (ver 2)
        # u_t - u_xx  - u(1-u^2)/eps^2 = 0 (ver 3)

        epsilon = self.epsilon
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)


        if current_sol is None:
            u_eval = np.zeros((x_in.shape[0], 1))
        else:
            u_eval = current_sol['u']

        # feature_pde = basis_eval['u0'] - basis_eval['u11'] - basis_eval['u22'] - basis_eval['u']/epsilon**2
        # target_pde = -u_eval**3/epsilon**2

        feature_pde = basis_eval['u0'] - basis_eval['u11']
        target_pde = (u_eval-u_eval ** 3) / epsilon ** 2

        # feature_pde = basis_eval['u0'] - basis_eval['u11'] - basis_eval['u']*(1-u_eval**2)/epsilon**2
        # target_pde = np.zeros_like(basis_eval['u'][:, [0]])

        # target_pde[target_pde > 1] = 1
        # target_pde[target_pde < -1] = -1

        return [feature_pde, target_pde]


    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_bd = basis_eval['u']
        feature_bd_all = {'bd_u': [feature_bd, self.target_bd['u']]}
        return feature_bd_all


class Wave_1d(Problem):
    def __init__(self, case=None, data=None):
        super(Wave_1d, self).__init__(case, data)
        # assert case in [1], f'case {case} not implemented'
        # var= [y, x1, x2]
        self.pde_name = 'Wave_1d'
        self.eval_list_pde = ['u', 'u00', 'u11']
        self.eq_names = ['pde']
        self.out_var = 'u'

        # model param
        self.c = 1/(4*np.pi)**2


    def u_exact(self, x_in):
        t = x_in[:, [0]]
        x = x_in[:, [1]]
        if type(x) == torch.Tensor:
            u = 0.5*(torch.sin(4*np.pi*x + t) + torch.sin(4*np.pi*x - t))
        else:
            u = 0.5*(np.sin(4*np.pi*x + t) + np.sin(4*np.pi*x - t))
        return{'u': u}

    def rhs(self, x_in):
        t = x_in[:, [0]]
        return np.zeros_like(t)


    def lhs(self,u_tt, u_xx):
        c = self.c
        op = u_tt - c*u_xx
        return op

    def check_solution(self, x_in=None):
        if x_in is None:
            x_in = self.x_pde

        t = x_in[:, [0]]
        x = x_in[:, [1]]

        # get auto grad
        t_t = torch.from_numpy(t)
        x_t = torch.from_numpy(x)
        t_t.requires_grad_()
        x_t.requires_grad_()
        x_in_t = torch.cat([t_t, x_t], dim=1)
        u = self.u_exact(x_in_t)['u']

        u_t, u_tt = self.get_grad_auto(u=u, x=t_t)
        u_x, u_xx = self.get_grad_auto(u=u, x=x_t)

        # eval
        u = u.detach().numpy()
        u_xx = u_xx.detach().numpy()
        u_tt = u_tt.detach().numpy()

        lhs_eval = self.lhs(u_tt, u_xx)
        rhs_eval = self.rhs(x_in)
        rmse = np.sqrt(np.mean((lhs_eval-rhs_eval)**2))
        print(f'pde rmse: {rmse}')
        return x_in, u


    def ls_feature_pde(self, feature_name, basis, current_sol=None):
        assert feature_name in self.eq_names, 'Invalid pde feature name.'
        # u_tt - c u_xx   = 0

        c = self.c
        x_in = self.x_pde
        basis_eval = basis.eval_basis(x_in, eval_list=self.eval_list_pde)


        feature_pde = basis_eval['u00'] - c*basis_eval['u11']
        target_pde = np.zeros_like(basis_eval['u'][:, [0]])

        return [feature_pde, target_pde]


    def ls_feature_value(self, basis):
        # assert value_name in self.out_var, 'Invalid value name.'
        feature_bd_all = {}
        # bd
        x_in = self.x_bd
        basis_eval = basis.eval_basis(x_in, eval_list=['u'])
        feature_bd = basis_eval['u']
        feature_bd_all['bd_u'] = [feature_bd, self.target_bd['u']]

        # ic
        x_in = self.x_ic
        basis_eval = basis.eval_basis(x_in, eval_list=['u', 'u0'])
        feature_ic = basis_eval['u']
        feature_bd_all['ic_u'] = [feature_ic, self.target_ic['u']]


        feature_ic0 = basis_eval['u0']
        feature_bd_all['ic_u0'] = [feature_ic0, self.target_ic['u0']]

        return feature_bd_all
