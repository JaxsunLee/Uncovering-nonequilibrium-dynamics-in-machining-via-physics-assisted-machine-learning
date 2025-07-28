import numpy as np
from gpytorch import constraints
from gpytorch.kernels import Kernel
import torch
import matplotlib.pyplot as plt
import os

plt.style.use(['science', 'nature'])
plt.rcParams['text.usetex'] = True
os.environ["TORCH_NNPACK"] = "0"


class WsKernel(Kernel):

    is_stationary = True

    def __init__(self, sigma_f_prior=None, l_prior=None, l_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # Sign up parameters
        self.register_parameter(
            name='raw_sigma_f', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name='raw_l', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # Constraint
        if l_constraint is None:
            l_constraint = constraints.Positive()

        # Sign up constraints
        self.register_constraint("raw_l", l_constraint)

        # Set parameters prior
        if sigma_f_prior is not None:
            self.register_prior(
                "sigma_f_prior",
                sigma_f_prior,
                lambda m: m.sigma_f,
                lambda m, v: m._set_sigma_f(v),
            )
        
        if l_prior is not None:
            self.register_prior(
                "l_prior",
                l_prior,
                lambda m: m.length,
                lambda m, v: m._set_l(v),
            )
        
    # Set actual parameters
    @property
    def sigma_f(self):
        return self.raw_sigma_f
    
    @property
    def l(self):
        return self.raw_l_constraint.transform(self.raw_l)

    @sigma_f.setter
    def sigma_f(self, value):
        return self._set_sigma_f(value)
    
    @l.setter
    def l(self, value):
        return self._set_l(value)

    def _set_sigma_f(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_f)
        self.initialize(raw_sigma_f=value)

    def _set_l(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_l)
        self.initialize(raw_l=self.raw_l_constraint.inverse_transform(value))

    @staticmethod
    def _wasserstein_distance(x1, x2):
        diff = (x1 - x2).pow(2).sum(dim=-1)
        return diff.sqrt()

    def forward(self, x1, x2, diag=False, **params):

        x1_ = x1.unsqueeze(-2) if x1.dim() == 1 else x1
        x2_ = x2.unsqueeze(-2) if x2.dim() == 1 else x2
        if diag:
            wsd = self._wasserstein_distance(x1, x2)
            return self.sigma_f**2 * torch.exp(-0.5 * (wsd / self.l) ** 2)
        
        x1_exp = x1_.unsqueeze(-2)
        x2_exp = x2_.unsqueeze(-3)

        wsd = self._wasserstein_distance(x1_exp, x2_exp)
        k = self.sigma_f**2 * torch.exp(-0.5 * (wsd / self.l) ** 2)
        return k
    
class EKernel(Kernel):

    is_stationary = True

    def __init__(self, sigma_p_prior=None, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(
            name='raw_sigma_p', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if sigma_p_prior is not None:
            self.register_prior(
                "sigma_p_prior",
                sigma_p_prior,
                lambda m: m.sigma_p,
                lambda m, v: m._set_sigma_p(v),
            )
        
    @property
    def sigma_p(self):
        return self.raw_sigma_p

    @sigma_p.setter
    def sigma_p(self, value):
        return self._set_sigma_p(value)

    def _set_sigma_p(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_p)
        self.initialize(raw_sigma_p=value)

    def forward(self, x1, x2):
        distance = self.covar_dist(
        x1=x1, 
        x2=x2, 
        diag=False, 
        last_dim_is_batch=False, 
        square_dist=False
    )
        k = self.sigma_p**2 * torch.exp(distance)
        
        return k
    
class ObKernel(Kernel):

    is_stationary = True

    def __init__(self, gamma_prior=None, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(
            name='raw_gamma', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if gamma_prior is not None:
            self.register_prior(
                "gamma_prior",
                gamma_prior,
                lambda m: m.gamma,
                lambda m, v: m._set_gamma(v),
            )
        

    @property
    def gamma(self):
        return self.raw_gamma

    @gamma.setter
    def gamma(self, value):
        return self._set_gamma(value)

    def _set_gamma(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_gamma=value)

    def forward(self, x1, x2, diag=False, **params):

        x1_ = x1.unsqueeze(-2) if x1.dim() == 1 else x1
        x2_ = x2.unsqueeze(-2) if x2.dim() == 1 else x2

        if diag:
            diff = x1 - x2
            diff = diff.pow(2).sum(dim=-1)
            return torch.exp(-self.gamma * diff)

        x1_exp = x1_.unsqueeze(-2)
        x2_exp = x2_.unsqueeze(-3)
        k = x1_exp.T * torch.tensor([[1 + self.gamma], [-self.gamma]]) * x2_exp
        
        return k

