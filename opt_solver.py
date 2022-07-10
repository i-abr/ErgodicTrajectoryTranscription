import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as np


#     avg_sq_grad = jnp.zeros_like(x0)
#     return x0, avg_sq_grad
#   def update(i, g, state):
#     x, avg_sq_grad = state
#     avg_sq_grad = avg_sq_grad * gamma + jnp.square(g) * (1. - gamma)

class AugmentedLagrangian(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args=None, step_size=1e-3, c=1.0):
        self.def_args = args
        self.loss = loss 
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self._x_shape = x0.shape
        self.solution = {'x' : x0, 'lam' : lam, 'mu' : mu}
        self.avg_sq_grad = np.zeros_like(x0)
        # self._flat_solution, self._unravel = ravel_pytree(self.solution)
        def lagrangian(solution, args):
            # solution = self._unravel(flat_solution)
            x   = solution['x']
            lam = solution['lam']
            mu  = solution['mu']
            _eq_constr   = eq_constr(x, args)
            _ineq_constr = ineq_constr(x, args)
            return loss(x, args) \
                + np.sum(lam * _eq_constr + 0.5 * (_eq_constr)**2) \
                + 0.5 * np.sum(np.maximum(0., mu + _ineq_constr)**2 - mu**2)

        dldx = jit(grad(lagrangian))
        gamma=0.9
        eps=1e-8
        @jit
        def step(solution, args, avg_sq_grad):
            _dldx   = dldx(solution, args)
            _eps    = np.linalg.norm(_dldx['x'])
            avg_sq_grad = avg_sq_grad * gamma + np.square(_dldx['x']) * (1. - gamma)
            solution['x']   = solution['x'] - step_size * _dldx['x'] / np.sqrt(avg_sq_grad + eps)
            solution['lam'] = solution['lam'] + c*eq_constr(solution['x'], args)
            solution['mu']  = np.maximum(0, solution['mu'] + c*ineq_constr(solution['x'], args))
            return solution, _eps, avg_sq_grad

        self.lagrangian      = lagrangian
        self.grad_lagrangian = dldx
        self.step = step

    def get_solution(self):
        return self.solution
        # return self._unravel(self._flat_solution)

    def solve(self, args=None, max_iter=10000, eps=1e-3):
        if args is None:
            args = self.def_args
        for k in range(max_iter):
            self.solution, _eps, self.avg_sq_grad = self.step(self.solution, args, self.avg_sq_grad)
            if _eps < eps:
                print('done in ', k, ' iterations')
                break


if __name__=='__main__':
    def f(x, args=None) : return 13*x[0]**2 + 10*x[0]*x[1] + 7*x[1]**2 + x[0] + x[1]
    def g(x, args) : return np.array([2*x[0]-5*x[1]-2])
    def h(x, args) : return x[0] + x[1] -1

    x0 = np.array([.5,-0.3])
    opt = AugmentedLagrangian(x0,f,g,h, step_size=0.1)
    opt.solve(max_iter=1000)
    sol = opt.get_solution()
    print(f(sol['x']), sol['x'])