import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as np

class AugmentedLagrangian(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args=None, step_size=1e-3, c=1.0):
        self.loss = loss 
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self._x_shape = x0.shape
        self.solution = {'x' : x0, 'lam' : lam, 'mu' : mu}
        self._flat_solution, self._unravel = ravel_pytree(self.solution)
        def lagrangian(flat_solution, args):
            solution = self._unravel(flat_solution)
            x   = solution['x']
            lam = solution['lam']
            mu  = solution['mu']
            _eq_constr   = eq_constr(x, args)
            _ineq_constr = ineq_constr(x, args)
            return loss(x, args) \
                - np.sum(lam * _eq_constr) \
                + np.sum(0.5*(_eq_constr)**2) \
                + 0.5 * np.sum(np.maximum(0., mu + _ineq_constr)**2) \
                + np.sum(mu**2)
        dldx = jit(grad(lagrangian))
        dl2dx2 = jacfwd(dldx)
        @jit
        def step(flat_solution, args):
            _dldx   = dldx(flat_solution, args)
            # _dl2dx2 = dl2dx2(flat_solution, args)
            flat_solution = flat_solution - step_size * _dldx
            _eps = np.linalg.norm(_dldx)
            # solution['x']   = solution['x'] - step_size * _dldx['x']
            # solution['lam'] = solution['lam'] + c*eq_constr(solution['x'], args)
            # solution['mu']  = np.maximum(0, solution['mu'] + c*ineq_constr(solution['x'], args))
            # theta['lam'] = theta['lam'] + step_size * _dldth['lam']
            # theta['mu']  = theta['mu'] + step_size * _dldth['mu']
            return flat_solution, _eps
        self.lagrangian = lagrangian
        self.grad_lagrangian = dldx
        self.step = step
    def get_solution(self):
        return self._unravel(self._flat_solution)
    def solve(self, args=None, max_iter=10000, eps=1e-3):
        for k in range(max_iter):
            self._flat_solution, _eps = self.step(self._flat_solution, args)
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