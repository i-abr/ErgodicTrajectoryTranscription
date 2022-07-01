import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
import jax.numpy as np

class AugmentedLagrangian(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args, step_size=1e-3, c=1.0):
        self.loss = loss 
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self._x_shape = x0.shape
        self.solution = {'x' : x0, 'lam' : lam, 'mu' : mu}
        def lagrangian(solution, args):
            x   = solution['x']
            lam = solution['lam']
            mu  = solution['mu']
            _eq_constr   = eq_constr(x, args)
            _ineq_constr = ineq_constr(x, args)
            return loss(x, args) \
                + np.sum(lam * _eq_constr + 0.5 * (_eq_constr)**2) \
                    + 0.5 * np.sum(np.maximum(0., mu + _ineq_constr)**2 - mu**2)
            # return loss(x) + np.sum(lam * _eq_constr + 0.5 * (_eq_constr)**2) + 0.5 * np.sum(mu * np.maximum(0., mu + _ineq_constr))

        dldth = jit(grad(lagrangian))
        # @jit
        def step(solution, args):
            _dldx = dldth(solution, args)
            # _d2dx2 = d2dth2(solution, args)
            # _d2dx2 = np.outer(_dldx, _dldx)
            # _eps = np.linalg.norm(_dldx['x'])
            _eps=0.3
            solution['x']   = solution['x'] - step_size * _dldx['x']
            solution['lam'] = solution['lam'] + c*eq_constr(solution['x'], args)
            solution['mu']  = np.maximum(0, solution['mu'] + c*ineq_constr(solution['x'], args))
            # theta['lam'] = theta['lam'] + step_size * _dldth['lam']
            # theta['mu']  = theta['mu'] + step_size * _dldth['mu']
            return solution, _eps
        self.lagrangian = lagrangian
        self.grad_lagrangian = dldth
        self.step = step
    def solve(self, args, max_iter=10000, eps=1e-3):
        for k in range(max_iter):
            self.solution, _eps = self.step(self.solution, args)
            if _eps < eps:
                print('done in ', k, ' iterations')
                break


if __name__=='__main__':
    def f(x) : return 13*x[0]**2 + 10*x[0]*x[1] + 7*x[1]**2 + x[0] + x[1]
    def g(x) : return np.array([2*x[0]-5*x[1]-2, x[0] + x[1] -1])
    def h(x) : return np.zeros(1)#x[0] + x[1] -1

    x0 = np.array([-2.,3.])
    opt = AugmentedLagrangian(x0,f,g,h, step_size=0.01)
    opt.solve(max_iter=1000)
    print(f(opt.solution['x']), opt.solution['x'])