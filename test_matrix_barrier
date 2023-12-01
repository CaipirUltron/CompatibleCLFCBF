import numpy as np
import cvxpy as cp
from functions import Kernel

n = 2
max_degree = 2
kernel = Kernel(*[0.0, 0.0], degree = max_degree)
p = kernel.kernel_dim
N_list = kernel.get_N_matrices()
r = len(N_list)

P = np.random.rand(p,p)
P = P @ P.T
Q = np.random.rand(p,p)
Q = Q @ Q.T
F = np.zeros([p,p])
# F = np.random.rand(p,p)
# F = F @ F.T

u_x = cp.Variable(n)
u_kappa = cp.Variable(r)
delta = cp.Variable()

gradCx = cp.Parameter(n)
gradCkappa = cp.Parameter(r)

kappa = cp.Parameter(r)
cost = cp.Parameter()

c = 1
q, alpha, beta = 1, 1, 1
objective = cp.Minimize( cp.norm(u_x)**2 + cp.norm(u_kappa)**2 + q*delta**2 )
CLF_constr = [ gradCx.T @ u_x + gradCkappa.T @ u_kappa + alpha * cost <= delta ]
B = cp.sum([ kappa[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) - 2 * F
CBF_constr = [ cp.sum([ u_kappa[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) + beta * B >> 0 ]
problem = cp.Problem(objective, CLF_constr + CBF_constr)

max_it = 100
it = 0
while it < max_it:
      it += 1
      gradCx.value = np.random.rand(n)
      gradCkappa.value = np.random.rand(r)
      kappa.value = np.random.rand(r)
      cost.value = np.random.rand()**2

      problem.solve()
      print("Iteration " + str(it))
      print("Problem status: " + problem.status + " with cost = " + str(problem.value))
      print("u_x = " + str(u_x.value))
      print("u_kappa = " + str(u_kappa.value))