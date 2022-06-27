import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from SumOfSquares import SOSProblem, poly_opt_prob, Basis
import time

x, y, s, t = sp.symbols('x y s t')
p = s*x**6 + t*y**6 - x**4*y**2 - x**2*y**4 - x**4 + 3*x**2*y**2 - y**4 - x**2 - y**2 + 1

# list_of_monomials = [ 1, x, y, x*y, x**2, y**2 ]

dimension = 2
degree = 1
b = Basis.from_degree(dimension, degree)
list_of_monomials = b.to_sym([x,y])

p = len(list_of_monomials)
P = sp.Matrix(sp.symarray('p',(p,p)))
m = sp.Matrix(list_of_monomials)

V = 

prob = SOSProblem()
V_constraint = prob.add_sos_constraint(V, [x, y], name="positive_definiteness")
Pv = prob.sp_mat_to_picos(P)

Pdes = np.random.rand(p,p)
Pdelta = Pv - Pdes

cost = 0.0
for i in range(p):
    for j in range(p):
        cost = Pdelta[i,j]**2
print(cost)

start = time.time()

prob.set_objective('min', cost)
prob.solve()

end = time.time()
print("Time: " + str(end - start))

lambdifed_poly = sp.lambdify( [x, y], sum(V_constraint.get_sos_decomp()) )
lamb_const = sp.lambdify( [x, y], 1.0 )

xlist = np.linspace(-10.0,10.0,100)
ylist = np.linspace(-10.0,10.0,100)
X, Y = np.meshgrid(xlist,ylist)
Z = lambdifed_poly(X,Y)

fig, ax = plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Polynomial Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()