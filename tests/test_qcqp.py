import picos as pc
import numpy as np

qcqp = pc.Problem()

dim = 3
A = np.random.rand(dim,dim)
P = A.T @ A                 # PSD matrix 
v = np.random.rand(dim)

z = pc.RealVariable("z", dim)
cost = ( z | P * z)

qcqp.set_objective("max", cost)

orthogonality_constr = (v | z) == 0
normalization_constr = abs(z) == 1

qcqp.add_constraint(orthogonality_constr)
qcqp.add_constraint(normalization_constr)

print(qcqp)
print(qcqp.solve())