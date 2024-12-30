from dynamic_systems import *

dyn = lambda x, u: x+u

f = lambda x: -x
g = lambda x: np.array([[-x, 0], [0, -x]])

n, m = 2, 2

cs = ControlSystem(n=n,m=m,dynamics=dyn)
affine = AffineSystem(n=n, m=m, f=f, g=g)
integrator = Integrator(n=n)

A = np.array([[4,5],[-1,2]])
B = np.array([[4],[-1]])
lti = LinearSystem(A=A, B=B)