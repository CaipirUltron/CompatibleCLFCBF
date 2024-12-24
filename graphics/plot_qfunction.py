import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from common import rot3D

radius = 1.0

sigmas = [ 0.5, 2.8, 6.0 ]
x_i = [ 1.0, 1.0, 0.5 ]

# sigmas = [ 5.0, 1.2, 2. ]
# x_i = [ 1.0, 1.6, 2.5 ]

residues = []
pencil_eigen = []
for k in range(len(sigmas)):
    pencil_eigen.append( (radius**2)*sigmas[k] )
    residues.append(  radius*sigmas[k]*x_i[k] )
dimension = len(pencil_eigen)

p_min, p_max = -2, 10
q_min, q_max = 0, 30
lambda_p = np.linspace(p_min, p_max, 1000)

def lyapunov(x):
    return 0.5 * x.T @ np.diag(sigmas) @ x

def barrier(x):
    return 0.5 * (1/radius**2) * (x-x_i).T @ np.eye(dimension) @ (x-x_i) - 0.5

def v_function(l):
    H = np.zeros([dimension, dimension])
    b = np.zeros(dimension)
    for k in range(dimension):
        H[k,k] = l - pencil_eigen[k]
        b[k] = sigmas[k]*x_i[k]
    return (radius**2)*np.linalg.inv(H) @ b

def q_function(l):
    sum = 0.0
    for k in range(dimension):
        sum += (residues[k])**2/(l - pencil_eigen[k])**2
    return sum

def q_root_function(l):
    return q_function(l) - 1

q = np.zeros(len(lambda_p))
for k in range(len(lambda_p)):
    q[k] = q_function(lambda_p[k])

max_eig = np.max(pencil_eigen)
lambda_sol = fsolve(q_root_function, max_eig + 1)
eq_point = v_function(lambda_sol) + x_i

V_constant = lyapunov(eq_point)

# Plot ellipsoids -------------------------------------------------------------------------------

fig = plt.figure(figsize=(8,4)) 
# fig.suptitle("3D Numerical Example", fontsize=12)

ax = fig.add_subplot(121, projection='3d')

# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# coefs = (1, 2, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
# Radii corresponding to the coefficients:
coefs =  [sigma/(2*V_constant) for sigma in sigmas]
rvx, rvy, rvz = 1/np.sqrt(coefs)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x_v = rvx * np.outer(np.cos(u), np.sin(v))
y_v = rvy * np.outer(np.sin(u), np.sin(v))
z_v = rvz * np.outer(np.ones_like(u), np.cos(v))    

coefs = [1.0/(radius**2) for sigma in sigmas ]
rhx, rhy, rhz = 1/np.sqrt(coefs)

x_h = rhx * np.outer(np.cos(u), np.sin(v)) + x_i[0]
y_h = rhy * np.outer(np.sin(u), np.sin(v)) + x_i[1]
z_h = rhz * np.outer(np.ones_like(u), np.cos(v)) + x_i[2]

# Plot:
ax.plot_surface(x_v, y_v, z_v, rstride=4, cstride=4, color='b', alpha=0.1)
ax.plot_surface(x_h, y_h, z_h, rstride=4, cstride=4, color='g', alpha=0.4)
ax.scatter(*eq_point.tolist(), color='b', linewidth=2.8, alpha=1.0)
ax.text(*(eq_point+np.array([0.2,0.2,0.2])).tolist(), "$x^\star$")

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")

ax.set_title('(a) CLF and CBF level sets', fontsize=12)

# Adjustment of the axes, so that they all have the same span:
max_radius = max(rvx, rvy, rvz, rhx + x_i[0], rhy + x_i[1], rhz + x_i[2])
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

# Plot q-function -------------------------------------------------------------------------------

# fig, ax = plt.subplots(tight_layout=True)
ax = fig.add_subplot(122)
ax.plot(lambda_p, q, color='blue', linewidth=0.8)
ax.plot([p_min, p_max], [1, 1], '--', color='green', linewidth=2.0)
ax.plot(lambda_sol, 1.0, 'o', color = "blue")
ax.plot(0.0, q_function(0.0), 'o', color="black")

for eig in pencil_eigen:
    ax.plot([eig, eig], [q_min, q_max], '--', color='red')

ax.set_xlim([p_min, p_max])
ax.set_ylim([q_min, q_max])
ax.set_xlabel("$\lambda$")
ax.legend(["$q(\lambda)$"], fontsize=10, loc="upper right")

ax.set_box_aspect(1)
ax.set_title('(b) Q-function', fontsize=12)

plt.subplots_adjust(wspace = 0.5)

# plt.savefig(test_config + ".eps", format='eps', transparent=True)
plt.savefig("q_function.eps", format='eps', transparent=True)
plt.savefig("q_function.png", format='png', transparent=True)

plt.show()