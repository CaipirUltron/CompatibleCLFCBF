import math
import numpy as np
import matplotlib.pyplot as plt

from common import rot2D

# np.set_printoptions(precision=3, suppress=True)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)
ax.set_title("Hyperbolic Expansion")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

N = 1000
t_list = np.linspace(-100, 100, N)

# ---------------- Real Hyperbola -------------
axis = [ 1, 1 ]
center = [ 0, 0 ]
angle = 0
R = rot2D( np.deg2rad(angle) )

''' First half '''
x1_list = axis[0]*np.cosh(t_list) 
y1_list = axis[1]*np.sinh(t_list)
x1_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x1_list, y1_list) ]
y1_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x1_list, y1_list) ]
ax.plot(x1_rot, y1_rot, 'b')

''' Second half '''
x2_list = -x1_list
y2_list = y1_list
x2_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x2_list, y2_list) ]
y2_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x2_list, y2_list) ]
ax.plot(x2_rot, y2_rot, 'b')

# ---------------- Approx. Hyperbola -------------
order = 10

''' First half approx. '''
num_samples = len(t_list)
sum_cosh, sum_sinh = np.zeros(num_samples), np.zeros(num_samples)
for k, t in enumerate(t_list):
    sum_cosh[k] = np.sum([ (1/math.factorial(2*n))*(t**(2*n)) for n in range(order) ])
    sum_sinh[k] = np.sum([ (1/math.factorial(2*n+1))*(t**(2*n+1)) for n in range(order) ])

print(f"cosh = {sum_cosh[0]}")
print(f"sinh = {sum_sinh[0]}")

x1_list = axis[0]*sum_cosh
y1_list = axis[1]*sum_sinh

x1_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x1_list, y1_list) ]
y1_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x1_list, y1_list) ]
ax.plot(x1_rot, y1_rot, 'r--')

''' Second half approx. '''
x2_list = -x1_list
y2_list = y1_list
x2_rot = [ R[0,:].T @ np.array([x,y]) + center[0] for x,y in zip(x2_list, y2_list) ]
y2_rot = [ R[1,:].T @ np.array([x,y]) + center[1] for x,y in zip(x2_list, y2_list) ]
ax.plot(x2_rot, y2_rot, 'r--')

plt.show()
