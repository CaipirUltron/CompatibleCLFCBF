import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create figure and meshgrids for plotting
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure(tight_layout=True)

ax = fig.add_subplot(111, projection='3d')

u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r)

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

x, y = np.meshgrid(x, y)
a, b = 0.1, -0.6
z = a*x + b*y
ax.plot_surface(x, y, z, alpha=0.4, color='green',)

def f(x, y):
   return (a*x+b*y)**2 + x**2 + y**2

# ax.contour3D(x, y, f(x,y), 1)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
# ax.set_xlabel("")
# ax.set_ylabel("")
# ax.set_title("Feasible domain")

plt.show()
