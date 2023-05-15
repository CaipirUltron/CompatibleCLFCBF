import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as mcolors
from functions import PolynomialFunction
from common import symmetric_basis

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111)

min_lim, max_lim = -10, 10
ax.set_xlim(min_lim, max_lim)
ax.set_ylim(min_lim, max_lim)

fps = 30
n = 2
state = np.random.rand(n)

maxdegree = 3
clf = PolynomialFunction(*state, degree = maxdegree)
cbf = PolynomialFunction(*state, degree = maxdegree)
kernel = clf.get_kernel()
k_dim = len(kernel)

P0 = np.random.rand(k_dim,k_dim)
P0 = P0.T @ P0
eig, V = np.linalg.eig(P0)
P0 = P0/max(eig)

clf_contour_color = mcolors.TABLEAU_COLORS['tab:green']
cbf_contour_color = mcolors.TABLEAU_COLORS['tab:green']

clf_contours = clf.contour_plot(ax, levels=[1.0], colors=clf_contour_color, min=min_lim, max=max_lim, resolution=0.5)
cbf_contours = cbf.contour_plot(ax, levels=[1.0], colors=cbf_contour_color, min=min_lim, max=max_lim, resolution=0.5)

basis = symmetric_basis(k_dim)
A = np.random.rand(len(basis))
a = np.random.rand(len(basis))
b = np.random.rand(len(basis))

def init():

    Pi = 0*P0
    clf.set_param(P = Pi)
    clf_contours = clf.contour_plot(ax, [1.0], colors=clf_contour_color, min=min_lim, max=max_lim, resolution=0.5)

    graphical_elements = []
    graphical_elements += clf_contours.collections
    return graphical_elements

def update(i):

    Pi = np.zeros([k_dim, k_dim])
    for k in range(len(basis)):
        Pi += A[k]*np.sin(a[k]*i + b[k]) * basis[k]
    eig, V = np.linalg.eig(Pi)
    Pi = Pi.T @ Pi/(100*max(eig)**2)
    Pi[0,0] += 0.2

    print(i)

    clf.set_param(P = Pi)

    # for coll in clf_contours.collections:
    #     coll.remove()
    clf_contours = clf.contour_plot(ax, [1.0], colors=clf_contour_color, min=min_lim, max=max_lim, resolution=0.2)

    graphical_elements = []
    graphical_elements += clf_contours.collections

    return graphical_elements

animation = anim.FuncAnimation(fig, func=update, init_func=init, frames=2000, interval=1000/fps, repeat=False, blit=True)
plt.show()
