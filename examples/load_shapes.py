'''
GOAL: 
use this script to create and save different shapes into a file, for later loading into CLFs or CBFs.
'''
import numpy as np

from functions import LeadingShape, Kernel, KernelLyapunov, KernelBarrier, KernelFamily, MultiPoly
from common import kernel_quadratic, rot2D, box, circular_boundary_shape, rot2D, polygon, load_compatible, discretize, segmentize, enclosing_circle

n = 2
limits = 12*np.array((-1,1,-1,1))

'''
Degree 1
'''
kernel_deg1 = Kernel(dim=2, degree=2)
dim_deg1 = kernel_deg1._num_monomials
powers_deg1 = kernel_deg1._powers





'''
Degree 2
'''
kernel_deg2 = Kernel(dim=n, degree=2)
dim_deg2 = kernel_deg2._num_monomials
powers_deg2 = kernel_deg2._powers

''' ------------------------------- Box-shaped obstacle ------------------------------ '''
box_center = [ 0.0, 2.0 ]
box_angle = 2
box_height, box_width = 5, 5

init_eig = [ 0.2/box_height, 0.2/box_width ]
init_R = rot2D(np.deg2rad(box_angle))
Qinit = kernel_quadratic(eigen=init_eig, R=init_R, center=box_center, kernel_dim=dim_deg2)
boundary_pts = box( center=box_center, height=box_height, width=box_width, angle=box_angle, spacing=0.4 )
cbf = KernelBarrier(kernel=kernel_deg2, boundary=boundary_pts, centers=[box_center], initial_shape = Qinit, limits=limits, spacing=0.1)