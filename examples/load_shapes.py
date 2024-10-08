'''
Use this script to create and save different CLF and CBF shapes into a file, for later loading.
'''
import numpy as np

from shapely import LineString, LinearRing, Polygon
from functions import LeadingShape, Kernel, KernelLyapunov, KernelBarrier, MultiPoly
from common import kernel_quadratic, rot2D, box, circular_boundary_shape, rot2D, polygon, load_compatible, discretize, segmentize, enclosing_circle

n = 2
limits = 12*np.array((-1,1,-1,1))

''' ------------------------------ Degree 2 CLFs/CBFs ------------------------------ '''
kernel_deg2 = Kernel(dim=n, degree=2)
dim_deg2 = kernel_deg2._num_monomials
powers_deg2 = kernel_deg2._powers

'''
Wierd looking CLF
'''
clf_center = [0.0, -3.0]
base_level = 15

clf_eig = np.array([ 8.0, 1.0 ])
clf_angle = np.deg2rad(0)
Pquadratic = kernel_quadratic(eigen=clf_eig, R=rot2D(clf_angle), center=clf_center, kernel_dim=dim_deg2)

points = []
points.append({ "coords": [-4.0,  6.0], "gradient": [-1.0,  0.5] })
points.append({ "coords": [ 4.0,  6.0], "gradient": [ 0.0,  6.5] })
points.append({ "coords": [ 0.0,  5.0], "gradient": [ 2.0,  6.0] })
points.append({ "coords": [ 0.0,  -8.0], "gradient": [ 0.0,  -1.0] })

clf_leading = LeadingShape(Pquadratic,approximate=True)
clf = KernelLyapunov(kernel=kernel_deg2, points=points, centers=[clf_center])
# clf = KernelLyapunov(kernel=kernel_deg2, points=points, centers=[clf_center], leading=clf_leading, limits=limits)
clf_poly = clf.to_multipoly()
clf_poly.save("wierd")

'''
Box-shaped obstacle
'''
box_center = [ 0.0, 2.0 ]
box_angle = 2
box_height, box_width = 5, 5

init_eig = [ 0.2/box_height, 0.2/box_width ]
init_R = rot2D(np.deg2rad(box_angle))
Qinit = kernel_quadratic(eigen=init_eig, R=init_R, center=box_center, kernel_dim=dim_deg2)
boundary_pts = box( center=box_center, height=box_height, width=box_width, angle=box_angle, spacing=0.4 )
cbf = KernelBarrier(kernel=kernel_deg2, boundary=boundary_pts, centers=[box_center], initial_shape = Qinit)
# cbf = KernelBarrier(kernel=kernel_deg2, boundary=boundary_pts, skeleton=skeleton, leading=LeadingShape(Qquadratic,bound='upper'), limits=limits, spacing=0.1)

cbf_poly = cbf.to_multipoly()
cbf_poly.save("box_shaped")

# Rotate obstacle
translation = [-0.0, -2.0]
rotation = rot2D(np.deg2rad(45))

rotated = cbf_poly.frame_transform( translation, rotation )
rotated.save("rotated_box")

''' ------------------------------ Degree 3 CLF/CBFs ------------------------------ '''
kernel_deg3 = Kernel(dim=n, degree=3)
dim_deg3 = kernel_deg3._num_monomials
powers_deg3 = kernel_deg3._powers

'''
U-shaped obstacle
'''
center = (0, 0)

# safe_pts = [(-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 3), (2, 3), (3, 3)]
# safe_pts += [(-3, 2.5), (-2, 2.5), (-1, 2.5), (0, 2.5), (1, 2.5), (2, 2.5), (3, 2.5)]

centers = [(-4, 3), (-4, 0), center, (4, 0), (4, 3)]
skeleton_line = LineString(centers)
skeleton_pts = discretize(skeleton_line, spacing=0.4)
skeleton_segs = segmentize(skeleton_pts, center)

obstacle_poly = skeleton_line.buffer(1.0, cap_style='flat')
boundary_pts = discretize(obstacle_poly, spacing=0.4)

# centers = [(-4, 2), (-4, 0), center, (4, 0), (4, 2)]
shape_matrix = circular_boundary_shape( radius=7, center=center, kernel_dim=dim_deg3 )
cbf_leading = LeadingShape(shape_matrix, bound='lower')

# quadratic_cbf = KernelBarrier(kernel=kernel, Q=shape_matrix, limits=limits, spacing=0.1)

# cbf = KernelBarrier(kernel=kernel, boundary=boundary_pts, centers=centers, limits=limits, spacing=0.1)
# cbf = KernelBarrier(kernel=kernel, boundary=boundary_pts, skeleton=skeleton_segs, limits=limits, spacing=0.1)
cbf = KernelBarrier(kernel=kernel_deg3, boundary=boundary_pts, skeleton=skeleton_segs, leading=cbf_leading)
cbf_poly = cbf.to_multipoly()

# Rotate obstacle
translation = [-2.0, 0.0]
rotation = rot2D(np.deg2rad(-30))

rotated = cbf_poly.frame_transform( translation, rotation )
rotated.save("rotated_U")