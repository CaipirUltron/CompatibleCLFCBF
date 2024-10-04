'''
GOAL: 
use this script to create and save different shapes into a file, for later loading into CLFs or CBFs.
'''
import numpy as np

from functions import LeadingShape, Kernel, KernelLyapunov, KernelBarrier, KernelFamily, MultiPoly
from common import kernel_quadratic, circular_boundary_shape, rot2D, polygon, load_compatible, discretize, segmentize, enclosing_circle

# ---------------------------------------------- Define kernel function ----------------------------------------------------
d1kernel = Kernel(dim=2, degree=2)
kernel_dim = d1kernel._num_monomials
powers = d1kernel._powers