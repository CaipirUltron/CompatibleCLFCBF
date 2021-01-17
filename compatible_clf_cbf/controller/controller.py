import numpy as np
from dynamic_systems import AffineSystem, QuadraticLyapunov, QuadraticBarrier

class QPController():
    def __init__(self, param):
        self.parameter = param