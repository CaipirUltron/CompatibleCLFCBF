'''
Interpolation example.
Must define the following methods:

    - Afun(t), Bfun(t) : callable method of a time variable t for system matrices A, B (for LTI system)
    - Hvfun(t), V0_fun(t): callable method of a time variable t for the CLF Hessian Hv and center vector
    - Hhfun(t), h0_fun(t): callable method of a time variable t for the CBF Hessian Hh and center vector    
'''

import numpy as np
from common import hessian_quadratic, rot2D, rot3D, interpolation

''' ---------------------------------- Define system ---------------------------------- '''
A = np.array([[-1, 0, 0],
              [ 0,-1, 0],
              [ 0, 0,-1]])

B = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

def Afun(t): 
    return A

def Bfun(t): 
    return B

n, m = A.shape[0], B.shape[1]

''' ---------------------------- CLF parameters--------------------------- '''
def Hvfun(t):
    eigs1 = np.array([1.0, 8.0, 2.0])
    eigs2 = np.array([1.0, 8.0, 2.0])
    eigs_fun = interpolation(eigs1, eigs2)

    R1 = rot3D(theta=  0.0, axis=[1.0, 0.0, 0.0])
    R2 = rot3D(theta=270.0, axis=[1.0, 0.0, 1.0])
    Rfun = interpolation(R1, R2)
    
    return hessian_quadratic( eigs_fun(t), Rfun(t) )

def V0_fun(t):
    CLFcenter = np.zeros(n)
    return CLFcenter

''' ---------------------------- CBF parameters --------------------------- '''
def Hhfun(t):
    eigs1 = np.array([2.0, 1.0, 3.0])
    eigs2 = np.array([2.0, 1.0, 3.0])
    eigs_fun = interpolation(eigs1, eigs2)

    R1 = rot3D(theta=0.0, axis=[1.0, 0.0, 0.0])
    R2 = rot3D(theta=0.0, axis=[1.0, 0.0, 0.0])
    Rfun = interpolation(R1, R2)
    
    return hessian_quadratic( eigs_fun(t), Rfun(t) )

def h0_fun(t):
    CBFcenter = np.array([6.0, 0.0, 2.0])
    return CBFcenter

p = 1.0