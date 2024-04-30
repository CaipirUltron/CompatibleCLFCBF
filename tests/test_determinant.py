from functions import timeit
import numpy as np

N = int(1e+6)

@timeit
def compute_det1():
    M_array = np.empty((N,2,2))
    for k in range(N):
        M_array[k,:,:] = np.random.rand(2,2)
    determinants = np.linalg.det(M_array)
    print(len(determinants))

@timeit
def compute_det2():
    M_list = []
    for k in range(N):
        M_list.append( np.random.rand(2,2) )
    determinants = np.linalg.det(M_list)
    print(determinants.shape)

@timeit
def compute_det3():
    determinants = []
    for k in range(N):
        determinants.append( np.linalg.det(np.random.rand(2,2)) )
    print(len(determinants))

compute_det1()
compute_det2()
compute_det3()