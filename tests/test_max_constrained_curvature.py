import numpy as np

def max_constrained_curvature(P, v):
    '''
    Given a n x n matrix P and a n-dimensional vector v, compute max z' P z s.t. z'v = 0 and z'z = 1.
    '''
    dimP = np.shape(P)[0]
    normalized_v = v.reshape(dimP,1)/np.linalg.norm(v)
    _, V = np.linalg.eig(P)

    values = []
    for i in range(dimP):
        values.append( 1-(v.dot(V[:,i]))**2 )
    index_to_delete = np.argmin(values)
    np.delete(V,index_to_delete,1)      # discart one of the columns

    M = np.hstack([normalized_v, V])
    Q, R = np.linalg.qr(M)              # columns of Q are vectors from Gram Schmidt orthogonalization

    Pbar = Q.T @ P @ Q
    sol_eigs, _ = np.linalg.eig(Pbar[1:,1:])

    return np.max(sol_eigs)

dim = 5
P = np.random.rand(dim,dim)
P = P + P.T
eigP, _ = np.linalg.eig(P)

v = np.random.rand(dim)

result = max_constrained_curvature(P,v)

print("Result = " + str(result) + "\n")
print("Eigenvalues of P = " + str(eigP))