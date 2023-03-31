import scipy
from scipy import signal
import numpy as np


def solve_PEP1(Q, P, **kwargs):
    '''
    Solves the eigenproblem of the type: (lambda1 * Q - lambda2 * C - P) @ z = 0, lambda1 = lambda_2 + z.T @ P @ z
    where P and Q are ( n x n ) p.s.d. matrices and C is a nilpotent matrix with index 2.

    Returns: lambda1:    n-array of mu1 eigenvalues, repeated according to its multiplicity
             lambda2:    n-array of mu2 eigenvalues, repeated according to its multiplicity
             Z:          (n x n)-array of lambda values, each column corresponding to the corresponding eigenpair (mu1, mu2)
    ''' 
    accuracy = 0.0000000001

    if np.shape(Q) != np.shape(P):
        raise Exception("Matrix shapes are not compatible with given initial value.")
    
    matrix_shapes = np.shape(Q)
    if matrix_shapes[0] != matrix_shapes[1]:
        raise Exception("Matrices are not square.")

    C = np.zeros(matrix_shapes)
    C[-1,-1] = 1

    init_lambda = 0.0
    step = 1
    tol = 0.00001
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "init_lambda":
            init_lambda = kwargs[key]
            continue
        if aux_key == "step":
            step = kwargs[key]
            continue
        if aux_key == "tolerance":
            tol = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    def compute_F(solution):
        '''
        This inner method computes the vector field F(mu1, mu2, z) and returns its value.
        '''        
        lambda1 = solution[0]
        lambda2 = solution[1]
        z = solution[2:]

        n = len(z)

        F = np.zeros(n+2)
        L = lambda1 * Q - lambda2 * C - P

        F[0:n] = L @ z
        F[n] = 0.5 - 0.5 * z @ C @ z
        F[n+1] = 0.5 * z @ Q @ z - 0.5

        return F

    def compute_Jacobian(solution):
        '''
        This inner method computes the vector field F(mu1, mu2, z) and returns its value.
        '''        
        lambda1 = solution[0]
        lambda2 = solution[1]
        z = solution[2:]

        n = len(z)

        L = lambda1 * Q - lambda2 * C - P

        Jac1 = np.vstack( [ (Q @ z).reshape(n,1), 0, 0 ] )
        Jac2 = np.vstack( [ -(C @ z).reshape(n,1), 0, 0 ] )
        Jac3 = np.vstack( [ L, -(C @ z).reshape(1,n), (Q @ z).reshape(1,n) ] )
        Jac = np.hstack([ Jac1, Jac2, Jac3 ])

        return Jac

    # Initial guess
    pencil = LinearMatrixPencil2( Q, init_lambda * C + P )
    lambda1 = pencil.eigenvalues
    lambda2 = init_lambda * np.ones(len(lambda1))
    Z = pencil.eigenvectors

    # Normalize eigenvectors and delete the degenerate ones
    index_to_be_deleted = []
    for k in range(len(lambda1)):
        if np.abs(lambda1[k]) == np.inf or np.abs(lambda2[k]) == np.inf:
            index_to_be_deleted.append(k)
        z = Z[:,k]
        if np.linalg.norm(C @ z) < accuracy or np.linalg.norm(Q @ z) < accuracy or np.linalg.norm(P @ z) < accuracy:
            index_to_be_deleted.append(k)

    Z = np.delete(Z, index_to_be_deleted, axis = 1)
    lambda1 = np.delete(lambda1, index_to_be_deleted)
    lambda2 = np.delete(lambda2, index_to_be_deleted)

    print("Initial lambdas = " + str(lambda1))

    # for k in range(len(lambda1)):
    #     z = Z[:,k]
    #     normalization_const = 1/np.sqrt(z @ Q @ z)
    #     Z[:,k] = normalization_const * z

    num_solutions = len(lambda1)

    costs = np.zeros(num_solutions)
    solutions = np.zeros([matrix_shapes[0]+2, num_solutions])
    for k in range(num_solutions):
        solutions[0,k] = lambda1[k]
        solutions[1,k] = lambda2[k]
        solutions[2:,k] = Z[:,k]
        costs[k] = np.linalg.norm(compute_F(solutions[:,k]))

    lambda1_list, lambda2_list = [], []
    for k in range(num_solutions):
        lambda1_list.append([])
        lambda2_list.append([])
    
    for k in range(num_solutions):
        lambda1_list[k].append(lambda1[k])
        lambda2_list[k].append(lambda2[k])

    # Main loop
    num_iter = 0
    while np.any( costs > tol*np.ones(num_solutions) ) and num_iter < max_iter:
        num_iter += 1
        for k in range(num_solutions):
            
            solution = solutions[:,k]

            F = compute_F(solution)
            # print("F = " + str(F))
            Jac = compute_Jacobian(solution)
            # print("Jac = \n" + str(Jac))
            # print("det(Jac) = " + str(np.linalg.det(Jac)))
            invJac = np.linalg.inv(Jac)

            new_solution = solution - step * (invJac @ F)

            # Reprojection into pencil ------------------------------------------------

            # new_lambda1 = new_solution[0]
            # new_lambda2 = new_solution[1]
            # new_Z = new_solution[2:]

            # pencil = LinearMatrixPencil2( Q, new_lambda2 * C + P )
            # new_lambda1 = pencil.eigenvalues
            # new_Z = pencil.eigenvectors

            # # Delete degenerate eigenvalues
            # index_to_be_deleted = []
            # for i in range(len(new_lambda1)):
            #     if np.abs(new_lambda1[i]) == np.inf:
            #         index_to_be_deleted.append(i)
            #     z = new_Z[:,i]
            #     if np.linalg.norm(C @ z) < accuracy or np.linalg.norm(Q @ z) < accuracy or np.linalg.norm(P @ z) < accuracy:
            #         index_to_be_deleted.append(i)

            # print(index_to_be_deleted)

            # new_lambda1 = np.delete(new_lambda1, index_to_be_deleted)
            # new_Z = np.delete(new_Z, index_to_be_deleted, axis = 1)

            # distances = []
            # for i in range(len(new_lambda1)):
            #     distance = np.linalg.norm( new_Z[:,i] - new_solution[2:] )
            #     distances.append( distance )

            # if len(distances) > 0:
            #     closest = np.argmin(np.array(distances))
            #     new_lambda1 = new_lambda1[closest]
            #     new_Z = new_Z[:,closest]
            # else:
            #     new_lambda1 = new_solution[0]
            #     new_Z = new_solution[2:]

            # new_solution[0] = new_lambda1
            # new_solution[1] = new_lambda2
            # new_solution[2:] = new_Z

            #--------------------------------------------------------------------------

            costs[k] = np.linalg.norm(compute_F(new_solution))

            print("Cost = " + str(costs[k]))

            lambda1[k] = new_solution[0]
            lambda2[k] = new_solution[1]
            Z[:,k] = new_solution[2:]

            solutions[:,k] = new_solution

            lambda1_list[k].append(lambda1[k])
            lambda2_list[k].append(lambda2[k])

        # print('Costs = ' + str(costs))
        # print('Solutions = ' + str(solutions))

    return lambda1, lambda2, Z, lambda1_list, lambda2_list


def solve_PEP2( Q, P, **kwargs ):
    '''
    Solves the eigenproblem of the type: (lambda1 * Q - lambda2 * C - P) @ z = 0, lambda1 = lambda_2 + z.T @ P @ z
    where P and Q are ( n x n ) p.s.d. matrices and C is a nilpotent matrix with index 2.

    Returns: lambda1:    n-array of mu1 eigenvalues, repeated according to its multiplicity
             lambda2:    n-array of mu2 eigenvalues, repeated according to its multiplicity
             Z:          (n x n)-array of lambda values, each column corresponding to the corresponding eigenpair (mu1, mu2)
    '''
    accuracy = 0.0000000001

    if np.shape(Q) != np.shape(P):
        raise Exception("Matrix shapes are not compatible with given initial value.")
    
    matrix_shapes = np.shape(Q)
    if matrix_shapes[0] != matrix_shapes[1]:
        raise Exception("Matrices are not square.")

    C = np.zeros(matrix_shapes)
    C[-1,-1] = 1

    init_lambda = 0.0
    step = 1
    tol = 0.00001
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "init_lambda":
            init_lambda = kwargs[key]
            continue
        if aux_key == "step":
            step = kwargs[key]
            continue
        if aux_key == "tolerance":
            tol = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    def cost(lambda1, lambda2, z):
        '''
        Computes the cost and returns its value.
        '''
        error = lambda1 - lambda2 - z @ P @ z
        return error**2

    def cost_derivative(lambda1, lambda2, z):
        '''
        Computes cost derivative.
        '''        
        return lambda1 - lambda2 - z @ P @ z

    # Initial guess
    pencil = LinearMatrixPencil2( Q, init_lambda * C + P )
    lambda1 = pencil.eigenvalues
    lambda2 = init_lambda * np.ones(len(lambda1))
    Z = pencil.eigenvectors

    # Delete degenerate eigenvalues
    index_to_be_deleted = []
    for k in range(len(lambda1)):
        z = Z[:,k]
        if np.abs(lambda1[k]) == np.inf or np.abs(lambda2[k]) == np.inf:
            index_to_be_deleted.append(k)
        if np.linalg.norm(C @ z) < accuracy or np.linalg.norm(Q @ z) < accuracy or np.linalg.norm(P @ z) < accuracy:
            index_to_be_deleted.append(k)

    Z = np.delete(Z, index_to_be_deleted, axis = 1)
    lambda1 = np.delete(lambda1, index_to_be_deleted)
    lambda2 = np.delete(lambda2, index_to_be_deleted)

    # Normalize eigenvectors
    for k in range(len(lambda1)):
        normalization_const = 1/np.sqrt(z @ Q @ z)
        Z[:,k] = normalization_const * Z[:,k]

    num_solutions = len(lambda1)

    costs = np.zeros(num_solutions)
    for k in range(num_solutions): 
        costs[k] = cost(lambda1[k], lambda2[k], Z[:,k])

    lambda1_list, lambda2_list = [], []
    for _ in range(num_solutions):
        lambda1_list.append([])
        lambda2_list.append([])

    # Main loop
    num_iter = 0
    while np.any( costs > tol*np.ones(num_solutions) ) and num_iter < max_iter:

        print("Costs = " + str(costs))

        num_iter += 1
        for k in range(num_solutions):
            
            # Advance one step (recompute mu2 and costs)
            mu2_delta = step * cost_derivative(lambda1[k], lambda2[k], Z[:,k])

            aux_pencil = LinearMatrixPencil2( mu2_delta * C, lambda1[k] * Q - lambda2[k] * C - P )
            aux_eig = aux_pencil.eigenvalues
            eigs_in_interval = aux_eig[np.where((aux_eig <= 1) & (aux_eig > 0) & (np.abs(aux_eig) > accuracy) )[0]]
            if eigs_in_interval.size > 0:
                lambda2[k] = lambda2[k] + mu2_delta * np.min(eigs_in_interval)
            else:
                lambda2[k] = lambda2[k] + mu2_delta

            # Reprojection
            pencil = LinearMatrixPencil2( Q, lambda2[k] * C + P )
            candidate_mu1s = pencil.eigenvalues
            auxZ = pencil.eigenvectors

            for i in range(len(candidate_mu1s)):
                z = pencil.eigenvectors[:,i]
                if np.linalg.norm(C @ z) < accuracy or np.linalg.norm(Q @ z) < accuracy or np.linalg.norm(P @ z) < accuracy:
                    index_to_be_deleted.append(i)
            candidate_mu1s = np.delete(candidate_mu1s, index_to_be_deleted)
            auxZ = np.delete(auxZ, index_to_be_deleted, axis=1)

            # Filter candidate mu1s
            if mu2_delta > 0:
                indexes = np.where(candidate_mu1s > lambda1[k])[0]
            else:
                indexes = np.where(candidate_mu1s < lambda1[k])[0]

            distances = []
            for index in indexes:
                distances.append( np.abs( candidate_mu1s[index] - lambda1[k] ) )

            closest = np.argmin(np.array(distances))
            lambda1[k] = candidate_mu1s[indexes[closest]]
            z = auxZ[:,indexes[closest]]

            if z.T @ Q @ z >= accuracy:
                normalization_const = 1/np.sqrt( z.T @ Q @ z )
            else:
                normalization_const = 1
            Z[:,k] = normalization_const * z

            # Recompute cost
            costs[k] = cost(lambda1[k], lambda2[k], Z[:,k])

            lambda1_list[k].append(lambda1[k])
            lambda2_list[k].append(lambda2[k])

    return lambda1, lambda2, Z, lambda1_list, lambda2_list


class LinearMatrixPencil2():
    '''
    Class for regular, symmetric linear matrix pencils of the form P(mu1,mu2) = mu1 A - mu2 B, where A and B are p.s.d. matrices.
    '''
    accuracy = 0.0000000001
    
    def __init__(self, A, B, **kwargs):

        dimA = A.shape
        dimB = B.shape
        if dimA != dimB:
            raise Exception("Matrix dimensions are not equal.")
        if (dimA[0] != dimA[1]) or (dimB[0] != dimB[1]):
            raise Exception("Matrices are not square.")
        self._A, self._B = A, B
        self.dim = dimA[0]

        self.compute_eigen()

    def value(self, lambda_param):
        '''
        Returns pencil value.
        '''
        return lambda_param * self._A  - self._B

    def compute_eigen(self):
        '''
        Computes the generalized eigenvalues and eigenvectors of the pencil.
        '''
        # Compute the pencil eigenvalues
        schurA, schurB, _, _, _, _ = scipy.linalg.ordqz(self._B, self._A)
        schurA_vec = np.diag(schurA)
        schurB_vec = np.diag(schurB)

        self.eigenvalues = np.zeros(self.dim)
        for k in range(self.dim):
            if np.abs(schurB_vec[k]) > LinearMatrixPencil2.accuracy:
                self.eigenvalues[k] = schurA_vec[k]/schurB_vec[k]
            else:
                self.eigenvalues[k] = np.sign(schurA_vec[k]) * np.inf

        # Compute the (normalized, if possible) pencil eigenvectors
        self.eigenvectors = np.zeros([self.dim,self.dim])
        for k in range(len(self.eigenvalues)):
            if np.abs(self.eigenvalues[k]) != np.inf:
                eig, Q = np.linalg.eig( self.value( self.eigenvalues[k]) )
            else:
                eig, Q = np.linalg.eig( schurA_vec[k] * self._A - schurB_vec[k] * self._B )
            for i in range(len(eig)):
                if np.abs(eig[i]) <= LinearMatrixPencil2.accuracy:
                    # if np.abs(Q[-1,i]) > LinearMatrixPencil2.accuracy:
                        # normalization_const = 1/Q[-1,i]
                        # normalization_const = 1
                        # self.eigenvectors[:,k] = normalization_const * Q[:,i]
                    # else:
                    self.eigenvectors[:,k] = Q[:,i]

    def solve_nonlinear(self, const):
        '''
        Compute all the solutions for the nonlinear system: (mu1 A - mu2 B) z = 0               ,
                                                                          mu2 = 0.5 k z.T @ B z , 
                                                                          z \in Im(m(x))        
        '''
        mu1_list, mu2_list, z_list = [], [], []
        for k in range(self.dim):
            z = self.eigenvectors[:,k]
            if z[-1] == 1.0 and np.abs(self.eigenvalues[k]) != np.inf:
                mu2 = 0.5 * const * (z.T @ self._B @ z)
                mu1 = self.eigenvalues[k] * mu2
                
                mu1_list.append( mu1 )
                mu2_list.append( mu2 )
                z_list.append( z.tolist() )
        
        return np.array(mu1_list), np.array(mu2_list), np.array( z_list ).T


class LinearMatrixPencil():
    '''
    Class for regular, symmetric linear matrix pencils of the form H(\lambda) = mu1 A - mu2 B, where A and B are p.s.d. matrices
    '''
    def __init__(self, A, B, **kwargs):

        dimA = A.shape
        dimB = B.shape
        if dimA != dimB:
            raise Exception("Matrix dimensions are not equal.")
        if (dimA[0] != dimA[1]) or (dimB[0] != dimB[1]):
            raise Exception("Matrices are not square.")
        self._A, self._B = A, B
        self.dim = dimA[0]

        self._lambda = 0.0
        for key in kwargs:
            if key == "parameter":
                self._lambda = kwargs[key]

        self.compute_eig()
        # self.compute_eig2()
    
    def value(self, lambda_param):
        '''
        Returns pencil value.
        '''
        return lambda_param * self._A - self._B

    def value2(self, lambda1_param, lambda2_param):
        '''
        Returns pencil value.
        '''
        return lambda2_param * self._A - lambda1_param * self._B

    def compute_eig(self):
        '''
        Given the pencil matrices A and B, this method solves the pencil eigenvalue problem.
        '''
        # Compute the sorted pencil eigenvalues
        schurHv, schurHh, _, _, Q, Z = scipy.linalg.ordqz(self._B, self._A)
        self.lambda1 = np.diag(schurHh)
        self.lambda2 = np.diag(schurHv)
        pencil_eig = self.lambda2/self.lambda1
        sorted_args = np.argsort(pencil_eig)

        # Compute the pencil eigenvectors
        pencil_eigenvectors = np.zeros([self.dim,self.dim])
        for k in range(len(pencil_eig)):
            # eig, Q = np.linalg.eig( self.value(pencil_eig[k]) )
            eig, Q = np.linalg.eig( self.value2(self.lambda1[k], self.lambda2[k]) )
            for i in range(len(eig)):
                if np.abs(eig[i]) <= 0.000001:
                    normalization_const = 1/np.sqrt(Q[:,i].T @ self._A @ Q[:,i])
                    pencil_eigenvectors[:,k] = normalization_const * Q[:,i]
                    break

        # Assumption: B is invertible => detB != 0
        detB = np.linalg.det(self._B)
        # if detB == 0:
        #     raise Exception("B is rank deficient.")

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        pencil_det = np.real(np.prod(pencil_eig))
        self.characteristic_poly = ( detB/pencil_det ) * np.real(np.polynomial.polynomial.polyfromroots(pencil_eig))

        # Sorts eigenpairs
        self.eigenvalues = pencil_eig[sorted_args]
        self.eigenvectors = pencil_eigenvectors[:,sorted_args]

    def __str__(self):         
        '''
        Print the given pencil.
        '''
        np.set_printoptions(precision=3, suppress=True)
        ret_str = '{}'.format(type(self).__name__) + " = {:.3f}".format(self._lambda) + ' A - B \n'
        ret_str = ret_str + 'A = ' + self._A.__str__() + '\n'
        ret_str = ret_str + 'B = ' + self._B.__str__()
        return ret_str


class CLFCBFPair():
    '''
    Class for a CLF-CBF pair. Computes the q-function, equilibrium points and critical points of the q-function.
    '''
    def __init__(self, clf, cbf):

        self.eigen_threshold = 0.000001
        self.update(clf = clf, cbf = cbf)

    def update(self, **kwargs):
        '''
        Updates the CLF-CBF pair.
        '''
        for key in kwargs:
            if key == "clf":
                self.clf = kwargs[key]
            if key == "cbf":
                self.cbf = kwargs[key]
        
        self.Hv = self.clf.get_hessian()
        self.x0 = self.clf.get_critical()
        self.Hh = self.cbf.get_hessian()
        self.p0 = self.cbf.get_critical()
        self.v0 = self.Hv @ ( self.p0 - self.x0 )

        self.pencil = LinearMatrixPencil( self.cbf.get_hessian(), self.clf.get_hessian() )
        self.dim = self.pencil.dim

        self.compute_q()
        self.compute_equilibrium()
        # self.compute_equilibrium2()
        self.compute_critical()

    def compute_equilibrium2(self):
        '''
        Compute the equilibrium points using new method.
        '''
        temp_P = -(self.Hv @ self.x0).reshape(self.dim,1)
        P_matrix = np.block([ [ self.Hv  , temp_P                        ], 
                              [ temp_P.T , self.x0 @ self.Hv @ self.x0 ] ])

        temp_Q = -(self.Hh @ self.p0).reshape(self.dim,1)
        Q_matrix = np.block([ [ self.Hh  , temp_Q                        ], 
                              [ temp_Q.T , self.p0 @ self.Hh @ self.p0 ] ])

        pencil = LinearMatrixPencil( Q_matrix, P_matrix )
        # print("Eig = " + str(pencil.eigenvectors))

        # self.equilibrium_points2 = np.zeros([self.dim, self.dim+1])1
        self.equilibrium_points2 = []
        for k in range(np.shape(pencil.eigenvectors)[1]):
            # if np.abs(pencil.eigenvectors[-1,k]) > 0.0001:
            # print(pencil.eigenvectors)
            self.equilibrium_points2.append( (pencil.eigenvectors[0:-1,k]/pencil.eigenvectors[-1,k]).tolist() )
        
        self.equilibrium_points2 = np.array(self.equilibrium_points2).T

        # print("Lambda 1 = " + str(pencil.lambda1))
        # print("Lambda 2 = " + str(pencil.lambda2))
        # print("Eq = " + str(self.equilibrium_points2))

    def compute_q(self):
        '''
        This method computes the q-function for the pair.
        '''
        # Compute denominator of q
        pencil_eig = self.pencil.eigenvalues
        pencil_char = self.pencil.characteristic_poly
        den_poly = np.polynomial.polynomial.polymul(pencil_char, pencil_char)

        detHv = np.linalg.det(self.Hv)
        try:
            Hv_inv = np.linalg.inv(self.Hv)
            Hv_adj = detHv*Hv_inv
        except np.linalg.LinAlgError as error:
            print(error)
            return

        # This computes the pencil adjugate expansion and the set of numerator vectors by the adapted Faddeev-LeVerrier algorithm.
        D = np.zeros([self.dim, self.dim, self.dim])
        D[:][:][0] = pow(-1,self.dim-1) * Hv_adj

        Omega = np.zeros( [ self.dim, self.dim ] )
        Omega[0,:] = D[:][:][0].dot(self.v0)
        for k in range(1,self.dim):
            D[:][:][k] = np.matmul( Hv_inv, np.matmul(self.Hh, D[:][:][k-1]) - pencil_char[k]*np.eye(self.dim) )
            Omega[k,:] = D[:][:][k].dot(self.v0)

        # Computes the numerator polynomial
        W = np.zeros( [ self.dim, self.dim ] )
        for i in range(self.dim):
            for j in range(self.dim):
                W[i,j] = np.inner(self.Hh.dot(Omega[i,:]), Omega[j,:])

        num_poly = np.polynomial.polynomial.polyzero
        for k in range(self.dim):
            poly_term = np.polynomial.polynomial.polymul( W[:,k], np.eye(self.dim)[:,k] )
            num_poly = np.polynomial.polynomial.polyadd(num_poly, poly_term)

        residues, poles, k = signal.residue( np.flip(num_poly), np.flip(den_poly), tol=0.001, rtype='avg' )

        index = np.argwhere(np.real(residues) < 0.0000001)
        residues = np.real(np.delete(residues, index))

        # Computes polynomial roots
        fzeros = np.real( np.polynomial.polynomial.polyroots(num_poly) )

        # Filters repeated poles from pencil_eig and numerator_roots
        repeated_poles = []
        for i in range( len(pencil_eig) ):
            for j in range( len(fzeros) ):
                if np.absolute(fzeros[j] - pencil_eig[i]) < self.eigen_threshold:
                    if np.any(repeated_poles == pencil_eig[i]):
                            break
                    else:
                        repeated_poles.append( pencil_eig[i] )
        repeated_poles = np.array( repeated_poles )

        self.q_function = {
                            "denominator": den_poly,
                            "numerator": num_poly,
                            "poles": pencil_eig,
                            "zeros": fzeros,
                            "repeated_poles": repeated_poles,
                            "residues": residues }

    def compute_equilibrium(self):
        '''
        Compute equilibrium solutions and equilibrium points.
        '''
        solution_poly = np.polynomial.polynomial.polysub( self.q_function["numerator"], self.q_function["denominator"] )
        
        equilibrium_solutions = np.polynomial.polynomial.polyroots(solution_poly)
        equilibrium_solutions = np.real(np.extract( equilibrium_solutions.imag == 0.0, equilibrium_solutions ))
        equilibrium_solutions = np.concatenate((equilibrium_solutions, self.q_function["repeated_poles"]))

        # Extract positive solutions and sort array
        equilibrium_solutions = np.sort( np.extract( equilibrium_solutions > 0, equilibrium_solutions ) )

        # Compute equilibrium points from equilibrium solutions
        self.equilibrium_points = np.zeros([self.dim,len(equilibrium_solutions)])
        for k in range(len(equilibrium_solutions)):
            if all(np.absolute(equilibrium_solutions[k] - self.pencil.eigenvalues) > self.eigen_threshold ):
                self.equilibrium_points[:,k] = self.v_values( equilibrium_solutions[k] ) + self.p0

    def compute_critical(self):
        '''
        Computes critical points of the q-function.
        '''
        dnum_poly = np.polynomial.polynomial.polyder(self.q_function["numerator"])
        dpencil_char = np.polynomial.polynomial.polyder(self.pencil.characteristic_poly)

        poly1 = np.polynomial.polynomial.polymul(dnum_poly, self.pencil.characteristic_poly)
        poly2 = 2*np.polynomial.polynomial.polymul(self.q_function["numerator"], dpencil_char)
        num_df = np.polynomial.polynomial.polysub( poly1, poly2 )

        self.q_critical_points = np.polynomial.polynomial.polyroots(num_df)
        self.q_critical_points = np.real(np.extract( self.q_critical_points.imag == 0.0, self.q_critical_points ))

        # critical_values = self.q_values(self.q_critical)
        # number_critical = len(self.critical_values)

        # # Get positive critical points
        # index, = np.where(self.q_critical > 0)
        # positive_q_critical = self.q_critical[index]
        # positive_critical_values = self.critical_values[index]
        # num_positive_critical = len(self.positive_q_critical)

    def q_values(self, args):
        '''
        Returns the q-function values at given points.
        '''
        numpoints = len(args)
        qvalues = np.zeros(numpoints)
        for k in range(numpoints):
            num_value = np.polynomial.polynomial.polyval( args[k], self.q_function["numerator"] )
            pencil_char_value = np.polynomial.polynomial.polyval( args[k], self.pencil.characteristic_poly )
            qvalues[k] = num_value/(pencil_char_value**2)

        return qvalues

    def v_values( self, lambda_var ):
        '''
        Returns the value of v(lambda) = H(lambda)^{-1} v0
        '''
        pencil_inv = np.linalg.inv( self.pencil.value( lambda_var ) )
        return pencil_inv.dot(self.v0)