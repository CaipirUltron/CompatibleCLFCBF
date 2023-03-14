import scipy
from scipy import signal
import numpy as np


def compute_mu(B, const, z):
    return 0.5 * const * z @ B @ z


def solve_PEP( A, B, C, const, init_mu2 ):
    '''
    Solves the eigenproblem of the type: (mu1 * A + mu2 * B + C) @ z = 0, mu2 = 0.5 k * z.T @ B @ z
    where A, B, C are ( n x n ) matrices and k > 0.

    Returns: mu1:    n-array of mu1 eigenvalues, repeated according to its multiplicity
             mu2:    n-array of mu2 eigenvalues, repeated according to its multiplicity
             Z:      (n x n)-array of lambda values, each column corresponding to the corresponding eigenpair (mu1, mu2)
    '''
    adjust = 0.1
    dim = len(init_mu2)

    mu2 = init_mu2
    pencil = LinearMatrixPencil( A, mu2 * B + C )
    
    mu1 = pencil.eigenvalues
    Z = pencil.eigenvectors

    new_mu1 = mu1
    new_mu2 = np.zeros(dim)
    for k in range(dim):
        new_mu2[k] = compute_mu( B, const, Z[:,k] )
        Z[:,k] = Z[:,k]/ np.sqrt( Z[:,k] @ A @ Z[:,k] )

    while np.any( np.abs( new_mu2 - mu2 ) >= 0.00001 ):

        print("Lacks = " + str(np.abs( new_mu2 - mu2 )))

        mu1 = new_mu1
        mu2 = new_mu2

        new_mu1 = np.zeros(dim)
        new_mu2 = np.zeros(dim)
        for k in range(dim):

            # Predict mu1 using Newton-Raphson rule 
            slope = ( Z[:,k] @ B @ Z[:,k] / Z[:,k] @ A @ Z[:,k] )
            new_mu1[k] = mu1[k] + adjust * slope * ( compute_mu( B, const, Z[:,k] ) - mu2[k] )

            pencil = LinearMatrixPencil( B, new_mu1[k] * A + C )

            # Compute closest eigenvector
            eigen_distances = np.zeros(dim)
            for i in range(dim):
                c = np.sqrt(pencil.eigenvectors[:,i] @ A @ pencil.eigenvectors[:,i])
                eigen_distances[i] = np.linalg.norm( pencil.eigenvectors[:,i]/c - Z[:,k] )
            closest = np.argmin(eigen_distances)

            # Update eigenvector and mu2 from pencil
            Z[:,k] = pencil.eigenvectors[:,closest]
            Z[:,k] = Z[:,k]/ np.sqrt( Z[:,k] @ A @ Z[:,k] )
            new_mu2[k] = pencil.eigenvalues[closest]

        # mu1 = new_mu1
        # mu2 = new_mu2

        # new_mu1 = np.zeros(dim)
        # new_mu2 = np.zeros(dim)
        # for k in range(dim):

        #     # Predict mu2 using Newton-Raphson rule 
        #     slope = -( Z[:,k] @ A @ Z[:,k] / Z[:,k] @ B @ Z[:,k] )
        #     desired_mu1 = - 0.5 * const * (( Z[:,k] @ B @ Z[:,k] )**2)/(Z[:,k] @ A @ Z[:,k]) - (Z[:,k] @ C @ Z[:,k] / Z[:,k] @ A @ Z[:,k])
        #     new_mu2[k] = mu2[k] + adjust * slope * ( desired_mu1 - mu1[k] )

        #     pencil = LinearMatrixPencil( A, new_mu2[k] * B + C )

        #     # Compute closest eigenvector
        #     eigen_distances = np.zeros(dim)
        #     for i in range(dim):
        #         c = np.sqrt(pencil.eigenvectors[:,i] @ A @ pencil.eigenvectors[:,i])
        #         eigen_distances[i] = np.linalg.norm( pencil.eigenvectors[:,i]/c - Z[:,k] )
        #     closest = np.argmin(eigen_distances)

        #     # Update eigenvector and mu1 from pencil
        #     Z[:,k] = pencil.eigenvectors[:,closest]
        #     Z[:,k] = Z[:,k]/ np.sqrt( Z[:,k] @ A @ Z[:,k] )
        #     new_mu1[k] = pencil.eigenvalues[closest]

        # print("Lacks = " + str(np.abs( new_mu2 - mu2 )))

    return mu1, mu2, Z


class LinearMatrixPencil():
    '''
    Class for regular, symmetric linear matrix pencils of the form H(\lambda) = \lambda A - B
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

    def compute_eig2(self):
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
        pencil_eigenvectors = []
        for k in range(len(self.lambda1)):
            eig, Q = np.linalg.eig( self.value2(self.lambda1[k], self.lambda2[k]) )
            for i in range(len(eig)):
                if np.abs(eig[i]) <= 0.000001:
                    pencil_eigenvectors.append( Q[:,i] )
        pencil_eigenvectors = np.array(pencil_eigenvectors).T

        # num_eig = np.shape(pencil_eigenvectors)[1]
        # if num_eig > 2:
        #     for k in range(num_eig):
        #         L = self.value2(self.lambda1[k], self.lambda2[k])
        #         print( "Eig" + str(k+1) + " = " + str(L @ pencil_eigenvectors[:,k]) )

        # Sorts eigenpairs
        # self.lambda1 = self.lambda1[sorted_args]
        # self.lambda2 = self.lambda2[sorted_args]
        # self.eigenvalues = pencil_eig
        self.eigenvectors = pencil_eigenvectors

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