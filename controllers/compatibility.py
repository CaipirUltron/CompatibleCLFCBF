import scipy
import numpy as np

from scipy import signal
from scipy.optimize import fsolve
from scipy.linalg import null_space
from numpy.polynomial import polynomial as poly

ZERO_ACCURACY = 0.0000000001

class PolynomialCLFCBFPair():
    '''
    Class for polynomial CLF-CBF pairs of the form:
    V(x,P) = m(x) P m(x) and h(x,Q) = m(x) Q m(x) - 1.
    In this initial implementation, the pair is represented by their respective shape matrices P and Q.
    '''
    def __init__(self, P, Q, max_iter = 1000):
        self.update(P = P, Q = Q, max_iter = max_iter)

    def update(self, **kwargs):
        '''
        Updates the CLF-CBF pair.
        '''
        for key in kwargs:
            if key == "P":
                self.P = kwargs[key]
                continue
            if key == "Q":
                self.Q = kwargs[key]
                continue
            if key == "max_iter":
                self.max_iter = kwargs[key]
                continue

        self.pencil = LinearMatrixPencil2( self.Q, self.P )
        self.n = self.pencil.dim-1

        self.C = scipy.linalg.block_diag(np.zeros([self.n,self.n]), 1) # C matrix for PEP

        self.asymptotes = self.compute_asymptotes()
        self.lambdas, self.kappas, self.equilibria, self.initial_lines = self.compute_equilibrium()

    def compute_asymptotes(self):
        '''
        Computes the asymptotes of the graph det( lambda Q - kappa C - P ) = 0.
        Returns a dict whose keys are the angular coefficients of the asymptotes.
        The values for each key are: the associated linear coefficient, in case that angular coefficient (key) is finite;
                                     the associated horizontal position of the asymptote, in case the angular coefficient is +-inf (vertical asymptote).
        '''
        # Compute angular coefficients of the asymptotes
        pencil_angular_coefs = LinearMatrixPencil2(self.Q, self.C)
        angular_coefs = pencil_angular_coefs.eigenvalues
        asymptotes = { angular_coef: [] for angular_coef in angular_coefs }

        # Compute linear coefficients of the asymptotes
        sorted_eigenvalues = np.sort(self.pencil.eigenvalues)
        to_be_deleted = []
        for i in range(len(sorted_eigenvalues)):
            if np.abs(sorted_eigenvalues[i]) == np.inf:
                to_be_deleted.append(i)
        sorted_eigenvalues = np.delete(sorted_eigenvalues, to_be_deleted)
        differences = np.diff(sorted_eigenvalues)
        '''
        Define initializers for the algorithm.
        If +-inf eigenvalues were found, bound the initiliazers to be inside the limits of the finite spectra: important to prevent singularities.
        '''
        initializers = []
        initializers.append( sorted_eigenvalues[0] - differences[0]/2 )
        for k in range(len(differences)):
            initializers.append( sorted_eigenvalues[k] + differences[k]/2 )
        initializers.append( sorted_eigenvalues[-1] + differences[-1]/2 )

        for i in range(len(angular_coefs)):
            if np.abs(angular_coefs[i]) == np.inf:
                null_space_Q = null_space(self.Q).reshape(self.n+1)
                sol = - (null_space_Q @ self.P @ null_space_Q) / (null_space_Q @ self.C @ null_space_Q)
                asymptotes[angular_coefs[i]].append(sol)
                continue
            def compute_trace(s):
                    invPencil = np.linalg.inv(self.pencil.value(s))
                    return np.trace( invPencil @ ( angular_coefs[i]*self.Q - self.C ) )
            for k in range(len(initializers)):
                sols, infodict, ier, mesg = fsolve( compute_trace, initializers[k], factor=0.1, full_output = True )
                if ier == 1:
                    for sol in sols:
                        if np.any( np.abs(asymptotes[angular_coefs[i]] - sol) < 0.00001 ):
                            continue
                        asymptotes[angular_coefs[i]].append(sol)

        return asymptotes

    def compute_equilibrium(self):
        '''
        Computes all equilibrium points of the CLF-CBF pair, using the Parametric Eigenvalue Problem (PEP)
        '''
        # Separate horizontal from non-horizontal asymptotes.
        # Non-horizontal asymptotes are represented by equation \kappa = m \lambda + p
        lambda_positions = []
        non_horizontal_lines = []
        for key in self.asymptotes.keys():
            if np.abs(key) < ZERO_ACCURACY:
                for p in self.asymptotes[key]:
                    if np.any( np.abs(lambda_positions - p) < ZERO_ACCURACY ):
                        continue
                    lambda_positions.append(p)
            else:
                if np.abs(key) == np.inf:
                    non_horizontal = [ ( 0.0,lin_coef ) for lin_coef in self.asymptotes[key] ]
                else:
                    non_horizontal = [ ( 1/key, -lin_coef/key ) for lin_coef in self.asymptotes[key] ]
                non_horizontal_lines = non_horizontal_lines + non_horizontal

        # Compute intersections with non-horizontal asymptotes
        intersection_pts = np.array([], dtype=float).reshape(2,0)
        for k in range(len(lambda_positions)):
            vert_pos = lambda_positions[k]
            for non_horizontal in non_horizontal_lines:
                m, p = non_horizontal[0], non_horizontal[1]
                kappa_val, lambda_val = m*vert_pos + p, vert_pos
                pt = np.array([kappa_val, lambda_val]).reshape(2,1)
                intersection_pts = np.hstack([intersection_pts, pt])
        diffs = np.diff(intersection_pts)
        # diffs = np.hstack([diffs, diffs[:,-1].reshape(2,1)])

        # Compute random initial points
        # def generate_pts(center, R, n):
        #     '''
        #     Generates n random points inside a circle of radius R centered on center,
        #     filtering points with negative y-value.
        #     '''
        #     pts = np.array([], dtype=float).reshape(2,0)
        #     for _ in range(n):
        #         pt = np.random.normal( center, R, 2 )
        #         # if pt[1]>=0:
        #         pts = np.hstack([pts, pt.reshape(2,1)])
        #     return pts

        # Compute random initial points
        # num_internal_points = 10
        # intermediate_pts = np.array([], dtype=float).reshape(2,0)
        # for k in range(np.shape(diffs)[1]):
        #     for i in range(num_internal_points):
        #         d = np.linalg.norm(diffs[:,k])
        #         vert_pt = intersection_pts[:,k] + i*diffs[:,k]/(num_internal_points)
        #         # hor_pt = intersection_pts[:,k] + (-num_internal_points/2 + i)*np.array([d, 0.0])/(num_internal_points)
        #         intermediate_pts = np.hstack([intermediate_pts, vert_pt.reshape(2,1)])

        # Compute intermediary points for defining initial lines
        num_internal_lines = 1
        intermediate_pts = np.array([], dtype=float).reshape(2,0)
        first_pt = intersection_pts[:,0] - diffs[:,0]/2
        intermediate_pts = np.hstack([intermediate_pts, first_pt.reshape(2,1)])
        for k in range(diffs.shape[1]):
            for i in range(1,num_internal_lines+1):
                pt = intersection_pts[:,k] + i*diffs[:,k]/(num_internal_lines+1)
                intermediate_pts = np.hstack([intermediate_pts, pt.reshape(2,1)])
        last_pt = intersection_pts[:,-1] + diffs[:,-1]/2
        intermediate_pts = np.hstack([intermediate_pts, last_pt.reshape(2,1)])

        # Compute the initial lines
        init_lines = []
        for pt in intermediate_pts.T:
            m = -0.1
            p = pt[1] - m*pt[0]
            init_lines.append( { "angular_coef": m, "linear_coef" : p } )

        # Solves the PEP problem for many different initial lines and store non-repeating results
        lambdas, kappas, equilibrium_points = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float).reshape(self.n,0)

        # lambda_p, kappa_p, Z, init_kappas, init_lambdas = solve_PEP( self.Q, self.P, initial_points = intermediate_pts, max_iter = self.max_iter )
        lambda_p, kappa_p, Z, init_kappas, init_lambdas = solve_PEP( self.Q, self.P, initial_lines = init_lines, max_iter = self.max_iter )
        for i in range(len(lambda_p)):
            equal_lambda = np.any( np.abs( lambda_p[i] - lambdas ) < ZERO_ACCURACY )
            # equal_kappa = np.any( np.abs( kappa_p[i] - kappas ) < ZERO_ACCURACY )
            eq = Z[0:-1,i].reshape(self.n,1)
            equal_eigenvec = np.any( np.linalg.norm( eq - equilibrium_points, axis=0 ) < ZERO_ACCURACY )
            if equal_lambda and equal_eigenvec:
                continue
            lambdas = np.hstack([lambdas, lambda_p[i]])
            kappas = np.hstack([kappas, kappa_p[i]])
            equilibrium_points = np.hstack([equilibrium_points, eq])

        # init_pts = np.vstack([init_kappas, init_lambdas])

        return lambdas, kappas, equilibrium_points, init_lines

class LinearMatrixPencil2():
    '''
    Class for regular, symmetric linear matrix pencils of the form P(\lambda) = \lambda A - B, where A and B are p.s.d. matrices.
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
        self.schurA_vec = np.diag(schurA)
        self.schurB_vec = np.diag(schurB)

        self.eigenvalues = np.zeros(self.dim)
        for k in range(self.dim):
            if np.abs(self.schurB_vec[k]) > ZERO_ACCURACY:
                self.eigenvalues[k] = self.schurA_vec[k]/self.schurB_vec[k]
            else:
                self.eigenvalues[k] = np.sign(self.schurA_vec[k]) * np.inf

        # Compute the (normalized, if possible) pencil eigenvectors
        self.eigenvectors = np.zeros([self.dim,self.dim])
        for k in range(len(self.eigenvalues)):
            if np.abs(self.eigenvalues[k]) != np.inf:
                eig, Q = np.linalg.eig( self.value( self.eigenvalues[k]) )
            else:
                eig, Q = np.linalg.eig( self.schurA_vec[k] * self._A - self.schurB_vec[k] * self._B )
            for i in range(len(eig)):
                if np.abs(eig[i]) <= ZERO_ACCURACY:
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

    def __str__(self):
        '''
        Print the given pencil.
        '''
        np.set_printoptions(precision=3, suppress=True)
        ret_str = '{}'.format(type(self).__name__) + " = \u03BB A - B \n"
        ret_str = ret_str + 'A = ' + self._A.__str__() + '\n'
        ret_str = ret_str + 'B = ' + self._B.__str__()
        return ret_str

class LinearMatrixPencil():
    '''
    Class for regular linear matrix pencils of the form P(λ) = λ A - B or P(l1,l2) = l1 A - l2 B : λ = l1/l2
    '''
    def __init__(self, A, B, **kwargs):
        self._shapeA = None
        self._shapeB = None
        self.set_pencil(A=A, B=B)
        self._lambda = 0.0
        if "parameter" in kwargs.keys():
            self._lambda = kwargs["parameter"]

    def set_pencil(self, **kwargs):
        '''
        Sets pencil attributes and computes pencil spectra
        '''
        for key in kwargs.keys():
            aux_key = key.lower()
            if aux_key == "a":
                self._A = kwargs[key]
                self._shapeA = self._A.shape
                continue
            if aux_key == "b":
                self._B = kwargs[key]
                self._shapeB = self._B.shape
                continue
        
        if self._shapeA != self._shapeB:
            raise Exception("Matrix dimensions are not equal.")
        if (self._shapeA[0] != self._shapeA[1]) or (self._shapeB[0] != self._shapeB[1]):
            raise Exception("Matrices are not square.")
        self.dim = self._shapeA[0]

        self.compute_eig()

    def value(self, lambda_param):
        '''
        Returns pencil value.
        '''
        return self.value_double( lambda_param, 1.0 )

    def value_double(self, lambda1_param, lambda2_param):
        '''
        Returns pencil value.
        '''
        return lambda1_param * self._A - lambda2_param * self._B

    def compute_eig(self):
        '''
        This method solves the generalized eigenvalue problem for matrices A,B
        '''
        # Schur decomposition of A and B
        schurA, schurB, _, _, Q, Z = scipy.linalg.ordqz( self._A, self._B )

        # Compute the corresponding pencil eigenvalues from the Schur decomposition
        lambda1, lambda2, pencil_eigs = [], [], []
        k = 0
        while k <= self.dim-1:
            
            if k == self.dim-1:
                lambda1.append(schurB[k,k])
                lambda2.append(schurA[k,k])
                pencil_eigs.append(schurB[k,k]/schurA[k,k])
                k+=1
                continue

            if schurA[k+1,k] != 0:
                # Hessenberg block - compute the double roots directly
                blockA = schurA[k:k+2,k:k+2]
                blockB = schurB[k:k+2,k:k+2]

                polynomial = poly.polysub( poly.polymul([-blockB[0,0], blockA[0,0] ], [-blockB[1,1], blockA[1,1] ]), [0.0, 0.0, blockA[1,0]*blockA[0,1] ] )
                pencil_eigs += poly.polyroots(polynomial).tolist()
                lambda1 += [ None, None ]
                lambda2 += [ None, None ]
                k+=2
            else:
                lambda1.append(schurB[k,k])
                lambda2.append(schurA[k,k])
                pencil_eigs.append(schurB[k,k]/schurA[k,k])
                k+=1

        # Finds sorting order
        # sorted_args = np.argsort(pencil_eigs)

        # Compute the corresponding pencil eigenvectors
        pencil_eigenvectors, zQzs = [], []
        for k in range(len(pencil_eigs)):

            if lambda1[k] != None:
                P = self.value_double(lambda1[k], lambda2[k])   # more accurate
            else:
                P = self.value(pencil_eigs[k])

            N = null_space(P)

            n = N[:,0]
            if np.isreal(n.T @ self._A @ n) and n.T @ self._A @ n > 1e-10:
                n = n.real
                normalization_const = 1.0 / np.sqrt(n.T @ self._A @ n)
                if n[0] < 0:
                    n = -n
                pencil_eigenvectors.append( normalization_const * n )
            else:
                pencil_eigenvectors.append(n)
            zQzs.append(pencil_eigenvectors[-1].T @ self._A @ pencil_eigenvectors[-1] )
        
        # Assumption: B is invertible => detB != 0
        # detB = np.linalg.det(self._B)
        # if detB == 0:
        #     raise Exception("B is rank deficient.")

        # Computes the pencil characteristic polynomial and denominator of f(\lambda)
        # pencil_det = np.real(np.prod(pencil_eigs))
        # self.characteristic_poly = ( detB/pencil_det ) * np.real(np.polynomial.polynomial.polyfromroots(pencil_eigs))

        # Sorts eigenpairs
        self.eigenvalues = pencil_eigs
        self.eigenvectors = pencil_eigenvectors
        self.zQzs = zQzs

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