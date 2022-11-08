import numpy as np
import picos as pc
import sympy as sp
import time 
from quadratic_program import QuadraticProgram
from common import *

class SoSController():
    '''
    Class for the nominal QP controller.
    '''
    def __init__(self, plant, clf, cbf, gamma = 1.0, alpha = 1.0, p = 1.0, epsilon = 1.0 , dt = 0.001):

        # Dimensions and system model initialization
        self._plant = plant
        self._lyapunov = clf
        self._barrier = cbf
        self._state_dim = self._plant.n
        self._ctrl_dim = self._plant.m
        
        # Important monomials
        self.lyapunov_degree = self._lyapunov.get_degree()
        self.lyapunov_symbols = self._lyapunov.get_symbols()
        self.lyapunov_monomials = self._lyapunov.get_monomials()

        self.V_alpha = generate_monomial_list(self._state_dim, self.lyapunov_degree)
        self.V_monomials = generate_monomials_from_symbols(self.lyapunov_symbols, self.lyapunov_degree)
        self.lambda_V_monomials = sp.lambdify( self.lyapunov_symbols, self.V_monomials )

        # PICOS variable
        self._mon_lyap_dim = len(self.lyapunov_monomials)
        self._mon_plant_dim = self._plant._num_monomials

        self._f_dim = self._plant.get_monomial_dimension_f()
        self._g_dim = self._plant.get_monomial_dimension_g()

        self._picos_P = pc.SymmetricVariable("P", self._mon_lyap_dim)
        self._picos_u = pc.RealVariable("u", self._ctrl_dim)

        self.compute_dot_lyapunov()
        self.R_sp = np.zeros([self.dimR,self.dimR])
        self.P = np.zeros([self.dimR,self.dimR])

        # QP parameters
        self.p = p
        self.gamma = gamma
        self._qp_dim = self._ctrl_dim + 1
        P = np.eye(self._qp_dim)
        P[self._ctrl_dim,self._ctrl_dim] = self.p
        q = np.zeros(self._qp_dim)
        self.QP = QuadraticProgram(P=P, q=q)
        self.ctrl_dt = dt

    def compute_dot_lyapunov(self):
        '''
        Computes the time derivative of the Lyapunov function along the system trajectories.
        dV = < l(x,P), p(x,u)>
        '''
        # Computes P-dependent vector polynomial
        lyap_monomials = np.array(self._lyapunov.get_monomials())
        A_list = self._lyapunov.get_matrices()
        self.P_sp = sp.Matrix(sp.symarray( 'p', (self._mon_lyap_dim, self._mon_lyap_dim)))

        P_sp_poly = np.empty(self._state_dim, dtype=object)
        for k in range(self._state_dim):
            P_sp_poly[k] = lyap_monomials.dot( ( self.P_sp @ A_list[k] ) @ lyap_monomials )

        # Computes u-dependent vector polynomial
        plant_monomials = np.array(self._plant.get_monomials())
        G_list = self._plant.get_G()
        self.u_sp = sp.symarray( 'u', self._ctrl_dim )

        u_sp_poly = 0.0
        for k in range(self._g_dim):
            u_sp_poly += plant_monomials[k]*G_list[k] @ self.u_sp

        # Computes dot product
        self.dot_lyapunov_poly = sp.Poly( P_sp_poly.dot( u_sp_poly ).as_expr(), self._plant.get_symbols() )

        self.dV_monoms_alpha = self.dot_lyapunov_poly.monoms()
        max_dV_degree = max(max(self.dV_monoms_alpha))
        if (max_dV_degree % 2) == 0:
            self.dV_alpha = generate_monomial_list(self._state_dim, max_dV_degree/2)
            self.dV_monomials = generate_monomials_from_symbols(self.lyapunov_symbols, max_dV_degree/2)
        else:
            self.dV_alpha = generate_monomial_list(self._state_dim, int((max_dV_degree+1)/2))
            self.dV_monomials = generate_monomials_from_symbols(self.lyapunov_symbols, int((max_dV_degree+1)/2))

        self.lambda_dV_monomials = sp.lambdify( self.lyapunov_symbols, self.dV_monomials )

        self.dimR = len(self.dV_alpha)
        self.symbolic_R = np.empty([self.dimR,self.dimR], dtype=object)
        for i in range(self.dimR):
            for j in range(i,self.dimR):
                exp_coeff = self.dV_alpha[i] + self.dV_alpha[j]
                coeff = self.dot_lyapunov_poly.coeff_monomial(exp_coeff.tolist())
                if i == j:
                    self.symbolic_R[i,j] = coeff
                else:
                    self.symbolic_R[i,j] = (1/2)*coeff
                    self.symbolic_R[j,i] = (1/2)*coeff

        self.lambda_R = sp.lambdify( [self.P_sp, self.u_sp], self.symbolic_R )

    def m(self, x):
        '''
        Compute m(x) vector.
        '''
        return np.array(self.lambda_V_monomials(*x))

    def n(self, x):
        '''
        Compute n(x) vector.
        '''
        return np.array(self.lambda_dV_monomials(*x))

    def V(self, x, P):
        '''
        Compute the lyapunov function V(x,P) = m(x) P m(x)
        '''
        return self.m(x).T @ P @ self.m(x)

    def dV(self, x, P, u):
        '''
        Compute the time derivative of the lyapunov function dV(x,P) = n(x) R(P,u) n(x)
        '''
        R = np.array(self.lambda_R(P.reshape(np.size(P)), u))
        return self.n(x).T @ R @ self.n(x)

    def dotV_SDP_Constraint(self, u):
        '''
        Computes the SDP constraint for dV = n(x)' R n(x)
        '''
        # Substitute the values of u
        dimR = len(self.symbolic_R)
        self.R_sp = np.array(self.lambda_R(self.P_sp, u))

        # Construct the LMI matrices like: [ ] * p_0_0 + ... + [ ] * p_6_6 + ...
        R_dict = dict()
        for i in range(dimR):
            for j in range(i,dimR):
                if hasattr(self.R_sp[i,j], "free_symbols"):
                    for symbol in self.R_sp[i,j].free_symbols:
                        if not (symbol in R_dict.keys()):
                            R_dict[symbol.name] = np.zeros([dimR, dimR])
                        R_dict[symbol.name][i,j] = self.R_sp[i,j].coeff(symbol)
                        R_dict[symbol.name][j,i] = self.R_sp[j,i].coeff(symbol)

        # Finally, construct the LMI constraint in PICOS
        self.dissipativity_lmi = pc.sum( R_dict[symbol]*self._picos_P[int(symbol[-3]),int(symbol[-1])] for symbol in R_dict.keys() )
        
    def solve_SDP(self):
        '''
        Solves the main SDP.
        '''
        sdp = pc.Problem()
        C1 = sdp.add_constraint(self._picos_P >> 0)
        C2 = sdp.add_constraint(self.dissipativity_lmi << 0)

        return sdp

    def get_control(self):
        '''
        Computes the QP control.
        '''
        a_clf, b_clf = self.get_clf_constraint()
        a_cbf, b_cbf = self.get_cbf_constraint()

        # Stacking the CLF and CBF constraints
        A = np.vstack( [a_clf, a_cbf ])
        b = np.array([ b_clf, b_cbf ],dtype=float)

        # Solve QP
        self.QP.set_constraints(A, b)
        QP_sol = self.QP.get_solution()
        control = QP_sol[0:self.control_dim,]

        return control

    def get_clf_constraint(self):
        '''
        Sets the Lyapunov constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Lyapunov function and gradient
        self.V = self.clf.evaluate_function(*state)[0]
        self.nablaV = self.clf.evaluate_gradient(*state)[0]
        
        # Lie derivatives
        self.LfV = self.nablaV.dot(f)
        self.LgV = g.T.dot(self.nablaV)

        # CLF contraint for the QP
        a_clf = np.hstack( [ self.LgV, -1.0 ])
        b_clf = -self.gamma * self.V - self.LfV

        return a_clf, b_clf

    def get_cbf_constraint(self):
        '''
        Sets the barrier constraint.
        '''
        # Affine plant dynamics
        f = self.plant.get_f()
        g = self.plant.get_g()
        state = self.plant.get_state()

        # Barrier function and gradient
        self.h = self.cbf.evaluate_function(*state)[0]
        self.nablah = self.cbf.evaluate_gradient(*state)[0]

        self.Lfh = self.nablah.dot(f)
        self.Lgh = g.T.dot(self.nablah)

        # CBF contraint for the QP
        a_cbf = -np.hstack( [ self.Lgh, 0.0 ])
        b_cbf = self.alpha * self.h + self.Lfh

        return a_cbf, b_cbf

    def max_constrained_curvature(self, P, v):
        '''
        Given a n x n matrix P and a n-dimensional vector v, compute max z' P z s.t. z'v = 0 and z'z = 1.
        '''
        dimP = np.shape(P)[0]
        normalized_v = v/np.linalg.norm(v)
        _, V = np.linalg.eig(P)

        values = []
        for i in range(dimP):
            values.append( v.dot(V[:,i]) )
        index_to_delete = np.argmin(values)
        np.delete(V,index_to_delete,1)      # discart one of the columns

        M = np.hstack([normalized_v, V])
        Q, R = np.linalg.qr(M)              # columns of Q are vectors from Gram Schmidt orthogonalization

        Pbar = Q.T @ P @ Q
        sol_eigs, _ = np.linalg.eig(Pbar[1:,1:])

        return np.max(sol_eigs)