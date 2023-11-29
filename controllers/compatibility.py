import time
import scipy
import numpy as np
import cvxpy as cp

from scipy import signal
from scipy.optimize import fsolve, root, least_squares, minimize
from scipy.linalg import null_space
from quadratic_program import QuadraticProgram
from dynamic_systems import Integrator
from common import adjugate

ZERO_ACCURACY = 0.0000000001

def compute_stability(plant, clf, cbf, eq_sol, **kwargs):
    '''
    Compute the equilibrium conditions for a given equilibrium point (only valid in R2)
    '''
    c = 1
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()

    # Optimization
    ACCURACY = 0.000000000001

    Jm = kernel.jacobian( eq_sol["point"] )
    z = kernel.function( eq_sol["point"] )

    nablah = Jm.T @ Q @ z

    def S_matrix(eq_sol):
        kappas = eq_sol["kappas"]        
        sum = np.zeros([p,p])
        for k in range(p):
            sum += kappas[k] * N_list[k]
        lambda0 = c * clf.function( eq_sol["point"] )

        return F + eq_sol["lambda"]* Q - lambda0 * P - sum - c * np.outer(P @ z, P @ z)

    v = np.array([ nablah[1], -nablah[0] ])

    return v.T @ Jm.T @ S_matrix( eq_sol ) @ Jm @ v

def compute_equilibria_algorithm1(F, clf, cbf, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "initial":
            initial_guess = kwargs[key]

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim

    # A_list = clf.kernel.get_A_matrices()
    N_list = kernel.get_N_matrices()

    def linear_pencil(linear_combs):
        '''
        linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
        Computes the linear matrix pencil.
        '''
        L = F + linear_combs[1] * Q - linear_combs[0] * P
        for k in range(2, p - n):
            L += linear_combs[k] * N_list[k]
        return L

    def linear_pencil_det(linear_combs):
        '''
        linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
        Computes determinant of linear matrix pencil.
        '''
        return np.linalg.det( linear_pencil(linear_combs) )

    def compute_F(solution):
        '''
        This inner method computes the vector field F(lambda, kappa, z) and returns its value.
        '''
        linear_combs, z = solution[0:p-n+2], solution[p-n+2:]
        kernel_constraints = kernel.get_constraints(z)
        num_kernel_constraints = len(kernel_constraints)

        F = np.zeros(p+num_kernel_constraints+2)
        L = linear_pencil(linear_combs)
        F[0:p] = L @ z
        V = 0.5 * z.T @ P @ z
        F[p] = linear_combs[0] - c*V
        F[p+1] = linear_pencil_det(linear_combs)
        F[p+2:] = kernel_constraints

        return F

    # t = time.time()
    solution = root(compute_F, initial_guess, method='lm', tol = 0.00001)
    # solution = fsolve(compute_F, initial_guess, maxfev = max_iter)
    # elapsed = time.time() - t

    l1 = solution.x[0]
    l2 = solution.x[1]
    kappas = solution.x[2:p-n+2]
    z = solution.x[p-n+2:]

    equilibrium_point = np.flip(z[1:n+1]).tolist()

    return equilibrium_point

def compute_equilibria_algorithm2(F, clf, cbf, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "initial":
            initial_guess = kwargs[key]

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim

    # A_list = clf.kernel.get_A_matrices()
    N_list = kernel.get_N_matrices()

    # Optimization
    import cvxpy as cp

    # General optimization problem

    # ----- Variables ------
    delta_var = cp.Variable()
    l1_var = cp.Variable()
    l2_var = cp.Variable()
    kappa_var = cp.Variable(p-n)
    z_var = cp.Variable(p)
    L_var = cp.Variable((p,p))

    # ----- Prob definition ------
    L_var = F + l1_var * Q - l2_var * P
    for k in range(p-n):
        L_var += kappa_var[k] * N_list[k]

    matrices = kernel.get_matrix_constraints()
    kernel_constraints = [ z_var[0] == 1 ]
    kernel_constraints += [ z_var.T @ matrices[k] @ z_var == 0 for k in range(len(matrices)) ]

    objective = cp.Minimize( cp.norm(delta_var)**2 )
    constraints = [ L_var @ z_var == 0,
                    l1_var == 0.5 * c * z_var.T @ P @ z_var,
                    l2_var >= delta_var,
                    delta_var >= 0 ]
    constraints += kernel_constraints
    problem = cp.Problem(objective, constraints)

    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % problem.value)
        for variable in problem.variables():
            print("Variable %s: value %s" % (variable.name(), variable.value))

    # CONCLUSION: problem is fundamentally nonconvex. Cannot be solved through CVXPY.

def compute_equilibria_algorithm3(F, clf, cbf, initial_point, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    c = 1
    l2_bound = 1.0
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "l2_bound":
            l2_bound = kwargs[key]
            continue
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim

    # A_list = clf.kernel.get_A_matrices()
    N_list = kernel.get_N_matrices()

    # Optimization
    import cvxpy as cp

    ACCURACY = 0.0001

    it = 0
    z_old = np.inf
    z = kernel.function( initial_point )
    while np.linalg.norm( z - z_old ) > ACCURACY or it < max_iter:
        it += 1

        # First optimization: QP

        # ----- Variables ------
        delta_var = cp.Variable()
        l1_var = cp.Variable()
        l2_var = cp.Variable()
        kappa_var = cp.Variable(p-n)
        L_var = cp.Variable((p,p))

        # ----- Parameters ------
        z_param = cp.Parameter(p)
        l1_param = cp.Variable()

        # ----- Prob definition ------
        l1_param = 0.5 * c * z_param.T @ P @ z_param

        L_var = F + l1_var * Q - l2_var * P
        for k in range(p-n):
            L_var += kappa_var[k] * N_list[k]

        QP_objective = cp.Minimize( cp.norm(l2_var)**2 )
        # QP_objective = cp.Minimize( cp.norm(l2_var)**2 + cp.norm(kappa_var)**2 )
        QP_constraints = [ L_var @ z_param == 0 ,
                           l1_var == 0.5 * c * z_param.T @ P @ z_param ,
                        #    l2_var >= delta_var ,
                           l2_var >= l2_bound ]
        QP = cp.Problem(QP_objective, QP_constraints)

        # ------- Solve QP --------
        z_param.value = z
        QP.solve()

        p_var = np.zeros(p-n+2)
        p_var[0] = l1_var.value
        p_var[1] = l2_var.value
        p_var[2:] = kappa_var.value

        if QP.status in ["infeasible", "unbounded"]:
            raise Exception("QP is " + QP.status)

        # Second problem: nonlinear Newton-Raphson method

        def linear_pencil(p_var):
            '''
            linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
            Computes the linear matrix pencil.
            '''
            L = F + p_var[1] * Q - p_var[0] * P
            for k in range(2, p - n):
                L += p_var[k] * N_list[k]
            return L

        def linear_pencil_det(p_var):
            '''
            linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
            Computes determinant of linear matrix pencil.
            '''
            return np.linalg.det( linear_pencil(p_var) )

        def compute_F(z):
            '''
            This inner method computes the vector field F(sol) and returns its value.
            '''
            kernel_constraints = kernel.get_constraints(z)
            num_kernel_constraints = len(kernel_constraints)

            F = np.zeros(p+2+num_kernel_constraints)
            L = linear_pencil(p_var)
            F[0:p] = L @ z
            F[p] = p_var[0] - 0.5 * c * z.T @ P @ z
            F[p+1] = np.eye(p)[0,:].T @ z - 1
            F[p+2:] = kernel_constraints

            return F

        z_old = z
        solution = root(compute_F, z, method='lm', tol = ACCURACY)
        z = solution.x

    equilibrium_point = np.flip(z[1:n+1]).tolist()
    sol_dict = { "equilibrium_point": equilibrium_point,
                 "lambda1": l1_var.value.tolist(),
                 "lambda2": l2_var.value.tolist(),
                 "kappas": kappa_var.value.tolist(),
                 "z": z.tolist() }

    return sol_dict

def compute_equilibria_algorithm4(plant, clf, cbf, initial_point, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    c = 1
    max_iter = 100
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()

    g = plant.get_g()
    nablaV = clf.get_gradient()
    nablah = cbf.get_gradient()
    # initial_l2 = nablaV.T @ g @ g.T @ nablaV / nablah.T @ g @ g.T @ nablah
    initial_l2 = 1.0

    print("Initial lambda2 = " + str(initial_l2))

    # Optimization
    import cvxpy as cp
    from scipy.optimize import least_squares

    ACCURACY = 0.000000001

    it = 0
    z_old = np.inf
    z = kernel.function( initial_point )

    cost = np.linalg.norm( z - z_old )
    while cost > ACCURACY and it < max_iter:
        it += 1

        # First optimization: least squares with bound on lambda2
        def linear_pencil(p_var):
            '''
            p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
            Computes the linear matrix pencil.
            '''
            L = F + p_var[1] * Q - p_var[0] * P
            for k in range(p - n):
                L += p_var[k+2] * N_list[k]
            return L

        def linear_pencil_det(p_var):
            '''
            p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
            Computes determinant of linear matrix pencil.
            '''
            return np.linalg.det( linear_pencil(p_var) )

        def F1(p_var):
            '''
            Function for constrained least squares
            '''
            return np.array([ linear_pencil_det(p_var), p_var[0] - 0.5 * c * z.T @ P @ z ])

        # ------- Solve constrained least squares --------
        initial_p_var = np.hstack([ 0.5 * c * z.T @ P @ z, initial_l2, np.zeros(p-n) ])
        lower_bounds = [ -np.inf, 0.0 ] + [ -np.inf for _ in range(p-n) ]
        solution1 = least_squares( F1, initial_p_var, bounds=(lower_bounds, np.inf) )

        p_var = np.zeros(p-n+2)
        p_var[0] = solution1.x[0]
        p_var[1] = solution1.x[1]
        p_var[2:] = solution1.x[2:]

        print("p = " + str(p_var))
        print("F1 = " + str(F1(p_var)))

        # Second problem: nonlinear Newton-Raphson method

        def F2(z):
            '''
            This inner method computes the vector field F(sol) and returns its value.
            '''
            kernel_constraints = kernel.get_constraints(z)
            num_kernel_constraints = len(kernel_constraints)

            F2 = np.zeros(p+1+num_kernel_constraints)
            L = linear_pencil(p_var)
            F2[0:p] = L @ z
            F2[p] = p_var[0] - 0.5 * c * z.T @ P @ z
            F2[p+1:] = kernel_constraints

            return F2

        z_old = z
        solution2 = root(F2, z_old, method='lm', tol = ACCURACY)
        z = solution2.x
        cost = np.linalg.norm( z - z_old )

        # print( kernel.get_constraints(z) )
        # print( kernel.is_in_kernel_space(np.array(z)) )
        # print(cost)

    equilibrium_point = np.flip(z[1:n+1]).tolist()
    sol_dict = { "equilibrium_point": equilibrium_point,
                 "lambda1": p_var[0],
                 "lambda2": p_var[1],
                 "kappas": p_var[2:].tolist(),
                 "z": z.tolist(),
                 "cost": cost }

    return sol_dict

def compute_equilibria_algorithm5(plant, clf, cbf, initial_point, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    c = 1
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()

    # Optimization

    ACCURACY = 0.000001

    z = kernel.function( initial_point )

    initial_l1 = 0.5 * c * z.T @ P @ z
    initial_l2 = 0.0
    initial_kappas = np.zeros(p-n)

    p_var = np.hstack([ initial_l1, initial_l2, initial_kappas ])

    # Least squares with bound on lambda2
    def linear_pencil(p_var):
        '''
        p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
        Computes the linear matrix pencil.
        '''
        L = F + p_var[1] * Q - p_var[0] * P
        for k in range(p - n):
            L += p_var[k+2] * N_list[k]
        return L

    def linear_pencil_det(p_var):
        '''
        p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
        Computes determinant of linear matrix pencil.
        '''
        return np.linalg.det( linear_pencil(p_var) )

    def boundary_equilibria(variables):
        '''
        Constraints for boundary equilibrium points
        '''
        z = variables[0:p]
        p_var = variables[p:]
        kernel_constraints = kernel.get_constraints(z)

        return np.hstack([ linear_pencil_det(p_var),
                           linear_pencil(p_var) @ z,
                           p_var[0] - 0.5 * c * z.T @ P @ z,
                           z.T @ Q @ z - 1,
                           kernel_constraints ])

    def interior_equilibria(variables):
        '''
        Constraints for interior equilibrium points
        '''
        z = variables[0:p]
        p_var = variables[p:]
        kernel_constraints = kernel.get_constraints(z)

        return np.hstack([ linear_pencil_det(p_var),
                           linear_pencil(p_var) @ z,
                           p_var[0] - 0.5 * c * z.T @ P @ z,
                           p_var[1],
                           kernel_constraints ])

    # ------- Solve constrained least squares --------
    initial_variables = np.hstack([ z, p_var ])
    lower_bounds = [ -np.inf for _ in range(p) ] + [ -np.inf, 0.0 ] + [ -np.inf for _ in range(p-n) ] # forces inequality lambda2 >= 0

    # Solves for boundary equilibrium points
    boundary_z = [ None for _ in range(p) ]
    boundary_p_var = np.array([ None for _ in range(p-n+2) ])
    boundary_equilibrium = [ None for _ in range(n) ]
    boundary_cost = None

    t = time.time()
    boundary_solution = least_squares( boundary_equilibria, initial_variables, bounds=(lower_bounds, np.inf), xtol = ACCURACY )
    boundary_delta = time.time() - t

    z = boundary_solution.x[0:p]
    if np.abs( z.T @ Q @ z - 1 ) < ACCURACY:
        boundary_z = boundary_solution.x[0:p]
        boundary_p_var = boundary_solution.x[p:]
        boundary_equilibrium = kernel.kernel2state(boundary_z).tolist()
        boundary_cost = boundary_solution.cost

    interior_z = [ None for _ in range(p) ]
    interior_p_var = np.array([ None for _ in range(p-n+2) ])
    interior_equilibrium = [ None for _ in range(n) ]
    interior_cost = None
    interior_delta = 0.0

    t = time.time()
    interior_solution = least_squares( interior_equilibria, initial_variables, bounds=(lower_bounds, np.inf), xtol = ACCURACY )
    interior_delta = time.time() - t

    if interior_solution.cost < ACCURACY:
        interior_z = interior_solution.x[0:p]
        interior_p_var = interior_solution.x[p:]
        interior_equilibrium = kernel.kernel2state(boundary_z).tolist()
        interior_cost = interior_solution.cost

    boundary_sol = { "equilibrium": boundary_equilibrium,
                     "z": boundary_z,
                     "lambda1": boundary_p_var[0],
                     "lambda2": boundary_p_var[1],
                     "kappas": boundary_p_var[2:].tolist(),
                     "cost": boundary_cost,
                     "time": boundary_delta }

    interior_sol = { "equilibrium": interior_equilibrium,
                     "z": interior_z,
                     "lambda1": interior_p_var[0],
                     "lambda2": interior_p_var[1],
                     "kappas": interior_p_var[2:].tolist(),
                     "cost": interior_cost,
                     "time": interior_delta }

    return { "boundary": boundary_sol, "interior": interior_sol }

def compute_equilibria_algorithm6(plant, clf, cbf, initial_points, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    c = 1
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()
    r = len(N_list)

    # matrices = kernel.get_matrix_constraints()
    # num_matrices = len(matrices)

    # Optimization
    ACCURACY = 0.0000001

    # ----------------------- Auxiliary functions ------------------
    def commutator(A,B,z):
        ZZt = np.outer(z,z)
        return A @ ZZt @ B - B @ ZZt @ A

    # def linear_pencil(vars):
    #     z = kernel.function(vars[0:n])
    #     kappas = vars[n:n+p]
    #     sum = np.zeros([p,p])
    #     for k in range(p):
    #         sum += kappas[k] * commutator(Q,N_list[k],z)
    #     L = 0.5 * c * commutator( commutator(Q,P,z), P, z) - commutator(Q,F,z) - sum
    #     return L

    def lambda2_matrix(z, kappas):
        sum = np.zeros([p,p])
        for k in range(p):
            sum += kappas[k] * N_list[k]
        return 0.5 * c * np.outer(P @ z, P @ z) - F - sum

    def linear_pencil(vars):
        z = kernel.function(vars[0:n])
        kappas = vars[n:n+p]
        l2 = vars[-1]
        return lambda2_matrix(z, kappas) - l2 * Q

    # def lambda2_fun(z, kappas):
    #     return z.T @ lambda2_matrix(z, kappas) @ z

    def projection(z):
        return np.eye(len(z)) - np.outer(z, Q @ z)

    # ------------------- Constraints on vars = [ x, kappas, l2 ], kappas is p-dimensional ----------------------

    def detL_eq(vars):
        return np.linalg.det( linear_pencil(vars) )

    def Lz_eq(vars):
        z = kernel.function(vars[0:n])
        return linear_pencil(vars) @ z

    # def lambda2_eq(vars):
    #     z = kernel.function(vars[0:n])
    #     kappas = vars[n:n+p]
    #     l2 = vars[-1]
    #     return l2 - lambda2_fun(z, kappas)

    # def lambda2_ineq_constraint(vars):
    #     l2 = vars[-1]
    #     return l2 # >> 0

    # def boundary_ineq(vars):
    #     z = kernel.function(vars[0:n])
    #     return z.T @ Q @ z - 1 # >> 0

    def boundary_eq(vars):
        z = kernel.function(vars[0:n])
        return z.T @ Q @ z - 1 # >= 0

    # def lambda2_ineq(vars):
    #     z = kernel.function(vars[0:n])
    #     kappas = vars[n:n+p]
    #     return z.T @ lambda2_matrix(z, kappas) @ z

    def lambda2_ineq(vars):
        return vars[-1]

    def kappa_eq(vars):
        z = kernel.function(vars[0:n])
        kappas = vars[n:n+p]
        sum = 0.0
        for k in range(p):
            sum += kappas[k] * z.T @ ( N_list[k] -  N_list[k].T ) @ z
        return sum

    def complementary_slackness_eq(vars):
        z = kernel.function(vars[0:n])
        kappas = vars[n:n+p]
        return z.T @ lambda2_matrix(z, kappas) @ projection(z) @ z
        # return lambda2_ineq_constraint(vars) * boundary_ineq_constraint(vars)

    eq_constr0 = {'type': 'eq', 'fun': detL_eq}
    eq_constr1 = {'type': 'eq', 'fun': Lz_eq}
    eq_constr2 = {'type': 'eq', 'fun': boundary_eq}
    eq_constr3 = {'type': 'eq', 'fun': kappa_eq}

    ineq_constr1 = {'type': 'ineq', 'fun': lambda2_ineq}
    # ineq_constr2 = {'type': 'ineq', 'fun': boundary_ineq}

    # Initialize with boundary points
    # boundary_pts = get_boundary_points( cbf, initial_points )
    num_pts = np.shape(initial_points)[0]

    # Try minimization
    solutions = {"points": [], "lambda2": [], "indexes": []}
    error_counter = 0
    for k in range(num_pts):

        x_n = initial_points[k,:]
        def objective(vars):
            x = vars[0:n]
            return np.linalg.norm(x - x_n)**2

        init_z = kernel.function(x_n)
        a = np.zeros([1,p])
        for i in range(p):
            a[:,i] = init_z.T @ ( N_list[i] -  N_list[i].T ) @ init_z

        init_kappas = np.zeros(p)
        for col in null_space(a).T:
            init_kappas += np.random.randn() * col

        init_l2 = init_z.T @ lambda2_matrix(init_z, init_kappas) @ init_z
        initial_vars = np.hstack([ x_n, init_kappas, init_l2 ])

        # Solve optimization problem
        min_sol = minimize(objective, initial_vars, method='trust-constr', constraints=[ eq_constr1, eq_constr2, eq_constr3, ineq_constr1 ])

        if min_sol.success:
            solutions["points"].append( min_sol.x[0:n] )
            solutions["lambda2"].append( min_sol.x[-1] )
            solutions["indexes"].append( k )
        else:
            error_counter += 1
            print(min_sol.message)

    return solutions

def compute_equilibria_algorithm7(plant, clf, cbf, initial_point, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    Returns:                array with boundary equilibrium point solutions
    '''
    c = 1
    max_iter = 10
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()
    q = len(N_list)

    # Optimization
    ACCURACY = 0.000000000001

    # vars = [ x, kappas, l2 ]

    # def commutator(A,B,z):
    #     ZZt = np.outer(z,z)
    #     return A @ ZZt @ B - B @ ZZt @ A

    # def linear_pencil(vars):
    #     z = kernel.function(vars[0:n])
    #     kappas = vars[n:p]
    #     sum = np.zeros([p,p])
    #     for k in range(len(N_list)):
    #         sum += kappas[k] * commutator(Q,N_list[k],z)
    #     L = 0.5 * c * commutator( commutator(Q,P,z), P, z) - commutator(Q,F,z) - sum
    #     return L

    def lambda2_matrix(z, kappas):
        sum = np.zeros([p,p])
        for k in range(q):
            sum += kappas[k] * N_list[k]
        L = 0.5 * c * np.outer(P @ z, P @ z) - F - sum
        return L

    def linear_pencil(vars):
        z = kernel.function(vars[0:n])
        kappas = vars[n:n+q]
        l2 = vars[-1]
        return lambda2_matrix(z, kappas) - l2 * Q

    def Lz_constraint(vars):
        z = kernel.function(vars[0:n])
        return linear_pencil(vars) @ z

    def lambda2_fun(z, kappas):
        # sum = np.zeros([p,p])
        # for k in range(p):
        #     sum += kappas[k] * N_list[k]
        return z.T @ lambda2_matrix(z, kappas) @ z

    def lambda2_constraint(vars):
        z = kernel.function(vars[0:n])
        kappas = vars[n:n+q]
        lambda2 = vars[-1]
        return lambda2 - lambda2_fun(z, kappas)

    # def kappa_constraint(vars):
    #     z = kernel.function(vars[0:n])
    #     kappas = vars[n:n+p]
    #     sum = 0.0
    #     for k in range(p):
    #         sum += kappas[k] * z.T @ ( N_list[k] -  N_list[k].T ) @ z
    #     return sum

    def boundary_constraint(vars):
        z = kernel.function(vars[0:n])
        return z.T @ Q @ z - 1 # >= 0

    def detL(vars):
        detL = np.linalg.det( linear_pencil(vars) )
        return detL

    def objective(vars):
        return np.hstack([
                           Lz_constraint(vars),
                           detL(vars),
                           boundary_constraint(vars),
                           lambda2_constraint(vars)
                         ])

    def stability(x, l, kappa):
        z = kernel.function(x)
        Jm = kernel.jacobian(x)
        nablah = Jm.T @ Q @ z

        sum = np.zeros([p,p])
        for k in range(q):
            sum += kappa[k] * N_list[k]

        lambda0 = c * clf.function(x)

        S = F + l * Q - lambda0 * P - sum - c * np.outer(P @ z, P @ z)
        v = np.array([ nablah[1], -nablah[0] ])

        return v.T @ Jm.T @ S @ Jm @ v

    iterations = 0
    while iterations < max_iter:

        iterations += 1

        initial_guess = find_nearest_boundary(cbf, initial_point)
        # initial_guess = initial_point.tolist()

        init_x = initial_guess
        init_z = kernel.function(init_x)
        init_kappas = np.random.randn(q)

        init_l2 = init_z.T @ lambda2_matrix(init_z, init_kappas) @ init_z
        initial_vars = np.hstack([ init_x, init_kappas, init_l2 ])

        # Solve least squares problem with bounds on s and l2
        lower_bounds = [ -np.inf for _ in range(n+q) ] + [ 0.0 ]
        try:
            LS_sol = least_squares( objective, initial_vars, method='trf', bounds=(lower_bounds, np.inf), max_nfev=500 )
            if "unfeasible" not in LS_sol.message:
                equilibrium_point = LS_sol.x[0:n].tolist()
                lambda_sol = LS_sol.x[-1]
                kappa_sol = LS_sol.x[n:n+q].tolist()
                stability_value = stability( equilibrium_point, lambda_sol, kappa_sol )
                return {"cost": LS_sol.cost, "point": equilibrium_point, "lambda": lambda_sol, "stability": stability_value, "kappa": kappa_sol, "boundary_start": initial_guess, "message": LS_sol.message, "iterations": iterations }
        except:
            continue
    
def compute_equilibria_algorithm8(plant, clf, cbf, initial_point, **kwargs):
    '''
    Compute the equilibrium points
    '''
    c = 1
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()

    ACCURACY = 0.000000000001

    # QP parameters
    QP_dim = n + p + 1
    Cost_matrix = np.eye(QP_dim)
    Cost_matrix[n,n] = 1
    q = np.zeros(QP_dim)
    QP = QuadraticProgram(P=Cost_matrix, q=q)

    sample_time = 1e+0
    initial_kappa = [ 0.0 for _ in range(p) ]
    state_dynamics = Integrator( initial_point, np.zeros(n) )
    kappa_dynamics = Integrator( initial_kappa, np.zeros(p) )

    it = 0
    curr_kappa = initial_kappa
    curr_state = initial_point
    state_log = [[np.inf, np.inf]]

    cost = np.linalg.norm(  curr_state - state_log[-1] )
    while cost > 0.001 and it < max_iter:
        it += 1

        z = kernel.function(curr_state)
        Jm = kernel.jacobian(curr_state)

        a = np.zeros(p)
        sumN = np.zeros([p,p])
        sum_kappa = np.zeros([p,p])
        for i in range(p):
            a[i] = z.T @ N_list[i] @ z
            sumN[:,i] = N_list[i] @ z
            sum_kappa += curr_kappa[i] * N_list[i]

        L = ( 0.5 * c * np.outer(P @ z, P @ z) - F + sum_kappa )

        alpha = z.T @ L @ z
        vec = alpha * Q @ z - L @ z
        M = 2 * L + c * (z.T @ P @ z) * P

        nabla_alpha_x = z.T @ M @ Jm
        nabla_alpha_kappa = a

        h = cbf.evaluate_function(*curr_state)[0]
        nablah = cbf.evaluate_gradient(*curr_state)[0]

        V1 = (1/2)*(h**2)
        nabla_V1_x = h*nablah
        nabla_V1_kappa = np.zeros(p)

        # a_V1 = np.hstack([ nabla_V1_x, nabla_V1_kappa, -1.0 ])
        # b_V1 = -V1

        V2 = (1/2)*(np.linalg.norm(vec)**2)
        nabla_V2_x = vec.T @ ( np.outer(Q @ z, z) @ M + alpha * Q - L - 0.5 * c * ( z.T @ P @ z * P + np.outer( P @ z, P @ z ) ) ) @ Jm
        nabla_V2_kappa = vec.T @ ( np.outer( Q @ z, a ) - sumN )

        Matrix = alpha * Q - L
        detMatrix = np.linalg.det(Matrix)
        A = adjugate( Matrix )
        symmA = 0.5 * ( A + A.T )
        trace_vec = np.zeros(p)
        for i in range(p):
            trace_vec[i] = np.trace( A @ N_list[i] )

        V3 = (1/2)*(detMatrix**2)
        nabla_V3_x = detMatrix * z.T @ ( np.trace(A @ Q) * M - c * P @ symmA @ P ) @ Jm
        nabla_V3_kappa = detMatrix * ( np.trace(A @ Q) * a.T - trace_vec.T )

        # a_V2 = np.hstack([ nabla_V2_x, nabla_V2_kappa, -1.0 ])
        # b_V2 = -V2

        mu1, mu2, mu3 = 1, 1, 0.01
        V = mu1 * V1 + mu2 * V2 + mu3 * V3
        nabla_V_x = mu1*nabla_V1_x + mu2*nabla_V2_x + mu3 * nabla_V3_x
        nabla_V_kappa = mu1*nabla_V1_kappa + mu2*nabla_V2_kappa + mu3 * nabla_V3_kappa

        a_V = np.hstack([ nabla_V_x, nabla_V_kappa, -1.0 ])
        b_V = -V

        a_lambda = -np.hstack([ nabla_alpha_x, nabla_alpha_kappa, 0.0 ])
        b_lambda = alpha

        A = np.vstack([ a_V, a_lambda ])
        b = np.hstack([ b_V, b_lambda ])

        QP.set_inequality_constraints(A, b)
        QP_sol = QP.get_solution()

        w_control = QP_sol[0:n]

        # mu = 1.0
        # w_control = - mu * (1/np.linalg.norm(nablah)**2) * nablah * np.sign( h )

        kappa_control = QP_sol[n:n+p]
        delta = QP_sol[-1]

        state_dynamics.set_control(w_control)
        state_dynamics.actuate(sample_time)

        kappa_dynamics.set_control(kappa_control) 
        kappa_dynamics.actuate(sample_time)

        state_log.append( curr_state.tolist() )
        curr_state = state_dynamics.get_state()
        curr_kappa = kappa_dynamics.get_state()
        cost = V

    kappas = kappa_dynamics.get_state()
    return {"cost": cost, "point": curr_state, "lambda": alpha, "kappas": kappas }, state_log

def compute_equilibria_using_pencil(plant, clf, cbf, initial_point, **kwargs):
    '''
    Compute the equilibrium points
    '''
    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()
    r = len(N_list)

    ACCURACY = 1e-3

    c = 1
    max_iter = 1000
    delta = 1e-1
    initial_kappa = [np.random.rand() for _ in range(r)]
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue
        if aux_key == "delta":
            delta = kwargs[key]
            continue
        if aux_key == "initial_kappa":
            initial_kappa = kwargs[key]

    def L_fun(x, kappa):
        '''
        L = 0.5 c P m(x) m(x).T P - F - SUM κ_i N_i
        '''
        m = kernel.function(x)
        sum_kappa = np.zeros([p,p])
        for i in range(r):
            sum_kappa += kappa[i] * N_list[i]
        return 0.5 * c * np.outer(P @ m, P @ m) - F - sum_kappa

    def cost_function(x, z):
        '''
        Cost function
        '''
        return np.linalg.norm( kernel.function(x) - z )

    def cost_gradient_x(x, kappa, l, z):
        '''
        Gradient of the cost function ||m(x) - z||^2 w.r.t. x, where z is an eigenvector of the pencil (λ Q - L)
        '''
        m = kernel.function(x)
        dm_dx = kernel.jacobian(x)  

        zzT = np.outer(z,z)
        ProjQ = np.eye(p) - Q @ zzT

        L = L_fun(x, kappa)
        symL = L + L.T
        Omega = l*Q - L + Q @ zzT @ symL
        dz_dx = 0.5 * c * np.linalg.inv(Omega) @ ProjQ @ ( m.T @ P @ z * np.eye(p) + P @ np.outer(m,z) ) @ P @ dm_dx

        return (dm_dx - dz_dx).T @ (kernel.function(x) - z)

    def cost_gradient_kappa(x, kappa, l, z):
        '''
        Gradient of the cost function ||m(x) - z||^2 w.r.t. κ, where z is an eigenvector of the pencil (λ Q - L)
        '''
        zzT = np.outer(z,z)
        ProjQ = np.eye(p) - Q @ zzT

        L = L_fun(x, kappa)
        symL = L + L.T
        Omega = l*Q - L + Q @ zzT @ symL

        dz_dkappa = np.zeros([p, r])
        for i in range(r):
            dz_dkappa[:,i] = - np.linalg.inv(Omega) @ ProjQ @ N_list[i] @ z 

        return -dz_dkappa.T @ (kernel.function(x) - z)

    def filter(pencil):
        '''
        Filters invalid eigenpairs
        '''
        valid_eigenvalues, valid_eigenvectors = [], []
        for k in range(len(pencil.eigenvalues)):
            eigenvalue = pencil.eigenvalues[k]
            eigenvector = pencil.eigenvectors[:,k]
            # Filters invalid eigenpairs
            if np.abs(eigenvector.T @ Q @ eigenvector - 1) > 1e-6:
                continue
            valid_eigenvalues.append( eigenvalue )
            valid_eigenvectors.append( eigenvector )

        return valid_eigenvalues, valid_eigenvectors

    '''
    Setup general cvxpy problem
    '''
    decision_var = cp.Variable(r)
    cost_center_param = cp.Parameter(r)
    kappa_param = cp.Parameter(r)
    delta_param = cp.Parameter()
    mmT_param = cp.Parameter((p,p), symmetric=True)

    objective = cp.Minimize( cp.norm( decision_var - cost_center_param ) )
    constraint = [ c * P @ mmT_param @ P - 2 * F
                    - cp.sum([ kappa_param[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) 
                    + delta_param * cp.sum([ decision_var[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) >> 0 ]
    problem = cp.Problem(objective, constraint)

    '''
    First, solve min ||κ - κ_initial|| s.t. L(x,κ) + L(x,κ).T >= 0, for κ
    '''
    cost_center_param.value = initial_kappa
    m_initial = kernel.function(initial_point)
    mmT_param.value = np.outer(m_initial,m_initial)
    kappa_param.value = np.zeros(r)
    delta_param.value = -1
    problem.solve()

    curr_kappa = decision_var.value
    curr_pt = initial_point
    '''
    The pencil (λ Q - L) must have non-negative spectra.
    '''
    pencil = LinearMatrixPencil( Q, L_fun(curr_pt, curr_kappa) )
    for k in range(len(pencil.eigenvalues)):
        print("lambda = " + str(pencil.eigenvalues[k]))
        print("zQz = " + str(pencil.eigenvectors[:,k].T @ Q @ pencil.eigenvectors[:,k]))

    eigenvalues, eigenvectors = filter(pencil)

    # Create copies of initial points with valid eigenpairs
    curr_sols = []
    for k in range(len(eigenvalues)):
        curr_sols.append( {"x": curr_pt, "kappa": curr_kappa,
                           "lambda": eigenvalues[k], "z": eigenvectors[k],
                           "cost": cost_function(curr_pt, eigenvectors[k]) } )

    print("Initial conditions = ")
    for curr_sol in curr_sols:
        print(str( curr_sol["lambda"]))

    sol_log = []
    sol_log.append( curr_sols )

    it = 0
    total_cost = np.inf
    while total_cost > ACCURACY and it < max_iter:
        it += 1
        delta = 0.1

        # For each existing solution...
        for curr_sol in curr_sols:

            m = kernel.function(curr_sol["x"])
            mmT = np.outer(m,m)
        
            # For each valid eigenpair, solve min ||∇κ - ∇κ_nom|| s.t. L(x,κ) + L(x,κ).T + δ SUM ∇κ_i (N_i + N_i.T) >= 0, for ∇κ
            cost_center_param.value = cost_gradient_kappa( curr_sol["x"], curr_sol["kappa"], curr_sol["lambda"], curr_sol["z"] )
            mmT_param.value = mmT
            kappa_param.value = curr_sol["kappa"]
            delta_param.value = delta
            problem.solve()
            
            # If no solution exists.... (WHAT TO DO?)
            if problem.status in ["infeasible", "unbounded"]:
                raise Exception("FUCK")

            # If the SDP can be solved, store that solution
            gradC_x = cost_gradient_x( curr_sol["x"], curr_sol["kappa"], curr_sol["lambda"], curr_sol["z"] )
            gradC_kappa = decision_var.value

            # Update current solution
            curr_sol["x"] -= delta * gradC_x
            curr_sol["kappa"] -= delta * gradC_kappa

            # Updates pencil and eliminates invalid eigenpairs
            pencil.set_pencil(B = L_fun(curr_sol["x"], curr_sol["kappa"]))
            eigenvalues, eigenvectors = filter(pencil)

            # Updates lambda and z with new values
            min_index = np.argmin([ np.abs( curr_sol["lambda"] - eigenvalue ) for eigenvalue in eigenvalues ])
            curr_sol["lambda"] = eigenvalues[min_index]
            curr_sol["z"] = eigenvectors[min_index]

            # Updates cost - if everything is working, this should -->> 0
            curr_sol["cost"] = np.linalg.norm( kernel.function(curr_sol["x"]) - curr_sol["z"] )

        print(str(it) + " iterations...")
        for curr_sol in curr_sols:
            print("lambda = " + str( curr_sol["lambda"]))

        sol_log.append( curr_sols )

    return curr_sols, sol_log

'''
Implement L + L.T >> 0 (the symmetric version of L must be p.s.d.)
Compute grad_kappa_k through the following opt. problem:
min || grad_kappa_k - grad_kappa_k_NOM ||^2
s.t L(x,kappa_k+1) + L(x,kappa_k+1).T >> 0
where:
(i) grad_kappa_k_NOM is known
(ii) kappa_k+1 = kappa_k - delta * grad_kappa_k is the next kappa value

This way, kappa evolves towards the solutions that keep L p.s.d 
'''

def get_boundary_points(cbf, points, **kwargs):
    '''
    Returns points on the CBF boundary. points is a N x n array containing N x n-dimensional initial points
    '''
    alpha = 0.01
    tol = 0.0001
    max_iter = 200
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue
        if aux_key == "alpha":
            alpha = kwargs[key]
            continue

    shape = np.shape(points)
    num_points = shape[0]
    n = shape[1]

    if n != cbf._dim:
        raise Exception("Dimensions do not match.")

    Q = cbf.Q
    kernel = cbf.kernel

    # Algorithm initialization
    ONES = np.ones(num_points)

    boundary_pts = np.zeros([num_points, n])
    ZQZ = np.zeros(num_points)
    for k in range(num_points):
        boundary_pts[k,:] = points[k,:]
        z = kernel.function(boundary_pts[k,:])
        ZQZ[k] = z.T @ Q @ z

    it = 0
    while it < max_iter and np.all(ZQZ - ONES) > tol:
        it += 1
        for k in range(num_points):

            pt = boundary_pts[k,:]
            h = cbf.evaluate_function(*pt)[0]
            nablah = cbf.evaluate_gradient(*pt)[0]

            # Update points
            boundary_pts[k,:] += - alpha * h * nablah/np.linalg.norm(nablah)

            z = kernel.function(boundary_pts[k,:])
            ZQZ[k] = z.T @ Q @ z

    return boundary_pts

def find_nearest_det(plant, clf, cbf, initial_point, **kwargs):
    '''
    This method finds the nearest point on the boundary of the CBF (initialization for finding equilibria)
    '''
    c = 1
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "c":
            c = kwargs[key]
            continue

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    n = clf._dim

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()

    if len(initial_point) != n:
        raise Exception("Incorrect dimension for initial point.")
    
    # Optimization
    def objective(vars):
        x = vars[0:n]
        return np.linalg.norm(x - initial_point)

    def det_constraint(vars):
        z = kernel.function(vars[0:n])
        kappas = vars[n:n+p]

        sum = np.zeros([p,p])
        for k in range(p):
            sum += kappas[k] * N_list[k]
        L = 0.5 * c * np.outer(P @ z, P @ z) - F - sum

        return np.linalg.det( z.T @ L @ z * Q - L )

    constr1 = {'type': 'eq', 'fun': det_constraint}
    sol = minimize(objective, initial_point.tolist() + [ np.random.rand() for _ in range(p) ], method='trust-constr', constraints=[ constr1 ])

    if sol.success:
        print("Initialization algorithm was a success.\n")
        return sol.x[0:n]
    else:
        print("Initialization algorithm exit with the following error: \n")
        print(sol.message)
        return initial_point

def find_nearest_boundary(cbf, initial_point):
    '''
    This method finds the nearest point on the boundary of the CBF (initialization for finding equilibria)
    '''
    Q = cbf.Q
    kernel = cbf.kernel
    n = cbf._dim

    if len(initial_point) != n:
        raise Exception("Incorrect dimension for initial point.")

    # Optimization
    def objective(x):
        return np.linalg.norm(x - initial_point)

    def boundary_constraint(x):
        z = kernel.function(x)
        return z.T @ Q @ z - 1

    constr1 = {'type': 'eq', 'fun': boundary_constraint}

    sol = minimize(objective, initial_point, method='trust-constr', constraints=[ constr1 ])
    # sol = least_squares( lambda x : np.linalg.norm( kernel.function(x).T @ Q @ kernel.function(x) - 1 ), initial_point )

    if sol.success:
        # print("Initialization algorithm was a success.\n")
        return sol.x[0:n].tolist()
    else:
        print("Initialization algorithm exit with the following error: \n")
        print(sol.message)
        return initial_point

def solve_PEP(Q, P, **kwargs):
    '''
    Solves the eigenproblem of the type: (\lambda * Q - \kappa * C - P) @ z = 0, z.T @ Q @ z = z.T @ C @ z = 1.0, \lambda > 0.
    where P and Q are ( n x n ) p.s.d. matrices and C is a nilpotent matrix.

    Returns: \lambda:   n-array, repeated according to its multiplicity
             \kappa:    n-array, repeated according to its multiplicity
             Z:         (n x n)-array, each column corresponding to the corresponding eigenvector z
    '''
    if np.shape(Q) != np.shape(P):
        raise Exception("Matrix shapes are not compatible with given initial value.")

    matrix_shapes = np.shape(Q)
    if matrix_shapes[0] != matrix_shapes[1]:
        raise Exception("Matrices are not square.")

    dim = matrix_shapes[0]

    C = np.zeros(matrix_shapes)
    C[-1,-1] = 1

    max_iter = 10000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "initial_lines":
            initial_lines = kwargs[key]
            continue
        if aux_key == "initial_points":
            initial_points = kwargs[key]
            continue
        if aux_key == "max_iter":
            max_iter = kwargs[key]
            continue

    def compute_L(lambda_p, kappa_p):
        return lambda_p * Q - kappa_p * C - P

    def compute_det_gradient(lambda_p, kappa_p):
        L = compute_L(lambda_p, kappa_p)
        grad_det_kappa = -np.linalg.det(L)*np.trace( np.linalg.inv(L)*C )
        grad_det_lambda = np.linalg.det(L)*np.trace( np.linalg.inv(L)*Q )
        return np.array([grad_det_kappa, grad_det_lambda])

    def compute_det(solution):
        L = solution[1] * Q - solution[0] * C - P
        return np.linalg.det(L)**2

    def compute_F(solution):
        '''
        This inner method computes the vector field F(lambda, kappa, z) and returns its value.
        '''
        lambda_p, kappa_p, z = solution[0], solution[1], solution[2:]
        n = len(z)
        F = np.zeros(n+3)
        L = compute_L(lambda_p, kappa_p)
        F[0:n] = L @ z
        F[n] = 0.5 - 0.5 * z @ C @ z
        F[n+1] = 0.5 * z @ Q @ z - 0.5
        F[n+2] = compute_det([kappa_p, lambda_p])
        return F

    def compute_Jac(solution):
        '''
        This inner method computes the vector field F(lambda, kappa, z) and returns its value.
        '''
        lambda_p, kappa_p, z = solution[0], solution[1], solution[2:]
        n = len(z)
        L = compute_L(lambda_p, kappa_p)
        Jac1 = np.vstack( [ (Q @ z).reshape(n,1), 0, 0 ] )
        Jac2 = np.vstack( [ -(C @ z).reshape(n,1), 0, 0 ] )
        Jac3 = np.vstack( [ L, -(C @ z).reshape(1,n), (Q @ z).reshape(1,n) ] )
        Jac = np.hstack([ Jac1, Jac2, Jac3 ])
        return Jac

    init_lambdas = np.random.rand()
    init_kappas = np.random.rand()
    init_zs = np.random.rand(dim)

    # Initial guess using line-based projection
    if 'initial_lines' in locals():
        init_kappas = np.array([], dtype=float)
        init_lambdas = np.array([], dtype=float)
        init_zs = np.array([], dtype=float).reshape(dim,0)

        for line in initial_lines:
            m = line["angular_coef"]
            p = line["linear_coef"]

            pencil = LinearMatrixPencil2( -m*Q + C, p*Q - P )
            kappas = pencil.eigenvalues
            # Remove infinite eigenvalues
            index_inf, = np.where(np.abs(kappas) == np.inf)
            kappas = np.delete(kappas, index_inf)

            lambdas = m * kappas + p
            Z = pencil.eigenvectors
            Z = np.delete(Z, index_inf, axis=1)

            init_kappas = np.hstack([init_kappas, kappas])
            init_lambdas = np.hstack([init_lambdas, lambdas])
            init_zs = np.hstack([init_zs, Z])

    # Initial guess using points in the kappa x lambda plane and random initial eigenvectors
    if 'initial_points' in locals():
        num_points = initial_points.shape[1]
        # init_kappas = np.array([], dtype=float)
        # init_lambdas = np.array([], dtype=float)
        # init_zs = np.array([], dtype=float).reshape(dim,0)
        # for pt in initial_points.T:
        #     res = minimize(compute_det, pt)
        #     solution = res.x
        #     init_kappas = np.hstack([init_kappas, solution[0]])
        #     init_lambdas = np.hstack([init_lambdas, solution[1]])

        #     L = compute_L(solution[1],solution[0])
        #     eigens, eigenvecs = np.linalg.eig(L)
        #     z = eigenvecs[np.where( np.abs(eigens) < ZERO_ACCURACY )]
        #     init_zs = np.hstack([init_zs, z.reshape(dim,1)])

        init_kappas = initial_points[0,:]
        init_lambdas = initial_points[1,:]
        init_zs = np.random.rand(dim, num_points)

    init_guesses = np.vstack([ init_lambdas, init_kappas, init_zs ])

    # Main loop ---------------------------------------------------------------------------------
    num_points = len(init_lambdas)
    lambdas = np.zeros(num_points)
    kappas = np.zeros(num_points)
    Z = np.zeros([matrix_shapes[0], num_points])
    for k in range(num_points):
        # solution = fsolve(compute_F, init_guesses[:,k], maxfev = max_iter, factor = 0.1)
        # solution = fsolve(compute_F, init_guesses[:,k], fprime = compute_Jac, maxfev = max_iter, factor = 0.1)
        solution = root(compute_F, init_guesses[:,k], method='lm')

        lambdas[k] = solution.x[0]
        kappas[k] = solution.x[1]
        Z[:,k] = solution.x[2:]

    # Filter bad points -------------------------------------------------------------------------
    index_to_be_deleted = []
    for k in range(num_points):
        L = (lambdas[k] * Q - kappas[k] * C - P)
        z = Z[:,k]
        if z[-1] < 0:
            Z[:,k] = -Z[:,k]
        if np.linalg.norm(L @ z) > ZERO_ACCURACY or np.abs( z @ C @ z - 1 ) > ZERO_ACCURACY or np.abs( z @ Q @ z - 1 ) > ZERO_ACCURACY or lambdas[k] <= 0:
            index_to_be_deleted.append(k)

    lambdas = np.delete(lambdas, index_to_be_deleted)
    kappas = np.delete(kappas, index_to_be_deleted)
    Z = np.delete(Z, index_to_be_deleted, axis = 1)

    return lambdas, kappas, Z, init_kappas, init_lambdas

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
    Class for regular, symmetric linear matrix pencils of the form H(\lambda) = mu1 A - mu2 B, where A and B are p.s.d. matrices
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
        return lambda2_param * self._A - lambda1_param * self._B

    def compute_eig(self):
        '''
        Given the pencil matrices A and B, this method solves the pencil eigenvalue problem.
        '''
        # Compute the sorted pencil eigenvalues
        schurHv, schurHh, _, _, Q, Z = scipy.linalg.ordqz( self._B, self._A )
        self.lambda1 = np.diag(schurHh)
        self.lambda2 = np.diag(schurHv)
        pencil_eig = self.lambda2/self.lambda1
        sorted_args = np.argsort(pencil_eig)

        # Compute the pencil eigenvectors
        pencil_eigenvectors = np.zeros([self.dim,self.dim])
        for k in range(len(pencil_eig)):
            P = self.value_double(self.lambda1[k], self.lambda2[k])



            N = null_space(P)
            print("lambda1 = " + str(self.lambda1[k]))
            print("lambda2 = " + str(self.lambda2[k]))
            print("Eigs of P = " + str(np.linalg.eigvals(P)) )
            print("null space = " + str(N))
            n = N[:,0]
            normalization_const = 1.0 / np.sqrt(n.T @ self._A @ n)
            pencil_eigenvectors[:,k] = normalization_const * n

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