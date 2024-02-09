import numpy as np
import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from shapely import geometry
from scipy.optimize import root, minimize, least_squares
from scipy.linalg import null_space

from common import compute_curvatures, KKTmultipliers, ellipsoid_parametrization, check_kernel
from controllers.compatibility import LinearMatrixPencil2
from functions import Kernel

ZERO_ACCURACY = 1e-9

''' 
Verification / computation of equilibrium points and their stability 
'''
def equilibrium_field(z, l, P, V, plant, clf, cbf, params):
    '''
    Returns equilibrium field: l vecQ(z) - vecP(x)
    If P or l are cvxpy variables, return cvxpy constraints of the type [ l vecQ(z) - vecP(x) == 0 ]
    '''
    kernel = check_kernel(plant, clf, cbf)
    if len(z) != kernel.kernel_dim:
        raise Exception("Input to equilibrium vector field must have the kernel dimension.")

    A_list = kernel.get_A_matrices()
    n = len(A_list)

    F = plant.get_F()
    Q = cbf.Q

    vecQ, vecP = [], []
    for k in range(n):
        vecQ.append( z.T @ A_list[k].T @ Q @ z )
        vecP.append( z.T @ A_list[k].T @ ( params["slack_gain"] * params["clf_gain"] * V * P - F ) @ z )

    if cp.expressions.variable.Variable in [ type(l), type(P) ]:
        return l * cp.vstack(vecQ) - cp.vstack(vecP)
    else:
        return l * np.array(vecQ) - np.array(vecP)

def L(x, l, P, plant, clf, cbf, params):
    '''
    Returns L matrix: L = F + l Q - p gamma V P 
    '''
    check_kernel(plant, clf, cbf)
    return plant.get_F() + l * cbf.Q - params["slack_gain"] * params["clf_gain"] * clf.function(x) * P

def S(x, l, P, plant, clf, cbf, params):
    '''
    Returns S matrix: S = H(x,l,P) - (1/pgV^2) * fc fc.T
    '''
    kernel = check_kernel(plant, clf, cbf)
    A_list = kernel.get_A_matrices()
    n = len(A_list)

    V = clf.function(x)
    m = kernel.function(x)

    fc = plant.get_fc(x)

    L_matrix = L(x, l, P, plant, clf, cbf, params)
    S_matrix = [ [ m.T @ A_list[i].T @ ( L_matrix @ A_list[j] + A_list[j].T @ L_matrix ) @ m - fc[i]*fc[j]/(params["slack_gain"] * params["clf_gain"] * (V**2)) for j in range(n) ] for i in range(n) ]

    if cp.expressions.variable.Variable in [ type(l), type(P) ]:
        return cp.bmat(S_matrix)
    else:
        return np.array(S_matrix)

def is_equilibria(eq_sol, plant, clf, cbf, params, **kwargs):
    '''
    Verifies if a point x is an equilibrium point.
    Returns: bool
    '''
    opt_tol, boundary_tol = 1e-05, 1e-03
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "opt_tol":
            opt_tol = kwargs[key]
            continue
        if aux_key == "boundary_tol":
            boundary_tol = kwargs[key]
            continue

    kernel = check_kernel(plant, clf, cbf)
    
    x = eq_sol["x"]
    m = kernel.function(x)
    if np.abs( m.T @ cbf.Q @ m - 1 ) > boundary_tol:
        return False
    
    '''
    var = λ (must be positive for a valid equilibrium solution)
    '''
    def invariant_set_constraint(l):
        '''
        Returns the vector residues of invariant set -> is zero for x in the invariant set
        '''
        return equilibrium_field(kernel.function(x), l[0], clf.P, clf.function(x), plant, clf, cbf, params)

    def objective(l):
        return np.linalg.norm(invariant_set_constraint(l))

    is_eq = False
    try:
        init_l = np.random.rand()
        init_var = [init_l]
        sol = minimize(objective, init_var, bounds=[ (0.0, np.inf) ])

        if sol.fun < opt_tol and sol.x[0] >= 0:
            is_eq = True
    except Exception as error_msg:
        print("No equilibrium points were found. Error: " + str(error_msg))
    
    if is_eq:
        eq_sol["lambda"], eq_sol["cost"] = sol.x[0], objective(sol.x)
        stability, eta = compute_stability(eq_sol, plant, clf, cbf, params)
        eq_sol["eta"], eq_sol["stability"] = eta, stability
        eq_sol["type"] = "stable"
        if stability > 0:
            eq_sol["type"] = "unstable"
    
    return is_eq

def is_removable(eq_sol, plant, clf, cbf):
    '''
    This code checks if a given equilibrium point is removable by performing arbitrary changes over the CLF matrix P (convex optimization problem)
    '''
    kernel = check_kernel(plant, clf, cbf)

    p = kernel.kernel_dim
    P = clf.P
    Q = cbf.Q
    eigP = np.linalg.eigvals(P)

    x = eq_sol["x"]
    m = kernel.function(x)
    Jm = kernel.jacobian(x)
    V = clf.function(x)

    Null = null_space(Jm.T)
    dimNull = Null.shape[1]

    def invariant_set_constr(l, P, alpha):
        return (l * Q - P) @ m - Null @ alpha

    l_var = cp.Variable()
    alpha = cp.Variable(dimNull)
    P_var = cp.Variable((p,p), symmetric=True)

    relevance = 100
    objective = cp.Minimize( relevance*l_var + cp.norm(P_var - P, 'fro') )
    constraint = [ invariant_set_constr(l_var, P_var, alpha) == 0
                    ,P_var >> P
                #   ,m.T @ P_var @ m == 2*V
                    ,cp.lambda_max(P_var) <= np.max(eigP)
                    ]
    problem = cp.Problem(objective, constraint)
    problem.solve(verbose=True)

    if "optimal" in problem.status:
        print("Invariant error = " + str(np.linalg.norm( invariant_set_constr(l_var.value, P_var.value, alpha.value) )) )
        print("Final level set V = " + str( 0.5 * m.T @ P_var.value @ m ) )
        print("Final P eigenvalues = " + str(np.linalg.eigvals(P_var.value)))
        print("Gradient norm = " + str(np.linalg.norm( P_var.value @ m )) )
        print("Minimum lambda = " + str(l_var.value))

    return P_var.value

def is_removable_by_rotations(eq_sol, plant, clf, cbf):
    '''
    This code checks if a given equilibrium point is removable by performing rotations over the CLF matrix P (nonconvex optimization problem)
    '''
    kernel = check_kernel(plant, clf, cbf)

    n = kernel._dim
    p = kernel.kernel_dim
    P = clf.P
    Q = cbf.Q
    eigP, eigvecP = np.linalg.eig(P)

    x = eq_sol["x"]
    m = kernel.function(x)
    Jm = kernel.jacobian(x)
    V = clf.function(x)

    Null = null_space(Jm.T)
    dimNull = Null.shape[1]

    '''
    var = [ λ, R, alpha ] is a (n+rankQ-1+1) dimensional array, where:
    λ is a scalar
    R is the vectorized version of a p-dimensional orthogonal matrix representing possible rotations of P (p^2 dimensional)
    alpha is an array with the dimension of the nullspace of the Jacobian transpose
    '''
    def invariant_set_constr(var):
        '''
        Invariant set constraint
        '''
        l = var[0]
        R = var[1:1+p**2].reshape((p, p))
        alpha = var[1+p**2:1+p**2+dimNull]
        return ( l * Q - R.T @ np.diag(eigP) @ R ) @ m - Null @ alpha

    def level_set_constr(var):
        '''
        Keeps the level set constant
        '''
        R = var[1:1+p**2].reshape((p, p))
        return m @ R.T @ np.diag(eigP) @ R @ m - 2 * V

    def orthonormality_constraint(var):
        '''
        Keeps matrix R orthonormal
        '''
        R = var[1:1+p**2].reshape((p, p))
        return np.linalg.norm( R.T @ R - np.eye(p), 'fro' )

    def objective(var):
        '''
        Minimizes lambda
        '''
        l = var[0]
        R = var[1:1+p**2].reshape((p, p))
        return l + np.linalg.norm( R.T @ np.diag(eigP) @ R - P , 'fro')

    init_lambda = np.random.rand()
    init_R = eigvecP.flatten()
    init_alpha = np.random.rand(dimNull)
    init_var = np.hstack([init_lambda, init_R, init_alpha])

    result = minimize(fun=objective,
                x0=init_var,
                # method='trust-constr',
                constraints = [ 
                                {'type': 'eq', 'fun': invariant_set_constr}, 
                                {'type': 'eq', 'fun': level_set_constr},
                                {'type': 'eq', 'fun': orthonormality_constraint}
                            ])

    l = result.x[0]
    R = result.x[1:1+p**2].reshape(p,p)
    alpha = result.x[1+p**2:1+p**2+dimNull]

    print("Lambda = " + str(l))
    print("Invariant set error = " + str( invariant_set_constr(result.x) ))
    print("Level set error = " + str( level_set_constr(result.x) ))
    print("Orthogonality error = " + str( orthonormality_constraint(result.x) ))

    Pnew = R.T @ np.diag(eigP) @ R
    Pnew = (Pnew+Pnew.T)/2

    print("Eigenvalues of P = " + str( np.linalg.eigvals(Pnew) ))

    return Pnew

def compute_equilibria(plant, clf, cbf, params, **kwargs):
    '''
    Finds equilibrium points solutions. If no initial point is specified, it selections a point at random from a speficied interval.
    Returns a dict containing all relevant data about the found equilibrium point, including its stability.
    '''
    tol = 1e-05
    init_x_def = False
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "init_x":
            init_x = kwargs[key]
            init_x_def = True
            continue
        if aux_key == "tol":
            tol = kwargs[key]
            continue

    kernel = check_kernel(plant, clf, cbf)

    n = kernel._dim
    minusones, plusones = -np.ones([n,1]), +np.ones([n,1])
    interval_limits = np.hstack([ minusones, plusones ]).tolist()
    if "limits" in kwargs.keys():
        interval_limits = kwargs["limits"]
        for i in range(n):
            if interval_limits[i][0] >=  interval_limits[i][1]:
                raise Exception("Lines should be sorted in ascending order.")

    if not init_x_def:
        init_x = [ np.random.uniform( interval_limits[k][0], interval_limits[k][1] ) for k in range(n) ]

    '''
    var = [ x, λ ] is a (n+rankQ-1+1) dimensional array, where:
    x is n-dimensional
    theta is (rankQ-1)-dimensional, representing the angular parameters of the ellipsoid
    λ is a scalar (must be positive for a valid equilibrium solution)
    '''
    def invariant_set_constraint(var):
        '''
        Returns the vector residues of invariant set -> is zero for x in the invariant set
        '''
        x = var[0:n]
        l = var[-1]
        return equilibrium_field(kernel.function(x), l, clf.P, clf.function(x), plant, clf, cbf, params)

    def boundary_constr(var):
        '''
        Returns the diff between mQm and 1
        '''
        x = var[0:n]
        m = kernel.function(x)
        return np.abs( m.T @ cbf.Q @ m - 1 )

    def objective(var):
        return np.linalg.norm(invariant_set_constraint(var))**2 + boundary_constr(var)**2

    x_bounds = [ (-np.inf, np.inf) for _ in range(n) ]
    l_bounds = [ (0.0, np.inf) ]

    eq_found = False
    eq_sol = {"x": None, "lambda": None, "init_x": init_x, "cost": np.inf, "stability": None, "type": None}
    try:
        init_l = np.random.rand()
        init_var = init_x + [init_l]
        sol = minimize(objective, init_var, bounds=x_bounds+l_bounds)

        if sol.fun < tol and sol.x[-1] > 0:
            eq_found = True
    except Exception as error_msg:
        print("No equilibrium points were found. Error: " + str(error_msg))
    
    if eq_found:
        eq_sol["x"], eq_sol["lambda"], eq_sol["init_x"], eq_sol["cost"] = sol.x[0:n].tolist(), sol.x[-1], init_x, np.linalg.norm( invariant_set_constraint(sol.x) )
        stability, eta = compute_stability(eq_sol, plant, clf, cbf, params)
        eq_sol["eta"], eq_sol["stability"] = eta, stability
        eq_sol["type"] = "stable"
        if stability > 0:
            eq_sol["type"] = "unstable"

    return eq_sol

def compute_stability(sol, plant, clf, cbf, params):
    '''
    Compute the stability number for a given equilibrium point.
    '''
    x = sol["x"]
    l = sol["lambda"]
    S_matrix = S(x, l, clf.P, plant, clf, cbf, params)

    '''
    Compute stability number
    '''
    nablaV = clf.gradient(x)
    nablah = cbf.gradient(x)
    norm_nablaV = np.linalg.norm(nablaV)
    norm_nablah = np.linalg.norm(nablah)
    unit_nablah = nablah/norm_nablah
    curvatures, basis_for_TpS = compute_curvatures( S_matrix, unit_nablah )

    # Compute eta - might be relevant latter
    g = plant.get_g(x)
    G = g @ g.T
    z1 = nablah / np.linalg.norm(nablah)
    z2 = nablaV - nablaV.T @ G @ z1 * z1
    eta = 1/(1 + params["slack_gain"] * z2.T @ G @ z2 )

    V = clf.function(x)
    max_index = np.argmax(curvatures)
    stability_number = curvatures[max_index] / ( params["slack_gain"] * params["clf_gain"] * V * norm_nablaV )
    # principal_direction = basis_for_TpS[:,max_index]

    '''
    If the CLF-CBF gradients are collinear, then the stability_number is equivalent to the diff. btw CBF and CLF curvatures at the equilibrium point
    '''
    # if (eta - 1) < 1e-10:
    #     curv_V = clf.get_curvature(x)
    #     curv_h = cbf.get_curvature(x)
    #     diff_curvatures = curv_h - curv_V
    #     if np.abs(diff_curvatures - stability_number) > 1e-5:
    #         raise Exception("Stability number is different then the difference of curvatures.")

    return stability_number, eta

def closest_compatible(plant, clf, cbf, eq_sols, **kwargs):
    '''
    Compute the closest P matrix that compatibilizes the CLF-CBF pair, given M known points in the invariant set.
    '''
    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    
    slack_gain, clf_gain, c_lim = 1.0, 1.0, 2.0
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "slack_gain":
            slack_gain = kwargs[key]
            continue
        if aux_key == "clf_gain":
            clf_gain = kwargs[key]
            continue
        if aux_key == "c_lim":
            c_lim = kwargs[key]

    n = clf._dim
    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    kernel = clf.kernel
    p = kernel.kernel_dim
    A_list = kernel.get_A_matrices()

    '''
    Setup cvxpy problem
    '''
    num_sols = len(eq_sols)

    P_nom = cp.Parameter((p,p), symmetric=True)
    P_var = cp.Variable((p,p), symmetric=True)
    lambdas_var = cp.Variable(num_sols)

    P_nom.value = P
    objective = cp.Minimize( cp.norm( P_var - P_nom ) )
    constraints = [ P_var >> 0 ]

    def S(lambda_var, P_var, sol):

        x = sol["x"]
        fc = plant.get_fc(x)

        V = clf.function(x)
        m = kernel.function(x)

        L = F + lambda_var * Q - slack_gain * clf_gain * V * P_var
        H = np.zeros([n,n])
        for i,j in itertools.product(range(n),range(n)):
            H[i,j] = m.T @ A_list[i].T @ ( L @ A_list[j] + A_list[j].T @ L ) @ m

        S_matrix = H - np.outer(fc, fc)/(slack_gain * clf_gain * (V**2))
        return S_matrix

    for k in range(len(eq_sols)):

        eq_sol = eq_sols[k]
        l_var = lambdas_var[k]

        x = eq_sol["x"]
        V = clf.function(x)
        m = kernel.function(x)
        Jm = kernel.jacobian(x)

        M = np.random.rand(n,n)
        normal = cbf.gradient(x) / np.linalg.norm(cbf.gradient(x))
        M[:,0] = normal
        while np.abs( np.linalg.det(M) ) <= 1e-10:
            M = np.random.rand(n,n)
            M[:,0] = normal
        Rot, _ = np.linalg.qr(M)

        aux_M = np.vstack([ np.zeros(n-1), np.eye(n-1) ])
        curv_constr = cp.lambda_min(aux_M.T @ Rot.T @ S(l_var, P_var, eq_sol) @ Rot @ aux_M) >= c_lim
        eq_constr = Jm.T @ ( F + l_var * Q - slack_gain * clf_gain * V * P_var ) @ m == 0
        clf_level_constr = V - 0.5 * m.T @ P_var @ m == 0
        pos_lambda_constr = l_var >= 0

        constraints.append( curv_constr )
        constraints.append( eq_constr )
        constraints.append( clf_level_constr )
        constraints.append( pos_lambda_constr )

    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % problem.value)

    return P_var.value

def closest_to_image(z, kernel):
    '''
    Returns a point in the state space such that minimizes the distance from its image to z.
    '''
    n = kernel._dim
    p = kernel.kernel_dim
    if len(z) != p:
        raise Exception("Dimensions are incorrect.")
    result = minimize( fun=lambda x: np.linalg.norm( kernel.function(x) - z ), x0 = [ 0.0 for _ in range(n) ] )

    return result.x

'''
The following algorithms are useful for initialization of the previous algorithms, among other utilities.
'''
def is_new(cur_sol, sols):
    '''
    Verify if equilibrium solution is new
    '''
    is_new_sol = True
    for sol in sols:
        error_x = np.linalg.norm( np.array(sol["x"]) - cur_sol["x"] )
        error_lambda = np.linalg.norm( np.array(sol["lambda"]) - cur_sol["lambda"] )
        error_kappa = np.linalg.norm( np.array(sol["kappa"]) - cur_sol["kappa"] )
        if error_x < 1e-2 and error_lambda < 1e-1 and error_kappa < 1e-1:
            is_new_sol = False
            break
    return is_new_sol

def generate_boundary(num_pts, **kwargs):
    '''
    This method returns points in the CBF boundary in the active region
    '''
    slack_gain, clf_gain = 1.0, 1.0
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "plant":
            plant = kwargs[key]
            continue
        if aux_key == "clf":
            clf = kwargs[key]
            continue
        if aux_key == "cbf":
            cbf = kwargs[key]
            continue
        if aux_key == "slack_gain":
            slack_gain = kwargs[key]
            continue
        if aux_key == "clf_gain":
            clf_gain = kwargs[key]
            continue
    
    if 'cbf' not in locals():
        raise Exception("A CBF must be specified.")

    n = cbf._dim
    minusones, plusones = -np.ones([n,1]), +np.ones([n,1])
    interval_limits = np.hstack([ minusones, plusones ]).tolist()
    if "limits" in kwargs.keys():
        interval_limits = kwargs["limits"]
        for i in range(n):
            if interval_limits[i][0] >=  interval_limits[i][1]:
                raise Exception("Lines should be sorted in ascending order.")

    Q = cbf.Q
    kernel = cbf.kernel

    rankQ = np.linalg.matrix_rank(Q)
    dim_elliptical_manifold = rankQ - 1

    '''
    var = [ x, theta, lambda0, lambda ] is a (n+rankQ-1+2) dimensional array, where:
    x is n-dimensional
    theta is (rankQ-1)-dimensional, representing the angular parameters of the ellipsoid.
    lambda0 and lambda are the corresponding KKT multipliers
    '''
    def cost_constr(var):
        '''
        Returns the cost constraint
        '''
        x = var[0:n]
        theta = var[n:(n+rankQ-1)]
        return ellipsoid_parametrization(Q, theta) - kernel.function(x)

    def constraint(var):
        '''
        Combination of all constraints
        '''
        return cost_constr(var)

    # Selects operation mode
    mode = 'boundary'
    if 'plant' and 'clf' in locals():
        mode = 'active'
        if clf._dim != cbf._dim:
            raise Exception("CLF and CBF must have the same dimension.")
        if clf.kernel != cbf.kernel:
            raise Exception("CLF and CBF must be based on the same kernel.")
        
        def active_region_constr(var):
            '''
            Returns the KKT multiplier constraints for the active region
            '''
            x = var[0:n]
            l0 = var[-2]
            l1 = var[-1]

            lambda0, lambda1 = KKTmultipliers(plant, clf, cbf, x, slack_gain, clf_gain)
            lambda0_error = l0 - lambda0
            lambda1_error = l1 - lambda1

            return np.array([lambda0_error, lambda1_error])
        
        def constraint(var):
            '''
            Combination of all constraints
            '''
            return np.hstack([ cost_constr(var), active_region_constr(var) ])

    log = {"num_trials": num_pts, "num_success": 0, "num_failure": 0, "initial":[]}
    sols = []
    for _ in range(num_pts):

        init_theta = np.zeros(dim_elliptical_manifold)
        for k in range(dim_elliptical_manifold):
            if k > 0:
                init_theta[k] = np.random.uniform(0, 2*np.pi)
                continue
            init_theta[k] = np.random.uniform(0, np.pi)
        init_x = [ np.random.uniform( interval_limits[k][0], interval_limits[k][1] ) for k in range(n) ]
        log["initial"].append(init_x)

        initial_var = init_x + init_theta.tolist() 
        lower_bounds = [ -np.inf for _ in range(n) ] + [ 0.0 for _ in range(dim_elliptical_manifold) ]
        upper_bounds = [ +np.inf for _ in range(n) ] + [ 2*np.pi for _ in range(dim_elliptical_manifold) ]

        if mode == 'active':
            init_l0, init_l1 = KKTmultipliers(plant, clf, cbf, init_x, slack_gain, clf_gain)
            initial_var += [ init_l0, init_l1 ]
            lower_bounds += [ 0.0, 0.0 ]
            upper_bounds += [ np.inf, np.inf ]
        
        # Try solving least squares
        try:
            error_flag = False
            sol = least_squares( constraint, initial_var, bounds=(lower_bounds, upper_bounds) )
        except Exception as error_msg:
            log["num_failure"] += 1 
            error_flag = True
            print(error_msg)

        if not error_flag:
            print(sol.message)
            if sol.success and sol.cost < 1e-6:
                log["num_success"] += 1
                sols.append({"x": sol.x[0:n].tolist(), "cost": sol.cost})

    return sols, log

def generate_point_grid(Q, resolution):
    '''
    Generate grid of points at the CBF boundary.
    Assumption: Q must be a positive semi definite matrix.
    The ellipsoid dimension is given by rank(Q)-1.
    'resolution' is the angular resolution for each dimension of the angular parameters.
    The total number of generated points is equal to resolution**(rank(Q)-1)
    '''
    eigsQ, eigvecsQ = np.linalg.eig(Q)

    if Q.shape[0] != Q.shape[1]:
        raise Exception("Q must be a square matrix.")
    p = Q.shape[0]
    if np.any(eigsQ < -1e-12):
        raise Exception("Q must be a positive semi-definite matrix.")

    rankQ = np.linalg.matrix_rank(Q)
    dim_elliptical_manifold = rankQ - 1

    '''
    Generate meshgrid with equally spaced angular parameters.
    '''
    params = []
    for k in range(dim_elliptical_manifold):
        if k == 0:
            params.append( np.arange(0, np.pi, resolution).tolist() )
        else:
            params.append( np.arange(0, 2*np.pi, resolution).tolist() )
    mesh_coords = np.meshgrid(*params)

    '''
    Basic recursion for nested loops. 
    Loops through all the dimensions of the angle parametrization of the (rank(Q)-1) dimensional ellipsoid,
    generating the boundary points.
    '''
    def get_points(dim=0, indexes=[], points=[]):
        if dim < dim_elliptical_manifold:
            for i in range( mesh_coords[0].shape[dim] ):
                indexes.append(i)
                get_points(dim+1, indexes, points)
                indexes.pop(-1)
        else:            
            theta_pt = [ mesh_coords[i][tuple( j for j in indexes )] for i in range(dim_elliptical_manifold) ]
            points.append( ellipsoid_parametrization( Q, theta_pt ) )

        if dim == 0:
            return points

    return get_points()

    '''
    Basic idea behind the above recursion (for 3 nested loops):
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                theta1v = mesh_coords[0]
                theta2v = mesh_coords[1]
                theta3v = mesh_coords[2]
                theta_pt = [ mesh_coords[dim][i,j,k] for dim in range(dim_elliptical_manifold) ]
                points.append( ellipsoid_parametrization( theta_pt ) )
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

def plot_invariant(plant, clf, cbf, params, **kwargs):
    '''
    Plots the invariant set at axis ax
    '''
    res = 0.1
    color = 'k'
    extended = False
    transparency = 1.0
    ax = plt
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "ax":
            ax = kwargs[key]
            continue
        if aux_key == "color":
            color = kwargs[key]
            continue
        if aux_key == "res":
            res = kwargs[key]
            continue
        if aux_key == "extended":
            extended = kwargs[key]
            continue
        if aux_key == "visible":
            if not kwargs[key]:
                transparency = 0.0
            continue

    kernel = check_kernel(plant, clf, cbf)
    F = plant.get_F()
    P, Q = clf.P, cbf.Q

    A_list = kernel.get_A_matrices()
    n = kernel._dim

    if n > 2:
        raise Exception("Plot invariant was not designed for dimensions > 2.")

    limits = [ [-1, +1] for _ in range(n) ]
    if "limits" in kwargs.keys():
        limits = kwargs["limits"]
        for i in range(n):
            if limits[i][0] >=  limits[i][1]:
                raise Exception("Lines should be sorted in ascending order.")
    
    x_min, x_max = limits[0][0], limits[0][1]
    y_min, y_max = limits[1][0], limits[1][1]

    x = np.arange(x_min, x_max, res)
    y = np.arange(y_min, y_max, res)

    x_grid, y_grid = np.meshgrid(x, y)

    def determinant( x_grid, y_grid ):
        '''
        Evaluates det([ vecQ, vecP ]) over the grid
        '''
        det_grid = np.zeros([len(x), len(y)])

        for (i,j) in itertools.product(range(len(x)), range(len(y))):
            vecQ, vecP = np.zeros(n), np.zeros(n)
            z = kernel.function([x_grid[i,j], y_grid[i,j]])
            V = 0.5 * z.T @ P @ z
            for k in range(n):
                vecQ[k] = z.T @ A_list[k].T @ Q @ z
                vecP[k] = z.T @ A_list[k].T @ ( params["slack_gain"] * params["clf_gain"] * V * P - F ) @ z
            l = vecQ.T @ vecP / vecQ.T @ vecQ
            W = np.hstack([vecQ.reshape(n,1), vecP.reshape(n,1)])
            if l >= 0 or extended:
                # det_grid[i,j] = np.sqrt( np.linalg.det( W.T @ W ) ) # does not work
                det_grid[i,j] = np.linalg.det(W)
            else:
                det_grid[i,j] = np.inf
        return det_grid
    
    return ax.contour(x_grid, y_grid, determinant( x_grid, y_grid ), levels=[0.0], colors=color, linestyles='dashed', linewidths=1.0, alpha=transparency)

def check_invariant(x, plant, clf, cbf, params, **kwargs):
    '''
    Given a state-space point, checks if it's inside the invariant manifold.
    Returns: bool [True, False], 
             lambda [float, None]
    '''
    tol = 1e-2
    extended = False
    if "tol" in kwargs.keys():
        tol = kwargs["tol"]
    if "extended" in kwargs.keys():
        extended = kwargs["extended"]

    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")

    F = plant.get_F()
    P = clf.P
    Q = cbf.Q
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    p = kernel.kernel_dim

    m = kernel.function(x)
    Jm = kernel.jacobian(x)

    '''
    Find a basis for the transpose Jacobian null space of dimension l = p - rank(Jm) >= p - n
    '''
    N = null_space(Jm.T)
    dimJm_nullspace = N.shape[1]

    '''
    Checks if the linear system ( λ Q - 0.5 c m P m P + F ) m(x) = SUM a_i n_i can be solved for (λ, a_1, ..., a_l),
    where n_i are elements of the basis for the transpose Jacobian null space.
    '''
    A = np.zeros([p,dimJm_nullspace+1])
    A[:,0] = Q @ m
    for k in range(dimJm_nullspace):
        A[:,k+1] = - N[:,k]
    b = ( 0.5 * params["slack_gain"] * params["clf_gain"] * (m.T @ P @ m) * P - F ) @ m

    sol = np.linalg.lstsq(A, b, rcond=None)
    l = sol[0][0]
    # alpha = sol[0][1:]          # nullspace coordinates
    residue = np.sum(sol[1])

    '''
    Returns True if the resulting under-determined linear system of equations has an exact solution
    '''
    if np.linalg.norm(residue) < tol and ( not extended or l >= 0 ):
        return True, l
    else:
        return False, None

def q_function(plant, clf, cbf, params, **kwargs):
    '''
    Builds corresponding q-function for a given plant and CLF-CBF pair.
    '''
    num_levels = 10
    max_level = 10
    if "max_level" in kwargs.keys():
        max_level = kwargs["max_level"]
    if "num_levels" in kwargs.keys():
        num_levels = kwargs["num_levels"]

    levels, level_step = np.linspace(-0.5, max_level, num_levels, retstep=True)
    levels = levels.tolist()
    levels.pop(0)

    cbf_contours = cbf.plot_levels(levels = levels, **kwargs)
    inv_contour = plot_invariant(plant, clf, cbf, params, **kwargs)
    inv_vertices = inv_contour.collections[0].get_paths()[0].vertices

    '''
    Finds the intersections btw the contours at each level set
    '''
    if len(cbf_contours.levels) != len(cbf_contours.collections):
        raise Exception("Error in the number of levels.")

    # Loop through each cbf contour: each corresponds to a different level (in increasing order)
    lambdas, q_levels, pts = [], [], []
    for k in range(len(levels)):

        current_lvl = levels[k]

        if len(cbf_contours.collections[k].get_paths()) == 0:
            continue

        k_cbf_contour_vertices = cbf_contours.collections[k].get_paths()[0].vertices
        poly_cbf_k = geometry.LineString(k_cbf_contour_vertices)
        poly_inv_k = geometry.LineString(inv_vertices)

        # finds intersection points between vertices
        intersections = poly_cbf_k.intersection(poly_inv_k)
        intersection_pts = [ [pt.x, pt.y] for pt in intersections.geoms ]

        # checks for outliers
        for pt in intersection_pts:
            is_inv, l = check_invariant(pt, plant, clf, cbf, params, **kwargs)
            if is_inv: 
                lambdas.append(l)
                q_levels.append(current_lvl)
                pts.append(pt)

    # sorts by increasing order of lambda
    indexes = np.argsort(lambdas)
    lambdas = np.array(lambdas)[indexes].tolist()
    q_levels = np.array(q_levels)[indexes].tolist()
    pts = np.array(pts)[indexes].tolist()

    # adds data to result struct
    return {"lambdas": lambdas, "levels": q_levels, "points": pts}

def optimize_branch(plant, clf, cbf, params, **kwargs):
    '''
    Finds the optimum value of λ along a branch of the invariant set (maximum or minimum)
    '''
    tol = 1e-05
    init_pt_def = False
    init_lambda_def = False

    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "init_pt":
            init_pt = kwargs[key]
            init_pt_def = True
            continue
        if aux_key == "init_lambda":
            init_lambda = kwargs[key]
            init_lambda_def = True
            continue
        if aux_key == "tol":
            tol = kwargs[key]
            continue

    kernel = check_kernel(plant, clf, cbf)
    n = kernel._dim
    p = kernel.kernel_dim

    limits = [ [-1, +1] for _ in range(n) ]
    if "limits" in kwargs.keys():
        limits = kwargs["limits"]
        for i in range(n):
            if limits[i][0] >=  limits[i][1]:
                raise Exception("Lines should be sorted in ascending order.")

    if not init_pt_def:
        init_pt = [ np.random.uniform( limits[k][0], limits[k][1] ) for k in range(n) ]
    if not init_lambda_def:
        init_lambda = np.random.rand()

    '''
    var = [ x, λ ] is a (n+1) dimensional array, where:
    λ is a scalar, x in a n-dimensional array
    '''
    def invariant_set_constr(var):
        '''
        Invariant set constraint
        '''
        x = var[0:n]
        z = kernel.function(x)
        V = clf.function(x)

        l = var[-1]
        V = 0.5 * z.T @ clf.P @ z
        return equilibrium_field(z, l, clf.P, V, plant, clf, cbf, params)

    def objective(var):
        '''
        Minimizes cbf value over the invariant set branch
        '''
        x = var[0:n]
        z = kernel.function(x)
        # l = var[-1]
        return cbf.function(x)

    init_var = np.hstack([init_pt, init_lambda])

    x_bounds = [ (-np.inf, np.inf) for _ in range(n) ]
    l_bounds = [ (0.0, np.inf) ]

    min_sol = {"pt": None, "lambda": None}
    max_sol = {"pt": None, "lambda": None}
    try:

        # Minimization
        result = minimize( fun= lambda var: objective(var), x0 = init_var,
                           constraints = [{'type': 'eq', 'fun': invariant_set_constr}],
                           bounds = x_bounds + l_bounds, options = {"disp": False}, tol=tol )
        print(f"Minimization exit: {result.message}")
        min_sol["pt"], min_sol["lambda"] = result.x[0:n].tolist(), result.x[-1]
        min_sol["h"] = objective(result.x)
        min_sol["norm_grad_h"] = np.linalg.norm( cbf.gradient(min_sol["pt"]) )

        # Maximization
        result = minimize( fun= lambda var: -objective(var), x0 = init_var,
                           constraints = [{'type': 'eq', 'fun': invariant_set_constr}],
                           bounds = x_bounds + l_bounds, options = {"disp": False}, tol=tol )
        print(f"Maximization exit: {result.message}")
        max_sol["pt"], max_sol["lambda"] = result.x[0:n].tolist(), result.x[-1]
        max_sol["h"]= objective(result.x)
        max_sol["norm_grad_h"] = np.linalg.norm( cbf.gradient(max_sol["pt"]) )

    except Exception as error_msg:
        print("No optimal point was found. Error: " + str(error_msg))

    return min_sol, max_sol

# --------------------------------------------------------------- DEPRECATED CODE ----------------------------------------------------------

def compute_null(plant, clf, cbf, x, KKT):
    '''
    Compute nullspace coordinates for a given x and kernel
    Returns: alpha: represents the coordinates of a vector of the nullspace of the transpose Jacobian, in a basis given by the (0, vi) eigenpairs.
             kappa: represents the coordinates on the Ni m space
    '''
    if clf._dim != cbf._dim:
        raise Exception("CLF and CBF must have the same dimension.")
    if clf.kernel != cbf.kernel:
        raise Exception("CLF and CBF must be based on the same kernel.")
    kernel = clf.kernel
    F = plant.get_F()
    P = clf.P
    Q = cbf.Q

    m = kernel.function(x)
    Jm = kernel.jacobian(x)

    N = null_space(Jm.T)
    b = ( F + KKT[1] * Q - KKT[0] * P ) @ m

    sol = np.linalg.lstsq(N, b, rcond=None)
    alpha = sol[0][0:]
    kappa = alpha2kappa({"x": x, "alpha":alpha}, kernel)

    return {"alpha": alpha, "kappa": kappa}

def alpha2kappa(eq_sol, kernel):
    '''
    Finds a representation for the Jacobian transpose null space at a point x.
    '''
    if type(kernel) != Kernel:
        raise Exception("A valid kernel must be specified.")
    
    p = kernel.kernel_dim
    N_list = kernel.get_N_matrices()
    r = len(N_list)

    # Get the solution
    x = eq_sol["x"]
    alpha = eq_sol["alpha"]

    m = kernel.function(x)
    Jm = kernel.jacobian(x)

    '''
    Compute Jacobian nullspace vector from the alpha coordinates of the equilibrium solutions
    '''
    N = null_space(Jm.T)
    dim_nullspace = N.shape[1]
    nullspace_vec = np.zeros(p)
    for k in range(dim_nullspace):
        nullspace_vec += alpha[k] * N[:,k]

    '''
    Finds kappa constants of the Jacobian nullspace vector written as L.C. of N_i m(x)
    '''
    System_matrix = np.array([ (N_list[l] @ m).tolist() for l in range(r) ]).T
    system_solution = np.linalg.lstsq(System_matrix, nullspace_vec, rcond=None)
    kappa, residuals = system_solution[0], system_solution[1]

    return kappa

def check_equilibrium(plant, clf, cbf, x, **kwargs):
    '''
    Given a state-space point, returns True if it's an equilibrium point or False otherwise.
    '''
    tol = 1e-3
    if "tol" in kwargs.keys():
        tol = kwargs["tol"]

    kernel = clf.kernel
    Q = cbf.Q

    type_dict = {"invariant": False, "equilibrium": False}
    is_invariant, eq_sol = check_invariant(plant, clf, cbf, x, **kwargs)
    if not is_invariant:
        return type_dict, eq_sol
    else:
        m = kernel.function(eq_sol["x"])
        mQm = m.T @ Q @ m
        boundary_residue = np.abs(mQm-1)
        if boundary_residue < tol:
            eq_sol["residue"] = eq_sol["residue"] + boundary_residue
            type_dict["invariant"], type_dict["equilibrium"] = True, True
            return type_dict, eq_sol
        else:
            type_dict["invariant"] = True
            return type_dict, eq_sol

def compute_equilibria_old(plant, clf, cbf, initial_points, **kwargs):
    '''
    Solve the general eigenproblem of the type:
    ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
    l1 = c V(z) P z,
    z \in Im(m)
    '''
    slack_gain, clf_gain = 1.0, 1.0
    max_iter = 1000
    for key in kwargs.keys():
        aux_key = key.lower()
        if aux_key == "slack_gain":
            slack_gain = kwargs[key]
            continue
        if aux_key == "clf_gain":
            clf_gain = kwargs[key]
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

    num_pts = len(initial_points)

    '''
    var = [ x, alpha, lambda ] is a (n+p+1) dimensional array
    x is n-dimensional
    alpha is p-dimensional, representing the coordinates of a vector of the nullspace of the transpose Jacobian, in a basis given by the (0, vi) eigenpairs. 
                            HOWEVER: alpha has actually the dimension of the transpose Jacobian null space (p is the maximum dimension of this null space).
    λ is a scalar (must be positive for a valid equilibrium solution)
    '''
    def gradient_collinearity_constraint(var):
        '''
        Returns the vector residues of gradient collinearity -> is zero iff the collinearity condition is satisfied 
        '''
        x = var[0:n]
        alpha = var[n:n+p]
        l = var[-1]

        m = kernel.function(x)
        Jm = kernel.jacobian(x)

        N = null_space(Jm.T)
        dim_nullspace = N.shape[1]

        '''
        IMPORTANT: notice that alpha is filtered (the last coords are just ignored - we really only need as much coords as the dimension of the Jacobian left-nullspace)
        '''
        nullspace_vec = np.zeros(p)
        for k in range(dim_nullspace):
            nullspace_vec += alpha[k] * N[:,k]

        return ( F + l * Q - 0.5 * slack_gain * clf_gain * (m.T @ P @ m) * P ) @ m - nullspace_vec      # = 0

    def boundary_constraint(var):
        '''
        Returns the diff between mQm and 1
        '''
        x = var[0:n]
        m = kernel.function(x)
        return np.abs( m.T @ Q @ m - 1 )

    def constraint(var):
        '''
        Combination of all constraints
        '''
        return np.hstack([ gradient_collinearity_constraint(var), boundary_constraint(var) ])

    '''
    Solve constrained least squares
    '''
    log = {"num_trials": num_pts, "num_success": 0, "num_failure": 0}
    sols = []
    for k in range(num_pts):
        initial_point = initial_points[k]

        initial_m = kernel.function(initial_point)
        initial_var = initial_point + np.random.rand(p).tolist() + [0.0]
        v_lambda0 = gradient_collinearity_constraint(initial_var)
        initial_lambda = -(initial_m.T @ v_lambda0)/(initial_m.T @ Q @ initial_m)
        initial_var[-1] = initial_lambda
        
        lower_bounds = [ -np.inf for _ in range(p+n) ] + [ 0.0 ]
        try:
            error_flag = False
            sol = least_squares( constraint, initial_var, bounds=(lower_bounds, np.inf), max_nfev=max_iter )
        except Exception as error_msg:
            log["num_failure"] += 1 
            error_flag = True
            print(error_msg)

        if not error_flag:
            print(sol.message)
            if sol.success and sol.cost < 1e-5:
                log["num_success"] += 1

                # Equilibrium pt found
                x = sol.x[0:n]

                # Ignore the last p - rank(Jm) entries of kappa
                Jm = kernel.jacobian(x)
                dim_nullspace = null_space(Jm.T).shape[1]
                alpha = sol.x[n:n+dim_nullspace]
                l = sol.x[-1]

                # Compute eta - might be relevant latter
                g = plant.g_method(x)
                G = g @ g.T
            
                V = clf.function(x)
                nablaV = clf.gradient(x)
                nablah = cbf.gradient(x)

                z1 = nablah / np.linalg.norm(nablah)
                z2 = nablaV - nablaV.T @ G @ z1 * z1
                eta = 1/(1 + slack_gain * z2.T @ G @ z2 )

                eq_sol = {"x": x.tolist(), "alpha": alpha.tolist(), "lambda": l,"eta": eta, "residue": sol.cost, "V": V }
                eq_sol["stability"], eq_sol["kappa"] = compute_stability(plant, clf, cbf, eq_sol, slack_gain=slack_gain, clf_gain=clf_gain)

                # If solution is new, append it      
                if is_new(eq_sol, sols): sols.append( eq_sol )
            else:
                log["num_failure"] += 1

    return sols, log

'''
The following algorithms implement some version of constrained least_squares or constrained minimization algorithms for finding the equilibrium points.
None of them work.
'''
# def compute_equilibria_algorithm1(F, clf, cbf, **kwargs):
    # '''
    # Solve the general eigenproblem of the type:
    # ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0,
    # l1 = c V(z) P z,
    # z \in Im(m)
    # '''
    # max_iter = 1000
    # for key in kwargs.keys():
    #     aux_key = key.lower()
    #     if aux_key == "max_iter":
    #         max_iter = kwargs[key]
    #         continue
    #     if aux_key == "c":
    #         c = kwargs[key]
    #         continue
    #     if aux_key == "initial":
    #         initial_guess = kwargs[key]

    # if clf._dim != cbf._dim:
    #     raise Exception("CLF and CBF must have the same dimension.")
    # n = clf._dim

    # P = clf.P
    # Q = cbf.Q
    # if clf.kernel != cbf.kernel:
    #     raise Exception("CLF and CBF must be based on the same kernel.")
    # kernel = clf.kernel
    # p = kernel.kernel_dim

    # # A_list = clf.kernel.get_A_matrices()
    # N_list = kernel.get_N_matrices()

    # def linear_pencil(linear_combs):
    #     '''
    #     linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
    #     Computes the linear matrix pencil.
    #     '''
    #     L = F + linear_combs[1] * Q - linear_combs[0] * P
    #     for k in range(2, p - n):
    #         L += linear_combs[k] * N_list[k]
    #     return L

    # def linear_pencil_det(linear_combs):
    #     '''
    #     linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
    #     Computes determinant of linear matrix pencil.
    #     '''
    #     return np.linalg.det( linear_pencil(linear_combs) )

    # def compute_F(solution):
    #     '''
    #     This inner method computes the vector field F(lambda, kappa, z) and returns its value.
    #     '''
    #     linear_combs, z = solution[0:p-n+2], solution[p-n+2:]
    #     kernel_constraints = kernel.get_constraints(z)
    #     num_kernel_constraints = len(kernel_constraints)

    #     F = np.zeros(p+num_kernel_constraints+2)
    #     L = linear_pencil(linear_combs)
    #     F[0:p] = L @ z
    #     V = 0.5 * z.T @ P @ z
    #     F[p] = linear_combs[0] - c*V
    #     F[p+1] = linear_pencil_det(linear_combs)
    #     F[p+2:] = kernel_constraints

    #     return F

    # # t = time.time()
    # solution = root(compute_F, initial_guess, method='lm', tol = 0.00001)
    # # solution = fsolve(compute_F, initial_guess, maxfev = max_iter)
    # # elapsed = time.time() - t

    # l1 = solution.x[0]
    # l2 = solution.x[1]
    # kappas = solution.x[2:p-n+2]
    # z = solution.x[p-n+2:]

    # equilibrium_point = np.flip(z[1:n+1]).tolist()

    # return equilibrium_point

# def compute_equilibria_algorithm2(F, clf, cbf, **kwargs):
#     '''
#     Solve the general eigenproblem of the type:
#     ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
#     l1 = c V(z) P z,
#     z \in Im(m)
#     '''
#     max_iter = 1000
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "initial":
#             initial_guess = kwargs[key]

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim

#     # A_list = clf.kernel.get_A_matrices()
#     N_list = kernel.get_N_matrices()

#     # Optimization
#     import cvxpy as cp

#     # General optimization problem

#     # ----- Variables ------
#     delta_var = cp.Variable()
#     l1_var = cp.Variable()
#     l2_var = cp.Variable()
#     kappa_var = cp.Variable(p-n)
#     z_var = cp.Variable(p)
#     L_var = cp.Variable((p,p))

#     # ----- Prob definition ------
#     L_var = F + l1_var * Q - l2_var * P
#     for k in range(p-n):
#         L_var += kappa_var[k] * N_list[k]

#     matrices = kernel.get_matrix_constraints()
#     kernel_constraints = [ z_var[0] == 1 ]
#     kernel_constraints += [ z_var.T @ matrices[k] @ z_var == 0 for k in range(len(matrices)) ]

#     objective = cp.Minimize( cp.norm(delta_var)**2 )
#     constraints = [ L_var @ z_var == 0,
#                     l1_var == 0.5 * c * z_var.T @ P @ z_var,
#                     l2_var >= delta_var,
#                     delta_var >= 0 ]
#     constraints += kernel_constraints
#     problem = cp.Problem(objective, constraints)

#     problem.solve()
#     if problem.status not in ["infeasible", "unbounded"]:
#         # Otherwise, problem.value is inf or -inf, respectively.
#         print("Optimal value: %s" % problem.value)
#         for variable in problem.variables():
#             print("Variable %s: value %s" % (variable.name(), variable.value))

#     # CONCLUSION: problem is fundamentally nonconvex. Cannot be solved through CVXPY.

# def compute_equilibria_algorithm3(F, clf, cbf, initial_point, **kwargs):
#     '''
#     Solve the general eigenproblem of the type:
#     ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
#     l1 = c V(z) P z,
#     z \in Im(m)
#     '''
#     c = 1
#     l2_bound = 1.0
#     max_iter = 1000
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "l2_bound":
#             l2_bound = kwargs[key]
#             continue
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim

#     # A_list = clf.kernel.get_A_matrices()
#     N_list = kernel.get_N_matrices()

#     # Optimization
#     import cvxpy as cp

#     ACCURACY = 0.0001

#     it = 0
#     z_old = np.inf
#     z = kernel.function( initial_point )
#     while np.linalg.norm( z - z_old ) > ACCURACY or it < max_iter:
#         it += 1

#         # First optimization: QP

#         # ----- Variables ------
#         delta_var = cp.Variable()
#         l1_var = cp.Variable()
#         l2_var = cp.Variable()
#         kappa_var = cp.Variable(p-n)
#         L_var = cp.Variable((p,p))

#         # ----- Parameters ------
#         z_param = cp.Parameter(p)
#         l1_param = cp.Variable()

#         # ----- Prob definition ------
#         l1_param = 0.5 * c * z_param.T @ P @ z_param

#         L_var = F + l1_var * Q - l2_var * P
#         for k in range(p-n):
#             L_var += kappa_var[k] * N_list[k]

#         QP_objective = cp.Minimize( cp.norm(l2_var)**2 )
#         # QP_objective = cp.Minimize( cp.norm(l2_var)**2 + cp.norm(kappa_var)**2 )
#         QP_constraints = [ L_var @ z_param == 0 ,
#                            l1_var == 0.5 * c * z_param.T @ P @ z_param ,
#                         #    l2_var >= delta_var ,
#                            l2_var >= l2_bound ]
#         QP = cp.Problem(QP_objective, QP_constraints)

#         # ------- Solve QP --------
#         z_param.value = z
#         QP.solve()

#         p_var = np.zeros(p-n+2)
#         p_var[0] = l1_var.value
#         p_var[1] = l2_var.value
#         p_var[2:] = kappa_var.value

#         if QP.status in ["infeasible", "unbounded"]:
#             raise Exception("QP is " + QP.status)

#         # Second problem: nonlinear Newton-Raphson method

#         def linear_pencil(p_var):
#             '''
#             linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
#             Computes the linear matrix pencil.
#             '''
#             L = F + p_var[1] * Q - p_var[0] * P
#             for k in range(2, p - n):
#                 L += p_var[k] * N_list[k]
#             return L

#         def linear_pencil_det(p_var):
#             '''
#             linear_combs = [ lambda1, lambda2, kappa1, kappa2, ... ]
#             Computes determinant of linear matrix pencil.
#             '''
#             return np.linalg.det( linear_pencil(p_var) )

#         def compute_F(z):
#             '''
#             This inner method computes the vector field F(sol) and returns its value.
#             '''
#             kernel_constraints = kernel.get_constraints(z)
#             num_kernel_constraints = len(kernel_constraints)

#             F = np.zeros(p+2+num_kernel_constraints)
#             L = linear_pencil(p_var)
#             F[0:p] = L @ z
#             F[p] = p_var[0] - 0.5 * c * z.T @ P @ z
#             F[p+1] = np.eye(p)[0,:].T @ z - 1
#             F[p+2:] = kernel_constraints

#             return F

#         z_old = z
#         solution = root(compute_F, z, method='lm', tol = ACCURACY)
#         z = solution.x

#     equilibrium_point = np.flip(z[1:n+1]).tolist()
#     sol_dict = { "equilibrium_point": equilibrium_point,
#                  "lambda1": l1_var.value.tolist(),
#                  "lambda2": l2_var.value.tolist(),
#                  "kappas": kappa_var.value.tolist(),
#                  "z": z.tolist() }

#     return sol_dict

# def compute_equilibria_algorithm4(plant, clf, cbf, initial_point, **kwargs):
#     '''
#     Solve the general eigenproblem of the type:
#     ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
#     l1 = c V(z) P z,
#     z \in Im(m)
#     '''
#     c = 1
#     max_iter = 100
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()

#     g = plant.get_g()
#     nablaV = clf.get_gradient()
#     nablah = cbf.get_gradient()
#     # initial_l2 = nablaV.T @ g @ g.T @ nablaV / nablah.T @ g @ g.T @ nablah
#     initial_l2 = 1.0

#     print("Initial lambda2 = " + str(initial_l2))

#     # Optimization
#     from scipy.optimize import least_squares

#     ACCURACY = 0.000000001

#     it = 0
#     z_old = np.inf
#     z = kernel.function( initial_point )

#     cost = np.linalg.norm( z - z_old )
#     while cost > ACCURACY and it < max_iter:
#         it += 1

#         # First optimization: least squares with bound on lambda2
#         def linear_pencil(p_var):
#             '''
#             p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
#             Computes the linear matrix pencil.
#             '''
#             L = F + p_var[1] * Q - p_var[0] * P
#             for k in range(p - n):
#                 L += p_var[k+2] * N_list[k]
#             return L

#         def linear_pencil_det(p_var):
#             '''
#             p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
#             Computes determinant of linear matrix pencil.
#             '''
#             return np.linalg.det( linear_pencil(p_var) )

#         def F1(p_var):
#             '''
#             Function for constrained least squares
#             '''
#             return np.array([ linear_pencil_det(p_var), p_var[0] - 0.5 * c * z.T @ P @ z ])

#         # ------- Solve constrained least squares --------
#         initial_p_var = np.hstack([ 0.5 * c * z.T @ P @ z, initial_l2, np.zeros(p-n) ])
#         lower_bounds = [ -np.inf, 0.0 ] + [ -np.inf for _ in range(p-n) ]
#         solution1 = least_squares( F1, initial_p_var, bounds=(lower_bounds, np.inf) )

#         p_var = np.zeros(p-n+2)
#         p_var[0] = solution1.x[0]
#         p_var[1] = solution1.x[1]
#         p_var[2:] = solution1.x[2:]

#         print("p = " + str(p_var))
#         print("F1 = " + str(F1(p_var)))

#         # Second problem: nonlinear Newton-Raphson method

#         def F2(z):
#             '''
#             This inner method computes the vector field F(sol) and returns its value.
#             '''
#             kernel_constraints = kernel.get_constraints(z)
#             num_kernel_constraints = len(kernel_constraints)

#             F2 = np.zeros(p+1+num_kernel_constraints)
#             L = linear_pencil(p_var)
#             F2[0:p] = L @ z
#             F2[p] = p_var[0] - 0.5 * c * z.T @ P @ z
#             F2[p+1:] = kernel_constraints

#             return F2

#         z_old = z
#         solution2 = root(F2, z_old, method='lm', tol = ACCURACY)
#         z = solution2.x
#         cost = np.linalg.norm( z - z_old )

#         # print( kernel.get_constraints(z) )
#         # print( kernel.is_in_kernel_space(np.array(z)) )
#         # print(cost)

#     equilibrium_point = np.flip(z[1:n+1]).tolist()
#     sol_dict = { "equilibrium_point": equilibrium_point,
#                  "lambda1": p_var[0],
#                  "lambda2": p_var[1],
#                  "kappas": p_var[2:].tolist(),
#                  "z": z.tolist(),
#                  "cost": cost }

#     return sol_dict

# def compute_equilibria_algorithm5(plant, clf, cbf, initial_point, **kwargs):
#     '''
#     Solve the general eigenproblem of the type:
#     ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
#     l1 = c V(z) P z,
#     z \in Im(m)
#     '''
#     c = 1
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()

#     # Optimization

#     ACCURACY = 0.000001

#     z = kernel.function( initial_point )

#     initial_l1 = 0.5 * c * z.T @ P @ z
#     initial_l2 = 0.0
#     initial_kappas = np.zeros(p-n)

#     p_var = np.hstack([ initial_l1, initial_l2, initial_kappas ])

#     # Least squares with bound on lambda2
#     def linear_pencil(p_var):
#         '''
#         p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
#         Computes the linear matrix pencil.
#         '''
#         L = F + p_var[1] * Q - p_var[0] * P
#         for k in range(p - n):
#             L += p_var[k+2] * N_list[k]
#         return L

#     def linear_pencil_det(p_var):
#         '''
#         p_var = [ lambda1, lambda2, kappa1, kappa2, ... ]
#         Computes determinant of linear matrix pencil.
#         '''
#         return np.linalg.det( linear_pencil(p_var) )

#     def boundary_equilibria(variables):
#         '''
#         Constraints for boundary equilibrium points
#         '''
#         z = variables[0:p]
#         p_var = variables[p:]
#         kernel_constraints = kernel.get_constraints(z)

#         return np.hstack([ linear_pencil_det(p_var),
#                            linear_pencil(p_var) @ z,
#                            p_var[0] - 0.5 * c * z.T @ P @ z,
#                            z.T @ Q @ z - 1,
#                            kernel_constraints ])

#     def interior_equilibria(variables):
#         '''
#         Constraints for interior equilibrium points
#         '''
#         z = variables[0:p]
#         p_var = variables[p:]
#         kernel_constraints = kernel.get_constraints(z)

#         return np.hstack([ linear_pencil_det(p_var),
#                            linear_pencil(p_var) @ z,
#                            p_var[0] - 0.5 * c * z.T @ P @ z,
#                            p_var[1],
#                            kernel_constraints ])

#     # ------- Solve constrained least squares --------
#     initial_variables = np.hstack([ z, p_var ])
#     lower_bounds = [ -np.inf for _ in range(p) ] + [ -np.inf, 0.0 ] + [ -np.inf for _ in range(p-n) ] # forces inequality lambda2 >= 0

#     # Solves for boundary equilibrium points
#     boundary_z = [ None for _ in range(p) ]
#     boundary_p_var = np.array([ None for _ in range(p-n+2) ])
#     boundary_equilibrium = [ None for _ in range(n) ]
#     boundary_cost = None

#     t = time.time()
#     boundary_solution = least_squares( boundary_equilibria, initial_variables, bounds=(lower_bounds, np.inf), xtol = ACCURACY )
#     boundary_delta = time.time() - t

#     z = boundary_solution.x[0:p]
#     if np.abs( z.T @ Q @ z - 1 ) < ACCURACY:
#         boundary_z = boundary_solution.x[0:p]
#         boundary_p_var = boundary_solution.x[p:]
#         boundary_equilibrium = kernel.kernel2state(boundary_z).tolist()
#         boundary_cost = boundary_solution.cost

#     interior_z = [ None for _ in range(p) ]
#     interior_p_var = np.array([ None for _ in range(p-n+2) ])
#     interior_equilibrium = [ None for _ in range(n) ]
#     interior_cost = None
#     interior_delta = 0.0

#     t = time.time()
#     interior_solution = least_squares( interior_equilibria, initial_variables, bounds=(lower_bounds, np.inf), xtol = ACCURACY )
#     interior_delta = time.time() - t

#     if interior_solution.cost < ACCURACY:
#         interior_z = interior_solution.x[0:p]
#         interior_p_var = interior_solution.x[p:]
#         interior_equilibrium = kernel.kernel2state(boundary_z).tolist()
#         interior_cost = interior_solution.cost

#     boundary_sol = { "equilibrium": boundary_equilibrium,
#                      "z": boundary_z,
#                      "lambda1": boundary_p_var[0],
#                      "lambda2": boundary_p_var[1],
#                      "kappas": boundary_p_var[2:].tolist(),
#                      "cost": boundary_cost,
#                      "time": boundary_delta }

#     interior_sol = { "equilibrium": interior_equilibrium,
#                      "z": interior_z,
#                      "lambda1": interior_p_var[0],
#                      "lambda2": interior_p_var[1],
#                      "kappas": interior_p_var[2:].tolist(),
#                      "cost": interior_cost,
#                      "time": interior_delta }

#     return { "boundary": boundary_sol, "interior": interior_sol }

# def compute_equilibria_algorithm6(plant, clf, cbf, initial_points, **kwargs):
#     '''
#     Solve the general eigenproblem of the type:
#     ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
#     l1 = c V(z) P z,
#     z \in Im(m)
#     '''
#     c = 1
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()
#     r = len(N_list)

#     # matrices = kernel.get_matrix_constraints()
#     # num_matrices = len(matrices)

#     # Optimization
#     ACCURACY = 0.0000001

#     # ----------------------- Auxiliary functions ------------------
#     def commutator(A,B,z):
#         ZZt = np.outer(z,z)
#         return A @ ZZt @ B - B @ ZZt @ A

#     # def linear_pencil(vars):
#     #     z = kernel.function(vars[0:n])
#     #     kappas = vars[n:n+p]
#     #     sum = np.zeros([p,p])
#     #     for k in range(p):
#     #         sum += kappas[k] * commutator(Q,N_list[k],z)
#     #     L = 0.5 * c * commutator( commutator(Q,P,z), P, z) - commutator(Q,F,z) - sum
#     #     return L

#     def lambda2_matrix(z, kappas):
#         sum = np.zeros([p,p])
#         for k in range(p):
#             sum += kappas[k] * N_list[k]
#         return 0.5 * c * np.outer(P @ z, P @ z) - F - sum

#     def linear_pencil(vars):
#         z = kernel.function(vars[0:n])
#         kappas = vars[n:n+p]
#         l2 = vars[-1]
#         return lambda2_matrix(z, kappas) - l2 * Q

#     # def lambda2_fun(z, kappas):
#     #     return z.T @ lambda2_matrix(z, kappas) @ z

#     def projection(z):
#         return np.eye(len(z)) - np.outer(z, Q @ z)

#     # ------------------- Constraints on vars = [ x, kappas, l2 ], kappas is p-dimensional ----------------------

#     def detL_eq(vars):
#         return np.linalg.det( linear_pencil(vars) )

#     def Lz_eq(vars):
#         z = kernel.function(vars[0:n])
#         return linear_pencil(vars) @ z

#     # def lambda2_eq(vars):
#     #     z = kernel.function(vars[0:n])
#     #     kappas = vars[n:n+p]
#     #     l2 = vars[-1]
#     #     return l2 - lambda2_fun(z, kappas)

#     # def lambda2_ineq_constraint(vars):
#     #     l2 = vars[-1]
#     #     return l2 # >> 0

#     # def boundary_ineq(vars):
#     #     z = kernel.function(vars[0:n])
#     #     return z.T @ Q @ z - 1 # >> 0

#     def boundary_eq(vars):
#         z = kernel.function(vars[0:n])
#         return z.T @ Q @ z - 1 # >= 0

#     # def lambda2_ineq(vars):
#     #     z = kernel.function(vars[0:n])
#     #     kappas = vars[n:n+p]
#     #     return z.T @ lambda2_matrix(z, kappas) @ z

#     def lambda2_ineq(vars):
#         return vars[-1]

#     def kappa_eq(vars):
#         z = kernel.function(vars[0:n])
#         kappas = vars[n:n+p]
#         sum = 0.0
#         for k in range(p):
#             sum += kappas[k] * z.T @ ( N_list[k] -  N_list[k].T ) @ z
#         return sum

#     def complementary_slackness_eq(vars):
#         z = kernel.function(vars[0:n])
#         kappas = vars[n:n+p]
#         return z.T @ lambda2_matrix(z, kappas) @ projection(z) @ z
#         # return lambda2_ineq_constraint(vars) * boundary_ineq_constraint(vars)

#     eq_constr0 = {'type': 'eq', 'fun': detL_eq}
#     eq_constr1 = {'type': 'eq', 'fun': Lz_eq}
#     eq_constr2 = {'type': 'eq', 'fun': boundary_eq}
#     eq_constr3 = {'type': 'eq', 'fun': kappa_eq}

#     ineq_constr1 = {'type': 'ineq', 'fun': lambda2_ineq}
#     # ineq_constr2 = {'type': 'ineq', 'fun': boundary_ineq}

#     # Initialize with boundary points
#     # boundary_pts = get_boundary_points( cbf, initial_points )
#     num_pts = np.shape(initial_points)[0]

#     # Try minimization
#     solutions = {"points": [], "lambda2": [], "indexes": []}
#     error_counter = 0
#     for k in range(num_pts):

#         x_n = initial_points[k,:]
#         def objective(vars):
#             x = vars[0:n]
#             return np.linalg.norm(x - x_n)**2

#         init_z = kernel.function(x_n)
#         a = np.zeros([1,p])
#         for i in range(p):
#             a[:,i] = init_z.T @ ( N_list[i] -  N_list[i].T ) @ init_z

#         init_kappas = np.zeros(p)
#         for col in null_space(a).T:
#             init_kappas += np.random.randn() * col

#         init_l2 = init_z.T @ lambda2_matrix(init_z, init_kappas) @ init_z
#         initial_vars = np.hstack([ x_n, init_kappas, init_l2 ])

#         # Solve optimization problem
#         min_sol = minimize(objective, initial_vars, method='trust-constr', constraints=[ eq_constr1, eq_constr2, eq_constr3, ineq_constr1 ])

#         if min_sol.success:
#             solutions["points"].append( min_sol.x[0:n] )
#             solutions["lambda2"].append( min_sol.x[-1] )
#             solutions["indexes"].append( k )
#         else:
#             error_counter += 1
#             print(min_sol.message)

#     return solutions

# def compute_equilibria_algorithm7(plant, clf, cbf, initial_point, **kwargs):
#     '''
#     Solve the general eigenproblem of the type:
#     ( F + l2 Q - l1 P - \sum k_i N_i ) z = 0, l2 >= 0,
#     l1 = c V(z) P z,
#     z \in Im(m)
#     Returns:                array with boundary equilibrium point solutions
#     '''
#     c = 1
#     max_iter = 10
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()
#     q = len(N_list)

#     # Optimization
#     ACCURACY = 0.000000000001

#     # vars = [ x, kappas, l2 ]

#     # def commutator(A,B,z):
#     #     ZZt = np.outer(z,z)
#     #     return A @ ZZt @ B - B @ ZZt @ A

#     # def linear_pencil(vars):
#     #     z = kernel.function(vars[0:n])
#     #     kappas = vars[n:p]
#     #     sum = np.zeros([p,p])
#     #     for k in range(len(N_list)):
#     #         sum += kappas[k] * commutator(Q,N_list[k],z)
#     #     L = 0.5 * c * commutator( commutator(Q,P,z), P, z) - commutator(Q,F,z) - sum
#     #     return L

#     def lambda2_matrix(z, kappas):
#         sum = np.zeros([p,p])
#         for k in range(q):
#             sum += kappas[k] * N_list[k]
#         L = 0.5 * c * np.outer(P @ z, P @ z) - F - sum
#         return L

#     def linear_pencil(vars):
#         z = kernel.function(vars[0:n])
#         kappas = vars[n:n+q]
#         l2 = vars[-1]
#         return lambda2_matrix(z, kappas) - l2 * Q

#     def Lz_constraint(vars):
#         z = kernel.function(vars[0:n])
#         return linear_pencil(vars) @ z

#     def lambda2_fun(z, kappas):
#         # sum = np.zeros([p,p])
#         # for k in range(p):
#         #     sum += kappas[k] * N_list[k]
#         return z.T @ lambda2_matrix(z, kappas) @ z

#     def lambda2_constraint(vars):
#         z = kernel.function(vars[0:n])
#         kappas = vars[n:n+q]
#         lambda2 = vars[-1]
#         return lambda2 - lambda2_fun(z, kappas)

#     # def kappa_constraint(vars):
#     #     z = kernel.function(vars[0:n])
#     #     kappas = vars[n:n+p]
#     #     sum = 0.0
#     #     for k in range(p):
#     #         sum += kappas[k] * z.T @ ( N_list[k] -  N_list[k].T ) @ z
#     #     return sum

#     def boundary_constraint(vars):
#         z = kernel.function(vars[0:n])
#         return z.T @ Q @ z - 1 # >= 0

#     def detL(vars):
#         detL = np.linalg.det( linear_pencil(vars) )
#         return detL

#     def objective(vars):
#         return np.hstack([
#                            Lz_constraint(vars),
#                            detL(vars),
#                            boundary_constraint(vars),
#                            lambda2_constraint(vars)
#                          ])

#     def stability(x, l, kappa):
#         z = kernel.function(x)
#         Jm = kernel.jacobian(x)
#         nablah = Jm.T @ Q @ z

#         sum = np.zeros([p,p])
#         for k in range(q):
#             sum += kappa[k] * N_list[k]

#         lambda0 = c * clf.function(x)

#         S = F + l * Q - lambda0 * P - sum - c * np.outer(P @ z, P @ z)
#         v = np.array([ nablah[1], -nablah[0] ])

#         return v.T @ Jm.T @ S @ Jm @ v

#     iterations = 0
#     while iterations < max_iter:

#         iterations += 1

#         initial_guess = find_nearest_boundary(cbf, initial_point)
#         # initial_guess = initial_point.tolist()

#         init_x = initial_guess
#         init_z = kernel.function(init_x)
#         init_kappas = np.random.randn(q)

#         init_l2 = init_z.T @ lambda2_matrix(init_z, init_kappas) @ init_z
#         initial_vars = np.hstack([ init_x, init_kappas, init_l2 ])

#         # Solve least squares problem with bounds on s and l2
#         lower_bounds = [ -np.inf for _ in range(n+q) ] + [ 0.0 ]
#         try:
#             LS_sol = least_squares( objective, initial_vars, method='trf', bounds=(lower_bounds, np.inf), max_nfev=500 )
#             if "unfeasible" not in LS_sol.message:
#                 equilibrium_point = LS_sol.x[0:n].tolist()
#                 lambda_sol = LS_sol.x[-1]
#                 kappa_sol = LS_sol.x[n:n+q].tolist()
#                 stability_value = stability( equilibrium_point, lambda_sol, kappa_sol )
#                 return {"cost": LS_sol.cost, "point": equilibrium_point, "lambda": lambda_sol, "stability": stability_value, "kappa": kappa_sol, "boundary_start": initial_guess, "message": LS_sol.message, "iterations": iterations }
#         except:
#             continue
    
# def compute_equilibria_algorithm8(plant, clf, cbf, initial_point, **kwargs):
#     '''
#     Compute the equilibrium points
#     '''
#     c = 1
#     max_iter = 1000
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue

#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()

#     ACCURACY = 0.000000000001

#     # QP parameters
#     QP_dim = n + p + 1
#     Cost_matrix = np.eye(QP_dim)
#     Cost_matrix[n,n] = 1
#     q = np.zeros(QP_dim)
#     QP = QuadraticProgram(P=Cost_matrix, q=q)

#     sample_time = 1e+0
#     initial_kappa = [ 0.0 for _ in range(p) ]
#     state_dynamics = Integrator( initial_point, np.zeros(n) )
#     kappa_dynamics = Integrator( initial_kappa, np.zeros(p) )

#     it = 0
#     curr_kappa = initial_kappa
#     curr_state = initial_point
#     state_log = [[np.inf, np.inf]]

#     cost = np.linalg.norm(  curr_state - state_log[-1] )
#     while cost > 0.001 and it < max_iter:
#         it += 1

#         z = kernel.function(curr_state)
#         Jm = kernel.jacobian(curr_state)

#         a = np.zeros(p)
#         sumN = np.zeros([p,p])
#         sum_kappa = np.zeros([p,p])
#         for i in range(p):
#             a[i] = z.T @ N_list[i] @ z
#             sumN[:,i] = N_list[i] @ z
#             sum_kappa += curr_kappa[i] * N_list[i]

#         L = ( 0.5 * c * np.outer(P @ z, P @ z) - F + sum_kappa )

#         alpha = z.T @ L @ z
#         vec = alpha * Q @ z - L @ z
#         M = 2 * L + c * (z.T @ P @ z) * P

#         nabla_alpha_x = z.T @ M @ Jm
#         nabla_alpha_kappa = a

#         h = cbf.evaluate_function(*curr_state)[0]
#         nablah = cbf.evaluate_gradient(*curr_state)[0]

#         V1 = (1/2)*(h**2)
#         nabla_V1_x = h*nablah
#         nabla_V1_kappa = np.zeros(p)

#         # a_V1 = np.hstack([ nabla_V1_x, nabla_V1_kappa, -1.0 ])
#         # b_V1 = -V1

#         V2 = (1/2)*(np.linalg.norm(vec)**2)
#         nabla_V2_x = vec.T @ ( np.outer(Q @ z, z) @ M + alpha * Q - L - 0.5 * c * ( z.T @ P @ z * P + np.outer( P @ z, P @ z ) ) ) @ Jm
#         nabla_V2_kappa = vec.T @ ( np.outer( Q @ z, a ) - sumN )

#         Matrix = alpha * Q - L
#         detMatrix = np.linalg.det(Matrix)
#         A = adjugate( Matrix )
#         symmA = 0.5 * ( A + A.T )
#         trace_vec = np.zeros(p)
#         for i in range(p):
#             trace_vec[i] = np.trace( A @ N_list[i] )

#         V3 = (1/2)*(detMatrix**2)
#         nabla_V3_x = detMatrix * z.T @ ( np.trace(A @ Q) * M - c * P @ symmA @ P ) @ Jm
#         nabla_V3_kappa = detMatrix * ( np.trace(A @ Q) * a.T - trace_vec.T )

#         # a_V2 = np.hstack([ nabla_V2_x, nabla_V2_kappa, -1.0 ])
#         # b_V2 = -V2

#         mu1, mu2, mu3 = 1, 1, 0.01
#         V = mu1 * V1 + mu2 * V2 + mu3 * V3
#         nabla_V_x = mu1*nabla_V1_x + mu2*nabla_V2_x + mu3 * nabla_V3_x
#         nabla_V_kappa = mu1*nabla_V1_kappa + mu2*nabla_V2_kappa + mu3 * nabla_V3_kappa

#         a_V = np.hstack([ nabla_V_x, nabla_V_kappa, -1.0 ])
#         b_V = -V

#         a_lambda = -np.hstack([ nabla_alpha_x, nabla_alpha_kappa, 0.0 ])
#         b_lambda = alpha

#         A = np.vstack([ a_V, a_lambda ])
#         b = np.hstack([ b_V, b_lambda ])

#         QP.set_inequality_constraints(A, b)
#         QP_sol = QP.get_solution()

#         w_control = QP_sol[0:n]

#         # mu = 1.0
#         # w_control = - mu * (1/np.linalg.norm(nablah)**2) * nablah * np.sign( h )

#         kappa_control = QP_sol[n:n+p]
#         delta = QP_sol[-1]

#         state_dynamics.set_control(w_control)
#         state_dynamics.actuate(sample_time)

#         kappa_dynamics.set_control(kappa_control) 
#         kappa_dynamics.actuate(sample_time)

#         state_log.append( curr_state.tolist() )
#         curr_state = state_dynamics.get_state()
#         curr_kappa = kappa_dynamics.get_state()
#         cost = V

#     kappas = kappa_dynamics.get_state()
#     return {"cost": cost, "point": curr_state, "lambda": alpha, "kappas": kappas }, state_log

'''
The following algorithms try to compute the equilibrium points by using the LinearMatrixPencil() class.
None of them work.
'''
# def compute_equilibria_using_pencil(plant, clf, cbf, initial_point, **kwargs):
#     '''
#     Compute the equilibrium points
#     '''
#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()
#     r = len(N_list)

#     ACCURACY = 1e-3

#     c = 1
#     max_iter = 1000
#     delta = 1e-1
#     limit_grad_norm = 1e+4
#     initial_kappa = [np.random.rand() for _ in range(r)]
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue
#         if aux_key == "delta":
#             delta = kwargs[key]
#             continue
#         if aux_key == "initial_kappa":
#             initial_kappa = kwargs[key]

#     def L_fun(x, kappa):
#         '''
#         L = 0.5 c m.T P m P - F + SUM κ_i N_i
#         '''
#         m = kernel.function(x)
#         sum_kappa = np.zeros([p,p])
#         for i in range(r):
#             sum_kappa += kappa[i] * N_list[i]
#         return 0.5 * c * (m.T @ P @ m) * P - F + sum_kappa

#     def cost_function(x, z):
#         '''
#         Cost function
#         '''
#         m = kernel.function(x)
#         return np.linalg.norm( m - z )

#     def cost_gradient_x(x, kappa, l, z):
#         '''
#         Gradient of the cost function 0.5||m - z||**2 w.r.t. x, where z is an eigenvector of the pencil (λ Q - L)
#         '''
#         m = kernel.function(x)
#         dm_dx = kernel.jacobian(x)

#         zzT = np.outer(z,z)
#         ProjQ = np.eye(p) - Q @ zzT

#         L = L_fun(x, kappa)
#         symL = L + L.T
#         Omega = l*Q - L + Q @ zzT @ symL
#         dz_dx = c * np.linalg.inv(Omega) @ ProjQ @ P @ np.outer(z, m) @ P @ dm_dx

#         # print("m = " + str(m))
#         # print("z = " + str(z))
#         # print("Omega^-1 = " + str(np.linalg.inv(Omega)))
#         # print("ProjQ = " + str(ProjQ))
#         # print("zmT = " + str(np.outer(z, m)))
#         # print("Jm = " + str(dm_dx))

#         # E = np.eye(p)
#         # Pz = P @ z
#         # dz_dm = c * np.linalg.inv(Omega) @ ProjQ @ np.array([ (m.T @ P @ E[:,i] * Pz).tolist() for i in range(p) ]).T

#         return (dm_dx - dz_dx).T @ (m - z)

#     def cost_gradient_kappa(x, kappa, l, z):
#         '''
#         Gradient of the cost function ||m(x) - z||^2 w.r.t. κ, where z is an eigenvector of the pencil (λ Q - L)
#         '''
#         zzT = np.outer(z,z)
#         ProjQ = np.eye(p) - Q @ zzT
#         m = kernel.function(x)

#         L = L_fun(x, kappa)
#         symL = L + L.T
#         Omega = l*Q - L + Q @ zzT @ symL
#         dz_dkappa = np.linalg.inv(Omega) @ ProjQ @ np.array([ (N_list[i] @ z).tolist() for i in range(r) ]).T

#         return -dz_dkappa.T @ (m - z)

#     def filter(pencil):
#         '''
#         Filters invalid eigenpairs
#         '''
#         valid_eigenvalues, valid_eigenvectors = [], []
#         for k in range(len(pencil.eigenvalues)):
#             eigenvalue = pencil.eigenvalues[k]
#             eigenvector = pencil.eigenvectors[k]
#             # Filters invalid eigenpairs
#             if (not np.isreal(eigenvalue)) or eigenvalue < 1e-4 or np.abs(eigenvector.T @ Q @ eigenvector - 1.0) > 1e-8:
#                 continue
#             valid_eigenvalues.append( eigenvalue )
#             valid_eigenvectors.append( eigenvector )

#         return valid_eigenvalues, np.array(valid_eigenvectors)

#     '''
#     Setup general cvxpy problem
#     '''
#     decision_var = cp.Variable(r)
#     cost_center_param = cp.Parameter(r)
#     kappa_param = cp.Parameter(r)
#     delta_param = cp.Parameter()

#     objective = cp.Minimize( cp.norm( decision_var - cost_center_param ) )
#     constraint = [ cp.sum([ kappa_param[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) - 2 * F
#                 - delta_param * cp.sum([ decision_var[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) >> 0 ]
#     problem = cp.Problem(objective, constraint)

#     '''
#     First, solve min ||κ - κ_initial|| s.t. L(x,κ) + L(x,κ).T >= 0, for κ
#     '''
#     cost_center_param.value = initial_kappa
#     kappa_param.value = np.zeros(r)
#     delta_param.value = -1
#     problem.solve()

#     # After solving for initial condition, change constraint
#     # constraint += [ cost_center_param.T @ decision_var >= 0 ]

#     curr_kappa = decision_var.value
#     curr_pt = initial_point
#     '''
#     The pencil (λ Q - L) must have non-negative spectra.
#     '''
#     pencil = LinearMatrixPencil( Q, L_fun(curr_pt, curr_kappa) )
#     eigenvalues, eigenvectors = filter(pencil)

#     for k in range(len(eigenvalues)):
#         eig = eigenvalues[k]
#         z = eigenvectors[k]
#         print("lambda = " + str(eig))
#         print("zQz = " + str(z.T @ Q @ z))

#     # Create copies of initial points with valid eigenpairs
#     curr_sols = []
#     for k in range(len(eigenvalues)):
#         curr_sols.append( {"x": curr_pt, "kappa": curr_kappa,
#                            "lambda": eigenvalues[k], "z": eigenvectors[k],
#                            "cost": cost_function(curr_pt, eigenvectors[k]), 
#                            "delta_cost": 0.0 } )

#     print("Initial conditions = ")
#     for curr_sol in curr_sols:
#         print(str( curr_sol["lambda"]))

#     sol_log = []
#     sol_log.append( curr_sols )

#     it = 0
#     total_cost = np.inf
#     limit_grad_norm = 1e+6
#     while total_cost > ACCURACY and it < max_iter:
#         it += 1
#         delta = 0.05

#         # For each existing solution...
#         for curr_sol in curr_sols:
        
#             # Compute new x
#             gradC_x = cost_gradient_x( curr_sol["x"], curr_sol["kappa"], curr_sol["lambda"], curr_sol["z"] )
#             if np.linalg.norm(gradC_x) > limit_grad_norm:
#                 gradC_x = gradC_x/limit_grad_norm
#             new_x = curr_sol["x"] - delta * gradC_x

#             # print("gradC_x = " + str(gradC_x))

#             # For each valid eigenpair, solve min ||∇κ - ∇κ_nom|| s.t. L(x,κ) + L(x,κ).T + δ SUM ∇κ_i (N_i + N_i.T) >= 0, for ∇κ
#             gradC_kappa = cost_gradient_kappa( curr_sol["x"], curr_sol["kappa"], curr_sol["lambda"], curr_sol["z"] )

#             cost_center_param.value = gradC_kappa
#             kappa_param.value = curr_sol["kappa"]
#             delta_param.value = delta
#             problem.solve()
#             if problem.status in ["infeasible", "unbounded"]:
#                 raise Exception("Problem is " + problem.status)
#             gradC_kappa = decision_var.value

#             if np.linalg.norm(gradC_kappa) > limit_grad_norm:
#                 gradC_kappa = gradC_kappa/limit_grad_norm

#             # Compute new kappa and compute pencil, eliminating invalid eigenpairs
#             new_kappa = curr_sol["kappa"] - delta * gradC_kappa

#             # print("gradC_kappa = " + str(gradC_kappa))

#             # print("inner = " + str( gradC_kappa.T @ cost_center_param.value ))
#             # print("λ = " + str(pencil.eigenvalues))
#             # print("zQzs = " + str(pencil.zQzs))

#             # Updates x and kappa
#             curr_sol["x"] = new_x
#             curr_sol["kappa"] = new_kappa

#             # Update pencil and eliminate invalid eigenpairs
#             pencil.set_pencil(B = L_fun(curr_sol["x"], curr_sol["kappa"]))
#             eigenvalues, eigenvectors = filter(pencil)

#             # Finds best cost after update for all valid eigenvectors
#             old_cost = curr_sol["cost"]
#             costs = [ cost_function( curr_sol["x"], eigenvector ) for eigenvector in eigenvectors ]
#             min_index = np.argmin(costs)

#             # Updates λ, z and cost - if everything is working, cost should -->> 0
#             curr_sol["lambda"] = eigenvalues[min_index]
#             curr_sol["z"] = eigenvectors[min_index]
#             curr_sol["cost"] = costs[min_index]
#             curr_sol["delta_cost"] = curr_sol["cost"] - old_cost

#         print(str(it) + " iterations...")
#         for curr_sol in curr_sols:
#             pass
#             # print("x = " + str( curr_sol["x"]))
#             # print("κ = " + str( curr_sol["kappa"]))
#             # print("λ = " + str( curr_sol["lambda"]))
#             # print("m = " + str(kernel.function(curr_sol["x"])))
#             # print("z = " + str(curr_sol["z"]))
#             print("cost = " + str( curr_sol["cost"]))
#             print("Δ cost = " + str( curr_sol["delta_cost"]))

#         sol_log.append( curr_sols )

#     return curr_sols, sol_log

# def compute_equilibria_using_pencil2(plant, clf, cbf, initial_point, **kwargs):
#     '''
#     Compute the equilibrium points
#     '''
#     if clf._dim != cbf._dim:
#         raise Exception("CLF and CBF must have the same dimension.")
#     n = clf._dim

#     F = plant.get_F()
#     P = clf.P
#     Q = cbf.Q
#     if clf.kernel != cbf.kernel:
#         raise Exception("CLF and CBF must be based on the same kernel.")
#     kernel = clf.kernel
#     p = kernel.kernel_dim
#     N_list = kernel.get_N_matrices()
#     r = len(N_list)

#     ACCURACY = 1e-3

#     c = 1
#     max_iter = 1000
#     limit_grad_norm = 1e+4
#     initial_kappa = [np.random.rand() for _ in range(r)]
#     for key in kwargs.keys():
#         aux_key = key.lower()
#         if aux_key == "c":
#             c = kwargs[key]
#             continue
#         if aux_key == "max_iter":
#             max_iter = kwargs[key]
#             continue
#         if aux_key == "delta":
#             delta = kwargs[key]
#             continue
#         if aux_key == "initial_kappa":
#             initial_kappa = kwargs[key]

#     def L_fun(x, kappa):
#         '''
#         L = 0.5 c m.T P m P - F + SUM κ_i N_i
#         '''
#         m = kernel.function(x)
#         sum_kappa = np.zeros([p,p])
#         for i in range(r):
#             sum_kappa += kappa[i] * N_list[i]
#         return 0.5 * c * (m.T @ P @ m) * P - F + sum_kappa

#     def filter(pencil):
#         '''
#         Filters invalid eigenpairs
#         '''
#         valid_eigenvalues, valid_eigenvectors = [], []
#         for k in range(len(pencil.eigenvalues)):
#             eigenvalue = pencil.eigenvalues[k]
#             eigenvector = pencil.eigenvectors[k]
#             # Filters invalid eigenpairs
#             if (not np.isreal(eigenvalue)) or eigenvalue < 1e-4 or np.abs(eigenvector.T @ Q @ eigenvector - 1.0) > 1e-8:
#                 continue
#             valid_eigenvalues.append( eigenvalue )
#             valid_eigenvectors.append( eigenvector )

#         return valid_eigenvalues, np.array(valid_eigenvectors)

#     '''
#     Setup SDP problem for kappa computation
#     '''
#     kappa_var = cp.Variable(r)
#     kappa_param = cp.Parameter(r)
#     objective = cp.Minimize( cp.norm( kappa_var - kappa_param ) )
#     constraint = [ cp.sum([ kappa_var[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) - 2 * F >> 0 ]
#     problem = cp.Problem(objective, constraint)

#     '''
#     Setup linear matrix pencil
#     '''
#     pencil = LinearMatrixPencil( Q, L_fun(initial_point, initial_kappa) )

#     def cost_function(var):
#         '''
#         Cost function, depending on x and kappa
#         '''
#         x = var[0:n]
#         kappa = var[n:]

#         # Solve SDP optimization
#         kappa_param.value = kappa
#         problem.solve()
#         kappa = kappa_var.value

#         # Update pencil and filters invalid eigenpairs
#         pencil.set_pencil(B = L_fun(x, kappa))
#         eigenvalues, eigenvectors = filter(pencil)

#         m = kernel.function(x)
#         costs = [ np.linalg.norm( m - eigenvector ) for eigenvector in eigenvectors ]
                
#         return np.array( np.min(costs) )

#     sol = least_squares( cost_function, initial_point + initial_kappa, max_nfev=1000 )

#     if sol.success:
#         print("Solution was found.\n")
#         return { "x": sol.x[0:n].tolist(), "kappa": sol.x[n:].tolist(), "cost": sol.cost }
#     else:
#         print("Algorithm exited with the following error: \n")
#         print(sol.message)
#         return initial_point

# def compute_equilibria_using_pencil3(plant, clf, cbf, initial_point, **kwargs):
    # '''
    # Compute the equilibrium points.
    # '''
    # c = 1
    # max_iter = 1000
    # for key in kwargs.keys():
    #     aux_key = key.lower()
    #     if aux_key == "c":
    #         c = kwargs[key]
    #         continue
    #     if aux_key == "max_iter":
    #         max_iter = kwargs[key]
    #         continue

    # if clf._dim != cbf._dim:
    #     raise Exception("CLF and CBF must have the same dimension.")
    # n = clf._dim

    # F = plant.get_F()
    # P = clf.P
    # Q = cbf.Q
    # if clf.kernel != cbf.kernel:
    #     raise Exception("CLF and CBF must be based on the same kernel.")
    # kernel = clf.kernel
    # p = kernel.kernel_dim
    # N_list = kernel.get_N_matrices()
    # r = len(N_list)

    # '''
    # Setup x and kappa dynamics
    # '''
    # initial_kappa = [ np.random.rand() for _ in range(r) ]
    # x_dynamics = Integrator( initial_point, np.zeros(n) )
    # kappa_dynamics = Integrator( initial_kappa, np.zeros(r) )

    # '''
    # Setup initial optimization problem 
    # '''
    # kappa_var = cp.Variable(r)
    # kappa_param = cp.Parameter(r)
    # init_objective = cp.Minimize( cp.norm( kappa_var - kappa_param ) )
    # init_constraint = [ cp.sum([ kappa_var[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) - 2 * F >> 0 ]
    # init_problem = cp.Problem(init_objective, init_constraint)

    # '''
    # Setup main optimization problem
    # '''
    # u_x = cp.Variable(n)
    # u_kappa = cp.Variable(r)

    # gradCx = cp.Parameter(n)
    # gradCkappa = cp.Parameter(r)
    # kappa = cp.Parameter(r)
    # cost = cp.Parameter()

    # alpha, beta = 1, 1
    # objective = cp.Minimize( cp.norm(u_x)**2 + cp.norm(u_kappa)**2 )
    # CLF_constr = [ gradCx.T @ u_x + gradCkappa.T @ u_kappa + alpha * cost <= 0.0 ]
    # B = cp.sum([ kappa[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) - 2 * F
    # MCBF_constr = [ cp.sum([ u_kappa[k] * (N_list[k] + N_list[k].T) for k in range(r) ]) + beta * B >> 0 ]
    # problem = cp.Problem(objective, CLF_constr + MCBF_constr)

    # '''
    # Define filtering function for the pencil
    # '''
    # def filter(pencil):
    #     '''
    #     Filters invalid eigenpairs
    #     '''
    #     valid_eigenvalues, valid_eigenvectors = [], []
    #     for k in range(len(pencil.eigenvalues)):
    #         eigenvalue = pencil.eigenvalues[k]
    #         eigenvector = pencil.eigenvectors[k]
    #         # Filters invalid eigenpairs
    #         if (not np.isreal(eigenvalue)) or eigenvalue < 1e-4 or np.abs(eigenvector.T @ Q @ eigenvector - 1.0) > 1e-8:
    #             continue
    #         valid_eigenvalues.append( eigenvalue )
    #         valid_eigenvectors.append( eigenvector )

    #     return valid_eigenvalues, np.array(valid_eigenvectors).T

    # '''
    # Some collection of useful methods
    # '''
    # def L_fun(x, kappa):
    #     '''
    #     L = 0.5 c m.T P m P - F + SUM κ_i N_i
    #     '''
    #     m = kernel.function(x)
    #     sum_kappa = np.zeros([p,p])
    #     for i in range(r):
    #         sum_kappa += kappa[i] * N_list[i]
    #     return 0.5 * c * (m.T @ P @ m) * P - F + sum_kappa

    # def get_updated_sol(new_pencil, old_sol):
    #     '''
    #     This function compares the eigenpairs of the updated pencil with the eigenpairs of the previous solution candidate,
    #     and returns the closest eigenpair.
    #     ASSUMPTION: the updated pencil is on a neighborhood of the previous pencil
    #     '''
    #     newL = new_pencil._B
    #     oldL = L_fun(old_sol["x"], old_sol["kappa"])
    #     deltaL = newL - oldL

    #     l = old_sol["lambda"]
    #     z = np.array(old_sol["z"])

    #     zzT = np.outer(z,z)
    #     ProjQ = np.eye(p) - Q @ zzT

    #     symL = oldL + oldL.T
    #     Omega = l*Q - oldL + Q @ zzT @ symL
    #     expected_delta_z = np.linalg.inv(Omega) @ ProjQ @ deltaL @ z
    #     expected_delta_l = z.T @ symL @ expected_delta_z + z.T @ deltaL @ z

    #     new_eigenvalues, new_eigenvectors = filter(new_pencil)
    #     costs = [ np.abs( new_eigenvalues[k] - (l + expected_delta_l) )**2 + np.linalg.norm( new_eigenvectors[:,k] - ( z + expected_delta_z ) )**2 for k in range(len(eigenvalues)) ]
    #     min_index = np.argmin(costs)

    #     return new_eigenvalues[min_index], new_eigenvectors[:,min_index].tolist()
        
    # def cost_function(x, z):
    #     '''
    #     Convex cost for minimization (aka, "Lyapunov" function of the problem)
    #     '''
    #     return 0.5 * np.linalg.norm( kernel.function(x) - np.array(z) )**2

    # def cost_gradient_x(sol):
    #     '''
    #     Gradient of the cost function 0.5||m(x) - z||**2 w.r.t. x, where z is an eigenvector of the pencil (λ Q - L)
    #     '''
    #     x, kappa, l, z = np.array(sol["x"]), sol["kappa"], sol["lambda"], np.array(sol["z"])

    #     m = kernel.function(x)
    #     dm_dx = kernel.jacobian(x)

    #     zzT = np.outer(z,z)
    #     ProjQ = np.eye(p) - Q @ zzT

    #     L = L_fun(x, kappa)
    #     symL = L + L.T
    #     Omega = l*Q - L + Q @ zzT @ symL
    #     dz_dx = c * np.linalg.inv(Omega) @ ProjQ @ P @ np.outer(z, m) @ P @ dm_dx

    #     return (dm_dx - dz_dx).T @ (m - z)

    # def cost_gradient_kappa(sol):
    #     '''
    #     Gradient of the cost function 0.5||m(x) - z||**2 w.r.t. κ, where z is an eigenvector of the pencil (λ Q - L)
    #     '''
    #     x, kappa, l, z = np.array(sol["x"]), sol["kappa"], sol["lambda"], np.array(sol["z"])

    #     zzT = np.outer(z,z)
    #     ProjQ = np.eye(p) - Q @ zzT
    #     m = kernel.function(x)

    #     L = L_fun(x, kappa)
    #     symL = L + L.T
    #     Omega = l*Q - L + Q @ zzT @ symL
    #     dz_dkappa = np.linalg.inv(Omega) @ ProjQ @ np.array([ (N_list[i] @ z).tolist() for i in range(r) ]).T

    #     return -dz_dkappa.T @ (m - z)

    # '''
    # Initialize linear matrix pencil with a valid kappa
    # '''
    # kappa_param.value = initial_kappa
    # init_problem.solve()
    # initial_kappa = kappa_var.value

    # pencil = LinearMatrixPencil( Q, L_fun(initial_point, initial_kappa) ) # ---> this pencil should have non-negative spectra
    # eigenvalues, eigenvectors = filter(pencil)

    # '''
    # Populate initial solutions with all valid eigenpairs
    # '''
    # base_sample_time = 1e-2
    # curr_sols = []
    # for k in range(len(eigenvalues)):
    #     z = eigenvectors[:,k].tolist()
    #     curr_sols.append( {"x": initial_point, "kappa": initial_kappa,
    #                        "lambda": eigenvalues[k], "z": z,
    #                        "cost": cost_function(initial_point, z), "delta_cost": 0.0,
    #                        "sample_time": base_sample_time } )
    
    # '''
    # Main optimization loop
    # '''
    # it = 0
    # ACCURACY = 1e-10
    # total_cost = np.inf
    # while total_cost > ACCURACY and it < max_iter:
    #     it += 1
        
    #     # Loop for every solution
    #     total_cost = 0.0
    #     for curr_sol in curr_sols:

    #         # Solve the CLF-MCBF optimization problem to find the optimal directions
    #         gradCx.value = cost_gradient_x(curr_sol)
    #         gradCkappa.value = cost_gradient_kappa(curr_sol)
    #         kappa.value = curr_sol["kappa"]
    #         cost.value = curr_sol["cost"]
    #         problem.solve()

    #         # print(problem.status)

    #         # print("u_x = " + str(u_x.value))
    #         # print("u_kappa = " + str(u_kappa.value))

    #         # Actuate (x,κ) dynamics
    #         x_dynamics.set_state(curr_sol["x"])
    #         x_dynamics.set_control(u_x.value.tolist())
    #         x_dynamics.actuate(curr_sol["sample_time"])

    #         kappa_dynamics.set_state(curr_sol["kappa"])
    #         kappa_dynamics.set_control(u_kappa.value.tolist())
    #         kappa_dynamics.actuate(curr_sol["sample_time"])

    #         # Update solution
    #         old_sol = curr_sol
            
    #         curr_sol["x"] = x_dynamics.get_state().tolist()
    #         curr_sol["kappa"] = kappa_dynamics.get_state().tolist()
    #         pencil.set_pencil(B = L_fun(curr_sol["x"], curr_sol["kappa"]))
    #         curr_sol["lambda"], curr_sol["z"] = get_updated_sol(pencil, old_sol)

    #         curr_sol["cost"] = cost_function( curr_sol["x"], curr_sol["z"] )
    #         curr_sol["delta_cost"] = curr_sol["cost"] - old_sol["cost"]

    #         '''
    #         If delta_cost is positive, ignore and reduce sample time.
    #         '''
    #         if curr_sol["delta_cost"] >= 0.0:
    #             curr_sol = old_sol
    #             curr_sol["sample_time"] = 0.9 * curr_sol["sample_time"]

    #         '''
    #         If delta_cost is negative, but small, increase sample time
    #         '''
    #         if curr_sol["delta_cost"] < 0.0 and np.abs(curr_sol["delta_cost"]) < 0.1:
    #             curr_sol["sample_time"] = 1.1 * curr_sol["sample_time"]

    #         total_cost += curr_sol["cost"]

    #     print(str(it) + " iterations...")
    #     cost_list, lambda_list, z_list, x_list = [], [], [], []
    #     for curr_sol in curr_sols:
    #         cost_list.append(curr_sol["cost"])
    #         lambda_list.append(curr_sol["lambda"])
    #         z_list.append(curr_sol["z"])
    #         x_list.append(curr_sol["x"])
    #         # print("x = " + str( curr_sol["x"]))
    #         # print("κ = " + str( curr_sol["kappa"]))
    #         # print("λ = " + str( curr_sol["lambda"]))
    #         # print("m = " + str(kernel.function(curr_sol["x"])[0]))
    #         # print("z = " + str(curr_sol["z"][0]))
    #         # print("cost = " + str(curr_sol["cost"]))
    #         # print("Δ cost = " + str( curr_sol["delta_cost"]))
    #     print("costs = " + str(cost_list))
    #     print("λ = " + str(lambda_list))
    #     # print("z = " + str(np.array(z_list)))
    #     # print("x = " + str(np.array(x_list)))

    # return curr_sols