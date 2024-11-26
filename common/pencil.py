import warnings
import numpy as np
import scipy as sp

from dataclasses import dataclass
from scipy.linalg import null_space
from numpy.polynomial import Polynomial as Poly

from .polynomial import MatrixPolynomial

def solve_poly_linearsys(T: np.ndarray, S: np.ndarray, b_poly: np.ndarray) -> np.ndarray:
    '''
    Finds the polynomial array x(λ) that solves (λ T - S) x(λ) = b(λ), where T, S are n x n (n=1 or n=2)
    and b(λ) is a polynomial array of size nxr or nxr.

    Input: - matrices T, S from linear matrix pencil (λ T - S)
    '''
    if isinstance(T, (int, float)): T = np.array([[ T ]])
    if isinstance(S, (int, float)): S = np.array([[ S ]])

    if T.shape != S.shape:
        raise TypeError("T and S must have the same shape.")
    
    if T.shape[0] != T.shape[1]:
        raise TypeError("T and S must be square matrices.")

    n = T.shape[1]
    r = b_poly.shape[1]

    if n != b_poly.shape[0]:
        raise TypeError("Number of lines in (λ T - S) and b(λ) must be the same.")

    # Extract max. degree of b_poly
    max_deg = 0
    for (i,j), poly in np.ndenumerate(b_poly):
        if not isinstance( poly, Poly ):
            raise TypeError("b(λ) is not an array of polynomials.")
        max_deg = max( max_deg, poly.degree() )

    # Initialize and populate bsys
    bsys = np.zeros(((max_deg+1)*n,r))
    for (i,j), poly in np.ndenumerate(b_poly):
        for k, c in enumerate(poly.coef):
            bsys[ k * n + i, j ] = c

    #  Initialize and populate Asys
    Asys = np.zeros(((max_deg+1)*n, max_deg*n))
    for i in range(max_deg):
        Asys[ i*n:(i+1)*n , i*n:(i+1)*n ] = -S
        Asys[ (i+1)*n:(i+2)*n , i*n:(i+1)*n ] = T

    results = np.linalg.lstsq(Asys, bsys, rcond=None)
    x_coefs = results[0]
    res = results[1]
    residue = np.linalg.norm(res)

    residue_tol = 1e-11
    if residue > residue_tol:
        warnings.warn(f"Large residue detected on linear system solution = {residue}")

    if max_deg == 0: max_deg = 1
    x_poly = np.array([[ Poly([0.0 for _ in range(max_deg) ], symbol='λ') for j in range(r) ] for i in range(n) ])
    for (i,j), c in np.ndenumerate(x_coefs):
        exp = int(i/n)
        x_poly[i%n,j].coef[exp] = c

    return x_poly

@dataclass
class Eigen():
    ''' 
    Data class for generalized eigenvalues/eigenvectors.
    Holds: - (α, β) polar form of eigenvalue
           - left/right eigenvectors
           - eigenvalue inertia, if real
    '''
    alpha: complex
    beta: float
    eigenvalue: complex
    rightEigenvectors: list | np.ndarray
    leftEigenvectors: list | np.ndarray
    inertia: float                  # value of z_left' M z_right

class MatrixPencil(MatrixPolynomial):
    '''
    Class for linear matrix pencils of the form P(λ) = λ M - N (derived from MatrixPolynomial class)
    Each object holds: - its M, N matrices;
                       - a list with all pencil eigenvalues in standard/polar form and corresponding left/right eigenvectors;
    '''
    def __init__(self, M: list | np.ndarray, N: list | np.ndarray, **kwargs):

        if isinstance(M, list): M = np.array(M)
        if isinstance(N, list): N = np.array(N)
        
        if M.ndim != 2 or N.ndim != 2:
            raise TypeError("M and N must be two dimensional arrays.")

        if M.shape != N.shape:
            raise TypeError("Matrix dimensions are not equal.")
        
        self.N, self.M = N, M
        super().__init__(coef=[ -N, M ], **kwargs)

        '''
        Uses QZ algorithm to decompose the two pencil matrices into M = Q MM Z' and N = Q NN Z',
        where MM is block upper triangular, NN is upper triangular and Q, Z are unitary matrices.
        '''
        if self.type == 'regular':
            self.NN, self.MM, self.alphas, self.betas, self.Q, self.Z = sp.linalg.ordqz(self.N, self.M, output='real')
            self.eigens = self._eigen(self.alphas, self.betas)

    def __call__(self, alpha: int | float, beta: int | float = 1.0) -> np.ndarray:
        '''
        Returns pencil value.
        If only one argument is passed, it is interpreted as the λ value and method returns matrix P(λ) = λ M - N.
        If two arguments are passed, they are interpreted as α, β values and method returns P(α, β) = α M - β N.
        '''
        if beta != 1:
            return alpha * self.M - beta * self.N
        else:
            return super().__call__(alpha)

    def _blocks(self) -> tuple[ list[Poly], list[np.ndarray[Poly]] ]:
        '''  
        Computes information about blocks of the QZ decomposition.
        Returns: - list of block poles
                 - list of block adjoint matrices
        '''
        n = self.M.shape[0]

        ''' Computes the block poles and adjoint matrices '''
        blk_poles, blk_adjs = [], []
        i = 0
        while i < n:
            # 2X2 BLOCKS OF COMPLEX CONJUGATE PENCIL EIGENVALUES
            if i < n-1 and self.NN[i+1,i] != 0.0:

                MMblock = self.MM[i:i+2,i:i+2]      # this is diagonal
                NNblock = self.NN[i:i+2,i:i+2]      # this is full 2x2

                a = np.linalg.det(MMblock)
                b = -( MMblock[0,0] * NNblock[1,1] + MMblock[1,1] * NNblock[0,0] )
                c = np.linalg.det(NNblock)
                blk_poles.append( Poly([ c, b, a ], symbol=self.symbol) )

                adj11 = Poly([ -NNblock[1,1],  MMblock[1,1] ], symbol=self.symbol)
                adj12 = Poly([ NNblock[0,1] ], symbol=self.symbol)
                adj21 = Poly([ NNblock[1,0] ], symbol=self.symbol)
                adj22 = Poly([ -NNblock[0,0],  MMblock[0,0] ], symbol=self.symbol)
                blk_adjs.append( np.array([[ adj11, adj12 ],[ adj21, adj22 ]]) )

                i+=2
            # 1X1 BLOCKS OF REAL PENCIL EIGENVALUES
            else:
                MMblock = self.MM[i,i]
                NNblock = self.NN[i,i]

                blk_poles.append( Poly([ -NNblock, MMblock ], symbol=self.symbol) )
                blk_adjs.append( np.array([Poly(1.0, symbol=self.symbol)]) )

                i+=1
                
        return blk_poles, blk_adjs

    def _eigen(self, alphas: np.ndarray, betas: np.ndarray) -> list[Eigen]:
        '''
        Computes generalized eigenvalues/eigenvectors from polar eigenvalues.
        '''
        if len(alphas) != len(betas):
            raise TypeError("The same number of polar eigenvalues must be passed.")

        eigens: list[Eigen] = []
        for alpha, beta in zip(alphas, betas):

            P = self(alpha, beta)
            zRight = null_space(P, rcond=1e-11)
            zLeft = null_space(P.T, rcond=1e-11)

            zRight = zRight.reshape(self.shape[0], )
            zLeft = zLeft.reshape(self.shape[0], )

            inertia = zLeft.T @ self.M @ zRight

            if beta != 0:
                eigens.append( Eigen(alpha, beta, alpha/beta, zRight, zLeft, inertia) )
            else:
                eigens.append( Eigen(alpha, 0.0, np.inf if alpha.real > 0 else -np.inf, zRight, zLeft, inertia) )

        return eigens

    def real_eigen(self) -> list[Eigen]:
        '''
        Returns an Eigen list with sorted real eigenvalues
        '''
        realAlphas, realBetas = [], []
        for eig in self.eigens:
            if np.abs(eig.alpha.imag) < self.realEigenTol:
                realAlphas.append( eig.alpha.real )
                realBetas.append( eig.beta )

        realEigens = self._eigen(realAlphas, realBetas)
        realEigens.sort(key=lambda eig: eig.eigenvalue)

        return realEigens

    def inverse(self) -> tuple[ Poly, MatrixPolynomial ]:
        '''
        Returns: - polynomial determinant det(λ)
                 - pencil adjoint polynomial matrix adj(λ)
        Used to compute the pencil inverse P(λ)^(-1) = det(λ)^(-1) adj(λ).

        TO DO (in the future): generalize the computation of the inverse for general MatrixPolynomials
        '''
        if not self.is_square:
            raise Exception("Inverse is not defined for non-square matrix polynomials.")

        n, m = self.shape[0], self.shape[1]

        ''' Computes blocks of the QZ decomposition '''
        blk_poles, blk_adjs = self._blocks()
        blk_dims = [ pole.degree() for pole in blk_poles ]

        ''' Computes pencil determinant '''
        determinant = np.prod(blk_poles)

        ''' Computes the pencil adjoint matrix '''
        num_blks = len(blk_poles)
        adjoint_arr = np.array([[ Poly([0.0], symbol=self.symbol) for _ in range(n) ] for _ in range(n) ])

        # Iterate over each block, starting by the last one
        for i in range(num_blks-1, -1, -1):
            blk_i_slice = slice( sum(blk_dims[0:i]), sum(blk_dims[0:i+1]) )

            for j in range(i, num_blks):
                blk_j_slice = slice( sum(blk_dims[0:j]), sum(blk_dims[0:j+1]) )

                ''' 
                j == i: Computes ADJOINT DIAGONAL BLOCKS
                j != i: Computes ADJOINT UPPER TRIANGULAR BLOCKS
                '''
                if j == i:
                    poles_ij = np.array([[ np.prod([ pole for k, pole in enumerate(blk_poles) if k != j ]) ]])
                    Lij = poles_ij * blk_adjs[j]
                else:
                    Tii = self.MM[ blk_i_slice, blk_i_slice ]
                    Sii = self.NN[ blk_i_slice, blk_i_slice ]

                    b_poly = np.array([[ Poly([0.0], symbol=self.symbol) for _ in range(blk_dims[j]) ] for _ in range(blk_dims[i]) ])
                    for k in range(i+1, j+1):
                        blk_k_slice = slice( sum(blk_dims[0:k]), sum(blk_dims[0:k+1]) )

                        # Compute polynomial (λ Tik - Sik) and get the kj slice of adjoint
                        Tik = self.MM[ blk_i_slice, blk_k_slice ]
                        Sik = self.NN[ blk_i_slice, blk_k_slice ]
                        poly_ik = np.array([[ Poly([ -Sik[a,b], Tik[a,b] ], symbol=self.symbol) for b in range(Tik.shape[1]) ] for a in range(Tik.shape[0]) ])
                        adjoint_kj = adjoint_arr[ blk_k_slice, blk_j_slice ]

                        b_poly -= poly_ik @ adjoint_kj

                    Lij = solve_poly_linearsys( Tii, Sii, b_poly )

                # Populate adjoint matrix
                adjoint_arr[ blk_i_slice, blk_j_slice ] = Lij

        adjoint = MatrixPolynomial.from_array( self.Z @ adjoint_arr @ self.Q.T )

        return determinant, adjoint

    def update(self, **kwargs):
        ''' 
        Update method for Linear Matrix Pencil. Existing coefficients can be modified, 
        but the pencil shape cannot be changed after creation.

        Inputs: - M np.array
                - N np.array        for pencil P(λ) = λ M - N
        '''
        N, M = self.N, self.M
        for key in kwargs.keys():
            if key == 'M':
                M = kwargs['M']
                continue
            if key == 'N':
                N = kwargs['N']
                continue

        super().update(coef=[-N, M])
        self.N, self.M = -self.coef[0], self.coef[1]

        # Recomputes eigenvalues
        if self.type == 'regular':
            self.NN, self.MM, self.alphas, self.betas, self.Q, self.Z = sp.linalg.ordqz(self.N, self.M, output='real')
            self.eigens = self._eigen(self.alphas, self.betas)