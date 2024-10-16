import numpy as np
import pandas as pd
import scipy
from preconditioner import MultiphaseBlockPreconditioner
import matplotlib.pyplot as plt
from utils import PI
from utils import fill_sol_and_RHS_vecs, print_norms
from scipy.sparse.linalg import gmres, LinearOperator, eigsh
from numpy.linalg import eig
from scipy.linalg import lstsq, solve, svd
from pyamg.krylov import fgmres
from sympy import Matrix, pretty
from petsc4py import PETSc
from slepc4py import SLEPc

def main(n: int = 4, c: int = 1, d: int = -1, xi: float = 1.0, mu: float = 1.0):
    """
    This solves the linear system Ax=b arising from the discretized multiphase momentum and incompressibility equations.   

    Args:
        n (int): Size of n x n matrix A. 
        c (int): Coefficient of 
        d (int): Coefficient of
        xi (float): drag coefficient
        mu (float): dynamic viscosity 
    
    Returns:
        Unpreconditioned matrix A of the form (F G; -D 0)
        Right-preconditioned matrix of the form A * M^-1 where M = (F G; 0 -S)
        S*Sinv
    """

    # Define the grid spacing
    dx = 1/n
    dy = 1/n
    
    # Define material specific parameters
    nu = 1.0
    etan = 1.0
    etas = 1.0

    # Build the big A matrix
    block_prec = MultiphaseBlockPreconditioner(n,xi,mu)
    A, S, F, D, G = block_prec.get_big_A_matrix(c=c, d_u=d)  
    # plt.spy(A)
    # plt.show()

    # Solution components
    u_n_x_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)  # x-component of Un
    u_n_y_fcn = lambda y,x: np.cos(2*PI*x)*np.sin(2*PI*y)  # y-component of Un

    u_s_x_fcn = lambda y,x: -np.sin(2*PI*x)*np.cos(2*PI*y)  # x-component of Us
    u_s_y_fcn = lambda y,x: -np.cos(2*PI*x)*np.sin(2*PI*y)  # y-component of Us

    p_fcn = lambda y,x: 0.0     # Pressure

    # ########## For Thn = 0.75 ###########
    # # RHS vector components for constant thn. 
    # b_n_x_fcn = lambda y,x: (3*(2*c*nu-d*(16*etan*nu*PI*PI+xi))*np.cos(2*PI*y)*np.sin(2*PI*x))/(8*nu)
    # b_n_y_fcn = lambda y,x: (3*(2*c*nu-d*(16*etan*nu*PI*PI+xi))*np.cos(2*PI*x)*np.sin(2*PI*y))/(8*nu)

    # b_s_x_fcn = lambda y,x: ((-2*c*nu+16*d*etas*nu*PI*PI+3*d*xi)*np.cos(2*PI*y)*np.sin(2*PI*x))/(8*nu)
    # b_s_y_fcn = lambda y,x: ((-2*c*nu+16*d*etas*nu*PI*PI+3*d*xi)*np.cos(2*PI*x)*np.sin(2*PI*y))/(8*nu)

    # b_p_fcn = lambda y,x: -2*PI*np.cos(2*PI*x)*np.cos(2*PI*y)

    ########### For Thn = 0.25*np.sin(2*PI*x)*np.sin(2*PI*y) + 0.5 ############
    # RHS vector components for variable thn. 
    b_n_x_fcn = lambda y,x: (np.cos(2*PI*y)*np.sin(2*PI*x)*(4*c*nu-4*d*(8*etan*nu*PI*PI+xi)+2*nu*(c-16*d*etan*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)+d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)
    b_n_y_fcn = lambda y,x: (np.cos(2*PI*x)*np.sin(2*PI*y)*(4*c*nu-4*d*(8*etan*nu*PI*PI+xi)+2*nu*(c-16*d*etan*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)+d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)

    b_s_x_fcn = lambda y,x: (np.cos(2*PI*y)*np.sin(2*PI*x)*(-4*c*nu+4*d*(8*etas*nu*PI*PI+xi)+2*nu*(c-16*d*etas*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)-d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)
    b_s_y_fcn = lambda y,x: (np.cos(2*PI*x)*np.sin(2*PI*y)*(-4*c*nu+4*d*(8*etas*nu*PI*PI+xi)+2*nu*(c-16*d*etas*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)-d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)

    b_p_fcn = lambda y,x: -PI*np.sin(4*PI*x)*np.sin(4*PI*y)

    u_vec, b_vec, u, p, b_u, b_p = fill_sol_and_RHS_vecs(n, u_n_x_fcn, u_n_y_fcn, u_s_x_fcn, u_s_y_fcn, p_fcn, 
                                    b_n_x_fcn, b_n_y_fcn, b_s_x_fcn, b_s_y_fcn, b_p_fcn)
    
    def print_true_res_norm(A, b_vec):
        iteration = 0
        def callback(xk):
            nonlocal iteration
            iteration += 1
            residual = b_vec - (A @ xk)   # Compute the true residual
            residual_norm = np.linalg.norm(residual)
            rel_res_norm = np.linalg.norm(residual)/np.linalg.norm(b_vec)
            print(f"GMRES Iteration {iteration}: True residual norm = {residual_norm}, Rel residual norm: {rel_res_norm}")
        return callback
    
    x_initial = np.zeros(2*n*n+2*n*n+n*n)
    print(f"\nPrinting error norms for solving Ax=b using fGMRES without preconditioner:")
    # u_approx, _ = gmres(A, b_vec, M=None, x0 = x_initial, rtol=1e-14, restart=20, maxiter=30, callback=print_true_res_norm(A,b_vec), callback_type='x')                       
    u_approx, _ = fgmres(A, b_vec, M=None, x0 = x_initial, tol=1e-14, maxiter=100, callback=print_true_res_norm(A, b_vec))  
    print_norms(u_approx, u_vec, dx, dy, n)

    # Define the combined matvec function encapsulating all schur complement operations.
    def exact_schur_op(v):
        A_inv = scipy.linalg.pinv(F)  # top left block
        Ainv_v = lstsq(F,  v[:F.shape[1]])[0]  # Dense solve on top left block. Note: lstsq, solve and matmul(A_inv,b_u) give same result
        rhs_interim = np.matmul(D, Ainv_v) +  v[F.shape[1]:]
        x_p = -1.0*gmres(S, rhs_interim)[0]  # Using GMRES here gives 2nd order convergence, but lstsq doesn't and matmul(S_inv, rhs) doesn't.
        G_xp = np.matmul(G, x_p)
        Ainv_G_xp = np.matmul(A_inv, G_xp) # OR solve(F, G_Sinv)
        u_schur = Ainv_v - Ainv_G_xp  
        # print(f"u_schur is {u_schur}")
        u_approx = np.concatenate((u_schur, x_p), axis=0)
        return u_approx

    u_approx = exact_schur_op(b_vec)
    print(f"\nPrinting error norms for solving Ax=b using schur complement:")
    print_norms(u_approx, u_vec, dx, dy, n)

    # Define custom preconditioner as a LinearOperator
    m = F.shape[0] + S.shape[0]
    exact_schur = LinearOperator(shape=(m, m), matvec=exact_schur_op)

    print(f"\nPrinting error norms for solving Ax=b using GMRES with exact Schur complement as preconditioner:")
    u_approx, _ = fgmres(A, b_vec, M=exact_schur, tol=1e-8, maxiter=40, callback=print_true_res_norm(A, b_vec))  
    print_norms(u_approx, u_vec, dx, dy, n)
    
    mD = -1.0*D
    Gt_G = np.matmul(mD,G)  # Note: G^T = D
    Gt_F = np.matmul(mD,F)
    Gt_F_G = np.matmul(Gt_F,G) 
    A_inv = scipy.linalg.pinv(F)  
    Gt_G_inv = scipy.linalg.pinv(Gt_G)
    S_int = Gt_F_G @ Gt_G_inv
    S_inv = Gt_G_inv @ S_int

    # Define the combined matvec function encapsulating all schur complement operations.
    def approx_schur_op(v):
        Ainv_v = np.matmul(A_inv,  v[:F.shape[1]]) 
        rhs_interim = np.matmul(D, Ainv_v) +  v[F.shape[1]:]
        x_a = lstsq(Gt_G,rhs_interim)[0]  
        x_b = np.matmul(Gt_F_G,x_a)
        x_p = lstsq(Gt_G,x_b)[0]
        G_xp = np.matmul(G, x_p)
        Ainv_G_xp = np.matmul(A_inv, G_xp) # OR solve(F, G_Sinv)
        u_approx_schur = Ainv_v - Ainv_G_xp
        u_approx = np.concatenate((u_approx_schur, x_p))
        return u_approx    

    # Define custom preconditioner as a LinearOperator
    m = F.shape[0] + S.shape[0]
    approx_schur = LinearOperator(shape=(m, m), matvec=approx_schur_op)

    # Get preconditioned system A * M^-1
    # Method (i)
    # n_s = 2*n*n+2*n*n+n*n
    # M_inv = scipy.linalg.pinv(approx_schur@np.eye(n_s), rtol=1e-6)
    # preconditioned_A = A @ M_inv

    # # Method (ii)
    S_apprx = scipy.linalg.pinv(S_inv)
    top_block = np.hstack((F, G))
    zero_block = np.zeros((S_apprx.shape[0], F.shape[1]))  # A zero block of shape (rows_S, cols_F)
    bottom_block = np.hstack((zero_block, -S_apprx))
    block_matrix = np.vstack((top_block, bottom_block))  # The matrix (F G; 0 -S)
    pre_A = A @ np.linalg.pinv(block_matrix)

    # Compute S*S^-1
    S_Sinv = S @ scipy.linalg.pinv(S, rtol=1e-8)
    
    print(f"\nPrinting error norms for solving Ax=b using fGMRES with approx schur complement as preconditioner:")
    # u_approx, _ = gmres(A, b_vec, M=approx_schur, rtol=1e-12, maxiter=20, callback=print_true_res_norm(A,b_vec), callback_type='x')
    u_approx, _ = fgmres(A, b_vec, x0 = x_initial, M=approx_schur, tol=1e-14, maxiter=20, callback=print_true_res_norm(A, b_vec))  
    print_norms(u_approx, u_vec, dx, dy, n)
    return A, pre_A, S_Sinv

def print_ev_from_sympy(A, pre_A):
    """
    Use SymPy to compute eigenvalues.

    Args:
        A: Unpreconditioned matrix
        pre_A:  Right-preconditioned matrix
    """

    M = Matrix(A)
    print("Eigenvalues of unpreconditioned system:")
    print(pretty(M.eigenvals(multiple=True, rational=True)))
    
    M = Matrix(pre_A)
    print("Eigenvalues of preconditioned system:")
    print(pretty(M.eigenvals(multiple=True, rational=True)))
          
def print_eigenvals(test_mat: np.ndarray, name: str):
    """
    This function outputs the converged eigenvalues for the input matrix test_mat.

    Args:
        test_mat: A numpy matrix 
        name: User-defined name for test_mat
    """
    m, n = test_mat.shape

    # Create PETSc matrix and set all elements to numpy matrix elements
    A_petsc = PETSc.Mat().create()
    A_petsc.setSizes([m, n])
    A_petsc.setFromOptions()
    A_petsc.setUp()

    for i in range(m):
        for j in range(n):
            A_petsc[i, j] = test_mat[i, j]
    A_petsc.assemble()

    # Create the eigenvalue solver and set the matrix
    eps = SLEPc.EPS().create()
    eps.setOperators(A_petsc)
    eps.setDimensions(nev=10) # change nev to request fewer eigenvalues
    eps.setTolerances(tol=1e-4)
    eps.setTolerances(max_it=40)
    eps.setFromOptions()

    print(f"{name}:")
    PETSc.Options().setValue("-eps_view_values", "")
    eps.solve()

    A_petsc.destroy()
    eps.destroy()

if __name__ == "__main__":

    # Solve the system
    A, pre_A, S_Sinv = main(n = 16)

    # Compute eigenvalues of A and preconditioned A
    print("\n")
    print_eigenvals(test_mat=pre_A, name="Preconditioned A")
    print_eigenvals(test_mat=A, name="Unpreconditioned A")
    print_eigenvals(test_mat=S_Sinv, name="S*Sinv")