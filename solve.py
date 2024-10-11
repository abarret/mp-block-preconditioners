import numpy as np
import pandas as pd
import scipy
from preconditioner import MultiphaseBlockPreconditioner
import matplotlib.pyplot as plt
from utils import PI
from utils import fill_sol_and_RHS_vecs, print_norms
from scipy.sparse.linalg import gmres, LinearOperator, eigsh
from scipy.linalg import lstsq, solve, svd
from pyamg.krylov import fgmres

def main():
    """
    This solves the linear system Ax=b arising from the discretized multiphase momentum and incompressibility equations.    
    """

    # Define the grid size and spacing
    n = 32
    dx = 1/n
    dy = 1/n

    # Define material specific parameters
    xi = 1.0
    mu = 1.0
    nu = 1.0
    etan = 1.0
    etas = 1.0

    # Define coefficients
    c = 1.0  # from c*Delta t
    d = -1.0

    # Build the big A matrix
    block_prec = MultiphaseBlockPreconditioner(n,xi,mu)
    A, S, A_block, D, G = block_prec.get_big_A_matrix(c=c, d_u=d)
    # plt.spy(A)
    # plt.show()

    # Solution components
    u_n_x_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)  # x-component of Un
    u_n_y_fcn = lambda y,x: np.cos(2*PI*x)*np.sin(2*PI*y)  # y-component of Un

    u_s_x_fcn = lambda y,x: -np.sin(2*PI*x)*np.cos(2*PI*y)  # x-component of Us
    u_s_y_fcn = lambda y,x: -np.cos(2*PI*x)*np.sin(2*PI*y)  # y-component of Us

    p_fcn = lambda y,x: 0.0     # Pressure

    ########## For Thn = 0.75 ###########
    # RHS vector components for constant thn. 
    b_n_x_fcn = lambda y,x: (3*(2*c*nu-d*(16*etan*nu*PI*PI+xi))*np.cos(2*PI*y)*np.sin(2*PI*x))/(8*nu)
    b_n_y_fcn = lambda y,x: (3*(2*c*nu-d*(16*etan*nu*PI*PI+xi))*np.cos(2*PI*x)*np.sin(2*PI*y))/(8*nu)

    b_s_x_fcn = lambda y,x: ((-2*c*nu+16*d*etas*nu*PI*PI+3*d*xi)*np.cos(2*PI*y)*np.sin(2*PI*x))/(8*nu)
    b_s_y_fcn = lambda y,x: ((-2*c*nu+16*d*etas*nu*PI*PI+3*d*xi)*np.cos(2*PI*x)*np.sin(2*PI*y))/(8*nu)

    b_p_fcn = lambda y,x: -2*PI*np.cos(2*PI*x)*np.cos(2*PI*y)

    # ########### For Thn = 0.25*np.sin(2*PI*x)*np.sin(2*PI*y) + 0.5 ############
    # # RHS vector components for variable thn. 
    # b_n_x_fcn = lambda y,x: (np.cos(2*PI*y)*np.sin(2*PI*x)*(4*c*nu-4*d*(8*etan*nu*PI*PI+xi)+2*nu*(c-16*d*etan*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)+d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)
    # b_n_y_fcn = lambda y,x: (np.cos(2*PI*x)*np.sin(2*PI*y)*(4*c*nu-4*d*(8*etan*nu*PI*PI+xi)+2*nu*(c-16*d*etan*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)+d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)

    # b_s_x_fcn = lambda y,x: (np.cos(2*PI*y)*np.sin(2*PI*x)*(-4*c*nu+4*d*(8*etas*nu*PI*PI+xi)+2*nu*(c-16*d*etas*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)-d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)
    # b_s_y_fcn = lambda y,x: (np.cos(2*PI*x)*np.sin(2*PI*y)*(-4*c*nu+4*d*(8*etas*nu*PI*PI+xi)+2*nu*(c-16*d*etas*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)-d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)

    # b_p_fcn = lambda y,x: -PI*np.sin(4*PI*x)*np.sin(4*PI*y)

    u_vec, b_vec, u, p, b_u, b_p = fill_sol_and_RHS_vecs(n, u_n_x_fcn, u_n_y_fcn, u_s_x_fcn, u_s_y_fcn, p_fcn, 
                                    b_n_x_fcn, b_n_y_fcn, b_s_x_fcn, b_s_y_fcn, b_p_fcn)
    
    def print_true_res_norm(A, b_vec):
        iteration = 0
        def callback(xk):
            nonlocal iteration
            iteration += 1
            residual = A @ xk - b_vec  # Compute the true residual
            residual_norm = np.linalg.norm(residual)
            print(f"GMRES Iteration {iteration}: True residual norm = {residual_norm}")
        return callback

    # least squares will take into account the null space. Tries to minimize A*u_approx-b.
    # u_approx, res, rnk, s = lstsq(A, b_vec) 
    print(f"\nPrinting error norms for solving Ax=b without preconditioner:")
    u_approx, _ = gmres(A, b_vec, rtol=1e-12, maxiter=20, callback=print_true_res_norm(A,b_vec), callback_type='x')                       
    print_norms(u_approx, u_vec, dx, dy, n)

    # Define the combined matvec function encapsulating all schur complement operations.
    def exact_schur_op(v):
        A_inv = scipy.linalg.inv(A_block)  # top left block
        Ainv_v = solve(A_block,  v[:A_block.shape[1]])  # Dense solve on top left block. Note: lstsq, solve and matmul(A_inv,b_u) give same result
        rhs_interim = np.matmul(D, Ainv_v) +  v[A_block.shape[1]:]
        x_p = -1.0*gmres(S, rhs_interim)[0]  # Using GMRES here gives 2nd order convergence, but lstsq doesn't and matmul(S_inv, rhs) doesn't.
        G_xp = np.matmul(G, x_p)
        Ainv_G_xp = np.matmul(A_inv, G_xp) # OR solve(A_block, G_Sinv)
        u_schur = Ainv_v - Ainv_G_xp
        u_approx = np.concatenate((u_schur, x_p))
        return u_approx

    u_approx = exact_schur_op(b_vec)
    print(f"\nPrinting error norms for solving Ax=b using schur complement:")
    print_norms(u_approx, u_vec, dx, dy, n)

    # Define custom preconditioner as a LinearOperator
    m = A_block.shape[0] + S.shape[0]
    combined_op = LinearOperator(shape=(m, m), matvec=exact_schur_op)

    print(f"\nPrinting error norms for solving Ax=b using GMRES with exact Schur complement as preconditioner:")
    u_approx, _ = gmres(A, b_vec, M=combined_op, rtol=1e-12, maxiter=20, callback=print_true_res_norm(A, b_vec), callback_type='x')  
    print_norms(u_approx, u_vec, dx, dy, n)

    
    # mD = -1.0*D
    # Gt_G = np.matmul(mD,G)  # Note: G^T = D
    # Gt_G_inv = scipy.linalg.inv(Gt_G)
    # Gt_F = np.matmul(mD,A_block)  # Note: F = A_block
    # Gt_F_G = np.matmul(Gt_F,G)
    # Fp = np.matmul(Gt_G_inv,Gt_F_G)
    # Fp_j = Fp[:,63]
    # G_Fp_j = np.matmul(G,Fp_j)

    # G_Fp = np.matmul(G,Fp)
    # F_G = np.matmul(A_block,G)
    # F_G_j = F_G[:,63]
    # print("\nDifference between matrices(See eqn 2.12 in Elman 2006):")
    # print_norms(G_Fp, F_G, dx, dy, n, show_max=False) # Difference between matrices(See eqn 2.12 in Elman 2006). 
    
    # print("\nDifference between column vectors:")
    # print_norms(G_Fp_j,F_G_j,1,1,n)  # Difference between column vectors
    
    # Define the combined matvec function encapsulating all schur complement operations.
    def approx_schur_op(v):
        mD = -1.0*D
        Gt_G = np.matmul(mD,G)  # Note: G^T = D
        Gt_F = np.matmul(mD,A_block)
        Gt_F_G = np.matmul(Gt_F,G) 
        A_inv = scipy.linalg.inv(A_block)  
        Ainv_v = np.matmul(A_inv,  v[:A_block.shape[1]])  # Dense solve on top left block. Note: lstsq makes it very slow
        rhs_interim = np.matmul(D, Ainv_v) +  v[A_block.shape[1]:]
        x_a = lstsq(Gt_G,rhs_interim)[0]
        x_b = np.matmul(Gt_F_G,x_a)
        x_p = lstsq(Gt_G,x_b)[0]
        G_xp = np.matmul(G, x_p)
        Ainv_G_xp = np.matmul(A_inv, G_xp) # OR solve(A_block, G_Sinv)
        u_approx_schur = Ainv_v - Ainv_G_xp
        u_approx = np.concatenate((u_approx_schur, x_p))
        return u_approx    

    # Define custom preconditioner as a LinearOperator
    m = A_block.shape[0] + S.shape[0]
    approx_schur = LinearOperator(shape=(m, m), matvec=approx_schur_op)

    # WIP
    # def preconditioned_A_op(b_vec):
    #     # Apply GMRES with the preconditioner to the vector 'v'
    #     result, _ = gmres(A, b_vec, M=approx_schur, rtol=1e-8, maxiter=20)
    #     return result

    # preconditioned_A = LinearOperator(shape=(m, m), matvec=preconditioned_A_op)

    # # Print eigenvalues of the unpreconditioned system
    # eigvals_unpreconditioned, _ = eigsh(A, k=5)  
    # print("Eigenvalues of the unpreconditioned system:")
    # print(eigvals_unpreconditioned)

    # # Print eigenvalues of the preconditioned system
    # eigvals_preconditioned, _ = eigsh(preconditioned_A, k=1)  
    # print("Eigenvalues of the preconditioned system:")
    # print(eigvals_preconditioned)

    print(f"\nPrinting error norms for solving Ax=b using GMRES with approx schur complement as preconditioner:")
    # u_approx, _ = fgmres(A, b_vec, M=approx_schur, tol=1e-14, maxiter=20)  
    u_approx, _ = gmres(A, b_vec, M=approx_schur, rtol=1e-12, maxiter=20, callback=print_true_res_norm(A,b_vec), callback_type='x')
    print_norms(u_approx, u_vec, dx, dy, n)

if __name__ == "__main__":
    main()