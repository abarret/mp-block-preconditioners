import numpy as np
import pandas as pd
import scipy
from preconditioner import MultiphaseBlockPreconditioner
import matplotlib.pyplot as plt
from utils import PI, weighted_L2, weighted_L1, max_norm
from utils import fill_sol_and_RHS_vecs, write_to_csv
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.linalg import lstsq, solve, svd

def main():
    n = 32
    xi = 1.0
    mu = 1.0
    dx = 1/n
    dy = 1/n

    block_prec = MultiphaseBlockPreconditioner(n,xi, mu)
    L_n, D_n, XI_n, G_n,_ = block_prec.get_block_matrices(is_ths=False)
    L_s, D_s, XI_s, G_s,_ = block_prec.get_block_matrices(is_ths=True)

    # Build the big A matrix
    c = 1.0
    d = -1.0
    nu = 1.0
    etan = 1.0
    etas = 1.0
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
    
    # cond = (np.linalg.cond(A))              # very high value so system is ill-conditioned
    # u_approx, exitCode = gmres(A, b_vec)  
    # least squares will take into account the null space. Tries to minimize A*u_approx-b.
    u_approx, res, rnk, s = lstsq(A, b_vec) 
                           
    # Calculate error norms for solve
    L2_norm = weighted_L2(u_approx,u_vec,dx*dy)
    L1_norm = weighted_L1(u_approx,u_vec,dx*dy)
    Max_norm = max_norm(u_approx,u_vec)
    print(f"\nPrinting error norms for solving Ax=b without preconditioner:")
    print(f"The L1_norm for n = {n} is {L1_norm}")
    print(f"The L2_norm for n = {n} is {L2_norm}")
    print(f"The max_norm for n = {n} is {Max_norm}")
    
    # Scond = np.linalg.eigvals(S)              # very high value so system is ill-conditioned
    # print(f"Eigenvalues of S: {np.linalg.eig(S)[0]}")
    # print(f"Condition no. of A is {np.linalg.cond(A_block)}")

   # Define the combined matvec function encapsulating all operations.
    def combined_matvec(v):
        A_inv = scipy.linalg.inv(A_block)  # top left block
        Ainv_v = solve(A_block,  v[:A_block.shape[1]])  # Dense solve on top left block. Note: lstsq, solve and matmul(A_inv,b_u) give same result
        rhs_interim = np.matmul(D, Ainv_v) +  v[A_block.shape[1]:]
        x_p = -1.0*gmres(S, rhs_interim)[0]  # Using GMRES here gives 2nd order convergence, but lstsq doesn't and matmul(S_inv, rhs) doesn't.
        G_xp = np.matmul(G, x_p)
        Ainv_G_xp = np.matmul(A_inv, G_xp) # OR solve(A_block, G_Sinv)
        u_approx_schur = Ainv_v - Ainv_G_xp
        u_approx = np.concatenate((u_approx_schur, x_p))
        return u_approx

    u_approx = combined_matvec(b_vec)
    L2_norm = weighted_L2(u_approx,u_vec,dx*dy)
    L1_norm = weighted_L1(u_approx,u_vec,dx*dy)
    Max_norm = max_norm(u_approx,u_vec)
    print(f"\nPrinting error norms for solving Ax=b using schur complement:")
    print(f"The L1_norm for n = {n} is {L1_norm}")
    print(f"The L2_norm for n = {n} is {L2_norm}")
    print(f"The max_norm for n = {n} is {Max_norm}")

    # Define custom preconditioner as a LinearOperator
    m = A_block.shape[0] + S.shape[0]
    combined_op = LinearOperator(shape=(m, m), matvec=combined_matvec)

    def print_iteration(residual_norm):
        print(f"GMRES Iteration: Residual norm = {residual_norm}")

    print(f"\nPrinting error norms for solving Ax=b using GMRES with preconditioner:")
    # Pass the combined operator as a preconditioner `M` to GMRES and use the callback.
    u_approx, exitCode = gmres(A, b_vec, M=combined_op, rtol=1e-5, maxiter=5, callback=print_iteration, callback_type='legacy')  
    
    L2_norm = weighted_L2(u_approx,u_vec,dx*dy)
    L1_norm = weighted_L1(u_approx,u_vec,dx*dy)
    Max_norm = max_norm(u_approx,u_vec)
    print(f"The L1_norm for n = {n} is {L1_norm}")
    print(f"The L2_norm for n = {n} is {L2_norm}")
    print(f"The max_norm for n = {n} is {Max_norm}")

if __name__ == "__main__":
    main()