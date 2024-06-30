import numpy as np
from preconditioner import MultiphaseBlockPreconditioner
import matplotlib.pyplot as plt
from utils import PI, weighted_L2, weighted_L1, max_norm
from utils import fill_sol_and_RHS_vecs
from scipy.sparse.linalg import gmres

if __name__ == "__main__":
    n = 32
    xi = 1.0
    dx = 1/n
    dy = 1/n

    block_prec = MultiphaseBlockPreconditioner(n,xi)
    L_n, D_n, XI_n, G_n = block_prec.get_block_matrices(is_ths=False)
    L_s, D_s, XI_s, G_s = block_prec.get_block_matrices(is_ths=True)

    # Build the big A matrix
    c = 1.0
    d = -1.0
    nu = 1.0
    etan = 1.0
    etas = 1.0
    A, S = block_prec.get_big_A_matrix(c=c, d_u=d)
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

    b_p_fcn = lambda y,x: 2*PI*np.cos(2*PI*x)*np.cos(2*PI*y)

    ########### For Thn = 0.25*np.sin(2*PI*x)*np.sin(2*PI*y) + 0.5 ############
    # RHS vector components for variable thn. 
    b_n_x_fcn = lambda y,x: (np.cos(2*PI*y)*np.sin(2*PI*x)*(4*c*nu-4*d*(8*etan*nu*PI*PI+xi)+2*nu*(c-16*d*etan*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)+d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)
    b_n_y_fcn = lambda y,x: (np.cos(2*PI*x)*np.sin(2*PI*y)*(4*c*nu-4*d*(8*etan*nu*PI*PI+xi)+2*nu*(c-16*d*etan*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)+d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)

    b_s_x_fcn = lambda y,x: (np.cos(2*PI*y)*np.sin(2*PI*x)*(-4*c*nu+4*d*(8*etas*nu*PI*PI+xi)+2*nu*(c-16*d*etas*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)-d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)
    b_s_y_fcn = lambda y,x: (np.cos(2*PI*x)*np.sin(2*PI*y)*(-4*c*nu+4*d*(8*etas*nu*PI*PI+xi)+2*nu*(c-16*d*etas*PI*PI)*np.sin(2*PI*x)*np.sin(2*PI*y)-d*xi*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)))/(8*nu)

    b_p_fcn = lambda y,x: PI*np.sin(4*PI*x)*np.sin(4*PI*y)

    u_vec, b_vec = fill_sol_and_RHS_vecs(n, u_n_x_fcn, u_n_y_fcn, u_s_x_fcn, u_s_y_fcn, p_fcn, 
                                    b_n_x_fcn, b_n_y_fcn, b_s_x_fcn, b_s_y_fcn, b_p_fcn)
    
    # print(np.linalg.cond(A))              # very high value so system is ill-conditioned
    u_approx, exitcode = gmres(A, b_vec,atol=1e-6) 
    # u_approx = np.linalg.solve(A,b_vec)   # Do not use, doesn't converge to the right solution.
    
    # Calculate error norms for solve
    L2_norm = weighted_L2(u_approx,u_vec,dx*dy)
    L1_norm = weighted_L1(u_approx,u_vec,dx*dy)
    Max_norm = max_norm(u_approx,u_vec)
    print(f"Printing error norms for solving Ax=b:")
    print(f"The L1_norm for n = {n} is {L1_norm}")
    print(f"The L2_norm for n = {n} is {L2_norm}")
    print(f"The max_norm for n = {n} is {Max_norm}")

