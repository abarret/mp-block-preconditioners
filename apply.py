import numpy as np
from preconditioner import MultiphaseBlockPreconditioner
import matplotlib.pyplot as plt
from utils import PI, weighted_L2, weighted_L1, max_norm
from utils import write_to_csv, check_individual_operators, apply_A_matrix


if __name__ == "__main__":
    write_to_csv = False
    check_laplacian_op = False
    check_divergence_op = False
    check_xi_op = False
    check_gradient_op = False
    n = 32
    xi = 1.0
    dx = 1/n
    dy = 1/n

    block_prec = MultiphaseBlockPreconditioner(n,xi)
    L_n, D_n, XI_n, G_n = block_prec.get_block_matrices(is_ths=False)
    L_s, D_s, XI_s, G_s = block_prec.get_block_matrices(is_ths=True)

    if write_to_csv:
        write_to_csv(L_n, D_n, XI_n, G_n)

    if check_laplacian_op or check_divergence_op or check_xi_op or check_gradient_op:
        check_individual_operators(n, xi, L_n, D_n, XI_n, G_n, check_laplacian_op, check_divergence_op, check_xi_op, check_gradient_op)

    # Apply operator
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

    # RHS vector components for constant thn. Thn = 0.75
    b_n_x_fcn = lambda y,x: (3*(2*c*nu-d*(16*etan*nu*PI*PI+xi))*np.cos(2*PI*y)*np.sin(2*PI*x))/(8*nu)
    b_n_y_fcn = lambda y,x: (3*(2*c*nu-d*(16*etan*nu*PI*PI+xi))*np.cos(2*PI*x)*np.sin(2*PI*y))/(8*nu)

    b_s_x_fcn = lambda y,x: ((-2*c*nu+16*d*etas*nu*PI*PI+3*d*xi)*np.cos(2*PI*y)*np.sin(2*PI*x))/(8*nu)
    b_s_y_fcn = lambda y,x: ((-2*c*nu+16*d*etas*nu*PI*PI+3*d*xi)*np.cos(2*PI*x)*np.sin(2*PI*y))/(8*nu)

    b_p_fcn = lambda y,x: 2*PI*np.cos(2*PI*x)*np.cos(2*PI*y)

    u_vec, b_vec = apply_A_matrix(n, u_n_x_fcn, u_n_y_fcn, u_s_x_fcn, u_s_y_fcn, p_fcn, 
                                    b_n_x_fcn, b_n_y_fcn, b_s_x_fcn, b_s_y_fcn, b_p_fcn)

    b_approx = np.zeros(5*n*n)
    b_approx = np.matmul(A,u_vec)
    
    # Calculate error norms
    L2_norm = weighted_L2(b_vec,b_approx,dx*dy)
    L1_norm = weighted_L1(b_vec,b_approx,dx*dy)
    max_norm = max_norm(b_vec,b_approx)
    print(f"Printing error norms for application of big A:")
    print(f"The L1_norm for n = {n} is {L1_norm}")
    print(f"The L2_norm for n = {n} is {L2_norm}")
    print(f"The max_norm for n = {n} is {max_norm}")



