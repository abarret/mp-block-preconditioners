import pandas as pd
import numpy as np
from preconditioner import thn, ths

PI = np.pi

def weighted_L2(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())

def weighted_L1(a,b,w):
    q = abs(a-b)
    w_q= w*q
    return w_q.sum()

def max_norm(a,b):
        return max(abs(a-b))

def write_to_csv(L_n, D_n, XI_n, G_n):
    # Write the data to CSV files
    L_mat = pd.DataFrame(L_n)
    L_mat.to_csv("L_matrix.csv", index=False, header=False)

    D_mat = pd.DataFrame(D_n)
    D_mat.to_csv("D_matrix.csv", index=False, header=False)

    xi_mat = pd.DataFrame(XI_n)
    xi_mat.to_csv("XI_matrix.csv", index=False, header=False)

    G_mat = pd.DataFrame(G_n)
    G_mat.to_csv("G_matrix.csv", index=False, header=False)

def check_individual_operators(n, xi, L_n, D_n, XI_n, G_n, check_laplacian_op, check_divergence_op, check_xi_op, check_gradient_op):
    """
    This function checks the convergence behavior for application of each block matrix.
    Note, the exact solutions here assume thn = 0.25*np.sin(2*PI*x)*np.sin(2*PI*y)+0.5
    """

    dx = 1/n
    dy = 1/n
    
    # Solution components
    u_n_x_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)  # x-component of Un
    u_n_y_fcn = lambda y,x: np.cos(2*PI*x)*np.sin(2*PI*y)  # y-component of Un
    p_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)      # Pressure
    u_n = np.zeros(2*n*n)   # 32-by-1 
    p_n = np.zeros(n*n)     # 16-by-1 
    
    if check_divergence_op:
        # Check that Divergence operator is 2nd order accurate
        b_p_fcn = lambda y,x: 2*PI*np.cos(2*PI*x)*np.cos(2*PI*y) + 1/2*PI*np.sin(4*PI*x)*np.sin(4*PI*y)
        b_p_exact = np.zeros(n*n)   # 16-by-1

        for row in range(n*n):
            if row < n:
                row_on_grid = 0
                col_on_grid = row
            else:
                row_on_grid = row//n
                col_on_grid = row%n

            #print((row_on_grid+0.5)*dy,col_on_grid*dx)
            u_n[row]=u_n_x_fcn(-(row_on_grid+0.5)*dy,col_on_grid*dx)       # x-components of velocity
            u_n[row+n*n] = u_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx) # y-components of velocity
            b_p_exact[row] = b_p_fcn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)  # Exact RHS vector

        b_p_approx = np.matmul(D_n,u_n)
        L2_norm = weighted_L2(b_p_exact,b_p_approx,dx*dy)
        L1_norm = weighted_L1(b_p_exact,b_p_approx,dx*dy)
        print(f"Printing error norms for application of D:")
        print(f"The L1_norm for n = {n} is {L1_norm}")
        print(f"The L2_norm for n = {n} is {L2_norm}\n\n")
    
    if check_gradient_op:
        # Check that the Gradient operator is 2nd order accurate
        b_n_x_fcn = lambda y,x: PI/2*np.sin(2*PI*x)*np.sin(2*PI*y)*np.cos(2*PI*x)*np.cos(2*PI*y)+PI*np.cos(2*PI*x)*np.cos(2*PI*y)
        b_n_y_fcn = lambda y,x: -PI/2*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)-PI*np.sin(2*PI*x)*np.sin(2*PI*y)
        b_n_exact = np.zeros(2*n*n)   # 32-by-1

        for row in range(n*n):
            if row < n:
                row_on_grid = 0
                col_on_grid = row
            else:
                row_on_grid = row//n
                col_on_grid = row%n

            p_n[row]=p_fcn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)                # components of pressure
            b_n_exact[row] = b_n_x_fcn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)        # x-component of exact RHS vector
            b_n_exact[row+n*n] = b_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)      # y-component of exact RHS vector

        b_n_approx = np.matmul(G_n,p_n)
        L2_norm = weighted_L2(b_n_exact,b_n_approx,dx*dy)
        L1_norm = weighted_L1(b_n_exact,b_n_approx,dx*dy)
        print(f"Printing error norms for application of G:")
        print(f"The L1_norm for n = {n} is {L1_norm}")
        print(f"The L2_norm for n = {n} is {L2_norm}\n\n")

    if check_xi_op:
        b_n_x_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)
        b_n_y_fcn = lambda y,x: np.cos(2*PI*x)*np.sin(2*PI*y)
        b_n_exact = np.zeros(2*n*n)   # 32-by-1

        for row in range(n*n):
            if row < n:
                row_on_grid = 0
                col_on_grid = row
            else:
                row_on_grid = row//n
                col_on_grid = row%n

            u_n[row]= u_n_x_fcn(-(row_on_grid+0.5)*dy,col_on_grid*dx)           # x-components of velocity
            u_n[row+n*n] = u_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)      # y-components of velocity
            b_n_exact[row] = xi*thn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)*ths(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)*b_n_x_fcn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)      # x-component of exact RHS vector
            b_n_exact[row+n*n] = xi*thn(-row_on_grid*dy,(col_on_grid+0.5)*dx)*ths(-row_on_grid*dy,(col_on_grid+0.5)*dx)*b_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)        # y-component of exact RHS vector

        b_n_approx = np.matmul(XI_n,u_n)
        L2_norm = weighted_L2(b_n_exact,b_n_approx,dx*dy)
        L1_norm = weighted_L1(b_n_exact,b_n_approx,dx*dy)
        print(f"Printing error norms for application of XI:")
        print(f"The L1_norm for n = {n} is {L1_norm}")
        print(f"The L2_norm for n = {n} is {L2_norm}\n\n")

    
    if check_laplacian_op:
        b_n_x_fcn = lambda y,x: -4*PI*PI*np.sin(2*PI*x)*np.sin(2*PI*x)*np.sin(2*PI*y)*np.cos(2*PI*y)-4*PI*PI*np.sin(2*PI*x)*np.cos(2*PI*y)
        b_n_y_fcn = lambda y,x: -4*PI*PI*np.sin(2*PI*x)*np.cos(2*PI*x)*np.sin(2*PI*y)*np.sin(2*PI*y)-4*PI*PI*np.cos(2*PI*x)*np.sin(2*PI*y)
        b_n_exact = np.zeros(2*n*n)   # 32-by-1

        for row in range(n*n):
            if row < n:
                row_on_grid = 0
                col_on_grid = row
            else:
                row_on_grid = row//n
                col_on_grid = row%n

            u_n[row]= u_n_x_fcn(-(row_on_grid+0.5)*dy,col_on_grid*dx)               # x-components of velocity
            u_n[row+n*n] = u_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)          # y-components of velocity
            b_n_exact[row] = b_n_x_fcn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)      # x-component of exact RHS vector
            b_n_exact[row+n*n] = b_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)    # y-component of exact RHS vector

        b_n_approx = np.matmul(L_n,u_n)
        L2_norm = weighted_L2(b_n_exact,b_n_approx,dx*dy)
        L1_norm = weighted_L1(b_n_exact,b_n_approx,dx*dy)
        print(f"Printing error norms for application of L:")
        print(f"The L1_norm for n = {n} is {L1_norm}")
        print(f"The L2_norm for n = {n} is {L2_norm}\n\n")
    
def fill_sol_and_RHS_vecs(n, u_n_x_fcn, u_n_y_fcn, u_s_x_fcn, u_s_y_fcn, p_fcn, b_n_x_fcn, b_n_y_fcn, b_s_x_fcn, b_s_y_fcn, b_p_fcn):
    """
    This function populates the solution and right-hand side vectors for a linear system of size n x n
    using the lambda functions passed in as arguments.
    """
    
    dx = 1/n
    dy = 1/n

    # Solution vector
    u_n = np.zeros(2*n*n)   # 32-by-1 
    u_s = np.zeros(2*n*n)   # 32-by-1 
    p = np.zeros(n*n)     # 16-by-1 

    # RHS vector
    b_n = np.zeros(2*n*n)   # 32-by-1
    b_s = np.zeros(2*n*n)   # 32-by-1
    b_p = np.zeros(n*n)   # 16-by-1

    for row in range(n*n):
            if row < n:
                row_on_grid = 0
                col_on_grid = row
            else:
                row_on_grid = row//n
                col_on_grid = row%n

            # Solution vector
            u_n[row]= u_n_x_fcn(-(row_on_grid+0.5)*dy,col_on_grid*dx)               # x-components of velocity
            u_n[row+n*n] = u_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)          # y-components of velocity
            
            u_s[row]= u_s_x_fcn(-(row_on_grid+0.5)*dy,col_on_grid*dx)               # x-components of velocity
            u_s[row+n*n] = u_s_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)          # y-components of velocity
            
            p[row]=p_fcn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)              # components of pressure

            # RHS vector
            b_n[row] = b_n_x_fcn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)       # x-component of exact RHS vector
            b_n[row+n*n] = b_n_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)     # y-component of exact RHS vector
            
            b_s[row] = b_s_x_fcn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)       # x-component of exact RHS vector
            b_s[row+n*n] = b_s_y_fcn(-row_on_grid*dy,(col_on_grid+0.5)*dx)     # y-component of exact RHS vector

            b_p[row] = b_p_fcn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)     # Exact RHS vector

    u_vec = np.concatenate((u_n,u_s,p))
    b_vec = np.concatenate((b_n,b_s,b_p))

    return u_vec, b_vec