import numpy as np
import pandas as pd
from preconditioner import MultiphaseBlockPreconditioner, thn, ths

PI = np.pi

if __name__ == "__main__":
    write_to_csv = False
    check_divergence_op = True
    check_gradient_op = True
    check_xi_op = True
    check_laplacian_op = True
    n = 16
    xi = 1.0
    dx = 1/n
    dy = 1/n

    block_prec = MultiphaseBlockPreconditioner(n,xi)
    L_n, D_n, XI_n, G_n = block_prec.get_block_matrices(is_ths=False)
    L_s, D_s, XI_s, G_s = block_prec.get_block_matrices(is_ths=True)

    # Solution components
    u_n_x_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)  # x-component of Un
    u_n_y_fcn = lambda y,x: np.cos(2*PI*x)*np.sin(2*PI*y)  # y-component of Un
    p_fcn = lambda y,x: np.sin(2*PI*x)*np.cos(2*PI*y)      # Pressure
    u_n = np.zeros(2*n*n)   # 32-by-1 
    p_n = np.zeros(n*n)     # 16-by-1 
    
    if write_to_csv:
        # Write the data to CSV files
        L_mat = pd.DataFrame(L_n)
        L_mat.to_csv("L_matrix.csv", index=False, header=False)

        D_mat = pd.DataFrame(D_n)
        D_mat.to_csv("D_matrix.csv", index=False, header=False)

        xi_mat = pd.DataFrame(XI_n)
        xi_mat.to_csv("xi_matrix.csv", index=False, header=False)

        G_mat = pd.DataFrame(G_n)
        G_mat.to_csv("G_matrix.csv", index=False, header=False)

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
        L2_norm = block_prec.weightedL2(b_p_exact,b_p_approx,dx*dy)
        L1_norm = block_prec.weightedL1(b_p_exact,b_p_approx,dx*dy)
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
        L2_norm = block_prec.weightedL2(b_n_exact,b_n_approx,dx*dy)
        L1_norm = block_prec.weightedL1(b_n_exact,b_n_approx,dx*dy)
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
        L2_norm = block_prec.weightedL2(b_n_exact,b_n_approx,dx*dy)
        L1_norm = block_prec.weightedL1(b_n_exact,b_n_approx,dx*dy)
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
        L2_norm = block_prec.weightedL2(b_n_exact,b_n_approx,dx*dy)
        L1_norm = block_prec.weightedL1(b_n_exact,b_n_approx,dx*dy)
        print(f"Printing error norms for application of L:")
        print(f"The L1_norm for n = {n} is {L1_norm}")
        print(f"The L2_norm for n = {n} is {L2_norm}")