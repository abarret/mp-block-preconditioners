import numpy as np
import pandas as pd

n = 4       # Size of N-by-N staggered grid
dx = 1/n
dy = 1/n
xi = 1.0    # Drag coefficient

                              # Sizes for a single phase. Later, we can generate the matrices for the second phase.
L = np.zeros((2*n*n,2*n*n))   # 32-by-32 
D = np.zeros((n*n,2*n*n))     # 16-by-32
XI = np.zeros((2*n*n,2*n*n))  # 32-by-32
G = np.zeros((2*n*n,n*n))     # 32-by-16

# Volume fractions 
thn = lambda y, x: np.cos(x)*np.sin(y)
ths = lambda y, x: 1 - thn(y,x)

def get_thn_vals(n, row_on_grid, col_on_grid):
    # Define the 6 cell-centered thn values we need that surround u_(i+0.5,j)
    if col_on_grid==0 or col_on_grid==n:
        if row_on_grid==0:
            thn_i_jp1 = thn((n-0.5)*dy,(n-0.5)*dx)
            thn_ip1_jp1 = thn((n-0.5)*dy,0.5*dx)
        else:
            thn_i_jp1 = thn((row_on_grid-0.5)*dy,(n-0.5)*dx)
            thn_ip1_jp1 = thn((row_on_grid-0.5)*dy,0.5*dx)
        if row_on_grid==(n-1):
            thn_i_jm1 = thn(0.5*dy,(n-0.5)*dx)
            thn_ip1_jm1 = thn(0.5*dy,0.5*dx)
        else:
            thn_i_jm1 = thn((row_on_grid+1+0.5)*dy,(n-0.5)*dx)
            thn_ip1_jm1 = thn((row_on_grid+1+0.5)*dy,0.5*dx)
        
        thn_i_j = thn((row_on_grid+0.5)*dy,(n-0.5)*dx)
        thn_ip1_j = thn((row_on_grid+0.5)*dy,0.5*dx)
    elif row_on_grid==0:
        thn_i_jp1 = thn((n-0.5)*dy,(col_on_grid-0.5)*dx)
        thn_ip1_jp1 = thn((n-0.5)*dy,(col_on_grid+0.5)*dx)

        thn_i_j = thn((row_on_grid+0.5)*dy,(col_on_grid-0.5)*dx)
        thn_i_jm1 = thn((row_on_grid+1+0.5)*dy,(col_on_grid-0.5)*dx)

        thn_ip1_j = thn((row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)
        thn_ip1_jm1 = thn((row_on_grid+1+0.5)*dy,(col_on_grid+0.5)*dx)
    elif row_on_grid==(n-1):
        thn_i_jm1 = thn(0.5*dy,(col_on_grid-0.5)*dx)
        thn_ip1_jm1 = thn(0.5*dy,(col_on_grid+0.5)*dx)

        thn_i_jp1 = thn((row_on_grid-0.5)*dy,(col_on_grid-0.5)*dx)
        thn_i_j = thn((row_on_grid+0.5)*dy,(col_on_grid-0.5)*dx)

        thn_ip1_jp1 = thn((row_on_grid-0.5)*dy,(col_on_grid+0.5)*dx)
        thn_ip1_j = thn((row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)
    else:
        thn_i_jp1 = thn((row_on_grid-0.5)*dy,(col_on_grid-0.5)*dx)
        thn_i_j = thn((row_on_grid+0.5)*dy,(col_on_grid-0.5)*dx)
        thn_i_jm1 = thn((row_on_grid+1+0.5)*dy,(col_on_grid-0.5)*dx)

        thn_ip1_jp1 = thn((row_on_grid-0.5)*dy,(col_on_grid+0.5)*dx)
        thn_ip1_j = thn((row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)
        thn_ip1_jm1 = thn((row_on_grid+1+0.5)*dy,(col_on_grid+0.5)*dx)

    return thn_i_j, thn_ip1_j, thn_i_jp1, thn_ip1_jp1, thn_i_jm1, thn_ip1_jm1

# The first n*n rows of L correspond to the u unknowns
# Row of L corresponding to unknown u(i,j)
for row in range(n*n):
    if row < n:
        row_on_grid = 0
        col_on_grid = row
    else:
        row_on_grid = row//n
        col_on_grid = row%n
    
    # print(f"Row on grid = {row_on_grid}, Col on grid = {col_on_grid}")

    thn_i_j, thn_ip1_j, thn_i_jp1, thn_ip1_jp1, thn_i_jm1, thn_ip1_jm1 = get_thn_vals(n,row_on_grid,col_on_grid)
    
    thn_iph_jph = 0.25*(thn_i_j+thn_i_jp1+thn_ip1_jp1+thn_ip1_j)
    thn_iph_jmh = 0.25*(thn_i_j+thn_ip1_j+thn_i_jm1+thn_ip1_jm1)
    thn_iph_j = 0.5*(thn_i_j+thn_ip1_j)
    thn_i_jph = 0.5*(thn_i_j+thn_i_jp1)
    thn_i_jmh = 0.5*(thn_i_j+thn_i_jm1)

    thn_im1_j, thn_i_j, _, _, _, _ = get_thn_vals(n,row_on_grid,col_on_grid-1)
    thn_imh_j = 0.5*(thn_im1_j+thn_i_j)
    # print(f'Thn(i+0.5, j+0.5) is {thn_iph_jph}). Thn(i+0.5,j-0.5) is {thn_iph_jmh}')
    
    XI[row][row] = xi*thn((row_on_grid+0.5)*dy,col_on_grid*dx)*ths((row_on_grid+0.5)*dy,col_on_grid*dx)
    XI[row+n*n][row+n*n] = xi*thn(row_on_grid*dy,(col_on_grid+0.5)*dx)*ths(row_on_grid*dy,(col_on_grid+0.5)*dx)
    
    G[row][row] = thn_imh_j  # Coeff of P(i,j)

    # Coeff of P(i-1,j)
    if col_on_grid==0:
        G[row][row_on_grid*n+(n-1)] = -thn_imh_j
    else:
        G[row][row-1] = -thn_imh_j

    L[row][row] = 1/(dx*dx)*(-thn_ip1_j-thn_i_j)+1/(dy*dy)*(-thn_iph_jph-thn_iph_jmh)   # Coeff of u_(i+0.5,j)
    D[row][row] = 1/dx*thn_iph_j # Coeff of u_(i+0.5,j)

    # No ghost cell handling needed for this:
    L[row][n*n+(n*row_on_grid)+col_on_grid] = 1/(dx*dy)*(-thn_ip1_j + thn_iph_jph)      # Coeff of v_(i+1,j+1/2)
    
    ################# Handles ghost cell values on the vertical edges ###################
    # Handle ghost cell values on the left edges before the left boundary
    if col_on_grid==0:
        L[row][(n-1)+row_on_grid*(n)] = 1/(dx*dx)*(thn_i_j)
        D[row][(n-1)+row_on_grid*(n)] = -1/dx*thn_imh_j
    else:
        L[row][row-1] += 1/(dx*dx)*(thn_i_j)                        # Coeff of u_(i-0.5,j)
        D[row][row-1] = -1/dx*thn_imh_j

    #Handle ghost cell values on the right edges after the right boundary
    if col_on_grid==(n-1):
        L[row][row_on_grid*(n)] = thn_ip1_j/(dx*dx)
    else:
        L[row][row+1] = thn_ip1_j/(dx*dx)                           # Coeff of u_(i+3/2,j)

    # Handle ghost cell values on the vertical edges above the top boundary
    if row_on_grid==0:
        L[row][(n-1)*(n)+col_on_grid] = 1/(dy*dy)*(thn_iph_jph)
    else:
        L[row][row-n] = 1/(dy*dy)*(thn_iph_jph)                     # Coeff of u_(i+1/2,j+1)

    # Handle ghost cell values on the vertical edges below the bottom boundary
    if row_on_grid==(n-1):
        L[row][col_on_grid] = 1/(dy*dy)*(thn_iph_jmh)
    else:
        L[row][row+n] = 1/(dy*dy)*(thn_iph_jmh)                     # Coeff of u_(i+1/2,j-1)

    ################## handle the ghost cell values on the horizontal edges #################
    # Handle ghost cell values on horizontal edges before the left boundary
    if col_on_grid==0:
        L[row][n*n+((row_on_grid+1)*n)-1] = 1/(dy*dx)*(thn_i_j-thn_iph_jph)                         # Coeff of v_(i,j+1/2)
        D[row][n*n+((row_on_grid+1)*n)-1] = 1/dy*(thn_i_jph)                                        # Coeff of v_(i,j+1/2)

        if row_on_grid==(n-1):
            L[row][n*n+n-1] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)                                       # Coeff of v_(i,j-1/2)
            D[row][n*n+n-1] = -1/dy*(thn_i_jmh)                                                     # Coeff of v_(i,j-1/2)
        else:
            L[row][n*n+((row_on_grid+2)*n)-1] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)                     # Coeff of v_(i,j-1/2)
            D[row][n*n+((row_on_grid+2)*n)-1] = -1/dy*(thn_i_jmh)                                   # Coeff of v_(i,j-1/2)

    else:  
        L[row][n*n+(n*row_on_grid)+(col_on_grid-1)] = 1/(dy*dx)*(thn_i_j-thn_iph_jph)               # Coeff of v_(i,j+1/2)
        D[row][n*n+(n*row_on_grid)+(col_on_grid-1)] = 1/dy*(thn_i_jph)                              # Coeff of v_(i,j+1/2)

        if row_on_grid==(n-1):
            L[row][n*n+(col_on_grid-1)] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)                           # Coeff of v_(i,j-1/2)
            D[row][n*n+(col_on_grid-1)] = -1/dy*(thn_i_jmh)                 

        else:
            L[row][n*n+(n*(row_on_grid+1))+(col_on_grid-1)] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)       # Coeff of v_(i,j-1/2)
            D[row][n*n+(n*(row_on_grid+1))+(col_on_grid-1)] = -1/dy*(thn_i_jmh)            

    # Handle ghost cell values on horizontal edges below the bottom boundary
    if row_on_grid==(n-1):
        L[row][n*n+col_on_grid] =  1/(dx*dy)*(thn_ip1_j - thn_iph_jmh)                              # Coeff of v_(i+1,j-1/2)
    else:   
        L[row][n*n+(n*(row_on_grid+1))+(col_on_grid)] =  1/(dx*dy)*(thn_ip1_j - thn_iph_jmh)        # Coeff of v_(i+1,j-1/2)

# The next n*n rows of L correspond to the v unknowns 
for row in range(n*n):
    if row < n:
        row_on_grid = 0
        col_on_grid = row
    else:
        row_on_grid = row//n
        col_on_grid = row%n

    nrow = row+n*n

    thn_i_j, thn_ip1_j, thn_i_jp1, thn_ip1_jp1, _, _ = get_thn_vals(n,row_on_grid,col_on_grid+1)
    thn_im1_j, _, thn_im1_jp1, _, _, _ = get_thn_vals(n,row_on_grid,col_on_grid)
    
    thn_imh_jph = 0.25*(thn_im1_j+thn_im1_jp1+thn_i_j+thn_i_jp1)
    thn_iph_jph = 0.25*(thn_i_j+thn_ip1_j+thn_i_jp1+thn_ip1_jp1)

    thn_i_jph = 0.5*(thn_i_j+thn_i_jp1)

    G[nrow][row] = thn_i_jph   # Coeff of P(i,j)

     # Coeff of P(i,j-1)
    if row_on_grid==0:
         G[nrow][n*(n-1)+col_on_grid] = -thn_i_jph
    else:
        G[nrow][row-n] = -thn_i_jph

    # Coeff of v_(i,j+0.5)
    L[nrow][nrow] = -1/(dy*dy)*(thn_i_jp1+thn_i_j)-1/(dx*dx)*(thn_iph_jph+thn_imh_jph)                                          
    
    # Coeff of v_(i-1,j+0.5)
    if col_on_grid==0:                                   
        L[nrow][nrow+(n-1)] = 1/(dx*dx)*thn_imh_jph     
    else:
        L[nrow][nrow-1] = 1/(dx*dx)*thn_imh_jph 

    # Coeff of v_(i+1,j+0.5)
    if col_on_grid==(n-1):
        L[nrow][nrow-(n-1)] = 1/(dx*dx)*thn_iph_jph      
    else:
        L[nrow][nrow+1] = 1/(dx*dx)*thn_iph_jph

    # Coeff of v_(i,j+3/2)
    if row_on_grid==0:
        L[nrow][nrow+n*(n-1)] = 1/(dy*dy)*thn_i_jp1
    else:
        L[nrow][nrow-n] = 1/(dy*dy)*thn_i_jp1

    # Coeff of v_(i,j-1/2)
    if row_on_grid==(n-1):
        L[nrow][nrow-n*(n-1)] = 1/(dy*dy)*thn_i_j
    else:
        L[nrow][nrow+n] = 1/(dy*dy)*thn_i_j   

    # Coeff of u_(i-1/2,j)
    # nrow-n*n is the same as n*row_on_grid+col_on_grid
    L[nrow][nrow-n*n]= 1/(dx*dy)*(thn_imh_jph-thn_i_j)

    # Coeff of u_(i+1/2,j)
    if col_on_grid==(n-1):
        # n*row_on_grid is the same as nrow-(n*n)-(n-1)
        L[nrow][n*row_on_grid] = 1/(dy*dx)*(thn_i_j-thn_iph_jph)
    else:
        L[nrow][nrow-(n*n)+1]= 1/(dy*dx)*(thn_i_j-thn_iph_jph)

    # Coeff of u_(i-1/2,j+1)
    if row_on_grid==0:
        L[nrow][n*(n-1)+col_on_grid] = 1/(dy*dx)*(thn_i_jp1-thn_imh_jph)
    else:
        L[nrow][n*(row_on_grid-1)+col_on_grid] = 1/(dy*dx)*(thn_i_jp1-thn_imh_jph)

    # Coeff of u_(i+1/2,j+1)
    if row_on_grid==0:
        if col_on_grid==(n-1):
            L[nrow][n*(n-1)] = 1/(dy*dx)*(thn_iph_jph-thn_i_jp1)
        else:
            L[nrow][n*(n-1)+col_on_grid+1] =  1/(dy*dx)*(thn_iph_jph-thn_i_jp1)
    else:
        if col_on_grid==(n-1):
            L[nrow][n*(row_on_grid-1)] =  1/(dy*dx)*(thn_iph_jph-thn_i_jp1)
        else:
            L[nrow][n*(row_on_grid-1)+col_on_grid+1] =  1/(dy*dx)*(thn_iph_jph-thn_i_jp1)


    # Write the data to CSV files
    L_mat = pd.DataFrame(L)
    L_mat.to_csv("L_matrix.csv", index=False, header=False)

    D_mat = pd.DataFrame(D)
    D_mat.to_csv("D_matrix.csv", index=False, header=False)

    xi_mat = pd.DataFrame(XI)
    xi_mat.to_csv("xi_matrix.csv", index=False, header=False)

    G_mat = pd.DataFrame(G)
    G_mat.to_csv("G_matrix.csv", index=False, header=False)