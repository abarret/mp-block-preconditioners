import numpy as np
from typing import Tuple
import scipy.linalg

PI = np.pi

# Volume fractions 
def thn(y,x)->float:
    thn = 0.75  # 0.25*np.sin(2*PI*x)*np.sin(2*PI*y)+0.5
    return thn

def ths(y,x)->float:
    ths = 1 - thn(y,x)
    return ths

class MultiphaseBlockPreconditioner:
    def __init__(self, n, xi):
        self.n = n
        self.dx = 1/n
        self.dy = 1/n
        self.xi = xi
        
    def weightedL2(self,a,b,w):
            q = a-b
            return np.sqrt((w*q*q).sum())

    def weightedL1(self,a,b,w):
            q = abs(a-b)
            w_q= w*q
            return w_q.sum()

    def get_thn_vals(self, n, row_on_grid, col_on_grid, is_ths: bool=False)->Tuple:
        dx = self.dx
        dy = self.dy
        
        # Define the 6 cell-centered thn values we need that surround u_(i+0.5,j)
        if col_on_grid==0 or col_on_grid==n:
            if row_on_grid==0:
                thn_i_jp1 = thn(-(n-0.5)*dy,(n-0.5)*dx)
                thn_ip1_jp1 = thn(-(n-0.5)*dy,0.5*dx)
            else:
                thn_i_jp1 = thn(-(row_on_grid-0.5)*dy,(n-0.5)*dx)
                thn_ip1_jp1 = thn(-(row_on_grid-0.5)*dy,0.5*dx)
            if row_on_grid==(n-1):
                thn_i_jm1 = thn(-0.5*dy,(n-0.5)*dx)
                thn_ip1_jm1 = thn(-0.5*dy,0.5*dx)
            else:
                thn_i_jm1 = thn(-(row_on_grid+1+0.5)*dy,(n-0.5)*dx)
                thn_ip1_jm1 = thn(-(row_on_grid+1+0.5)*dy,0.5*dx)
            
            thn_i_j = thn(-(row_on_grid+0.5)*dy,(n-0.5)*dx)
            thn_ip1_j = thn(-(row_on_grid+0.5)*dy,0.5*dx)
        elif row_on_grid==0:
            thn_i_jp1 = thn(-(n-0.5)*dy,(col_on_grid-0.5)*dx)
            thn_ip1_jp1 = thn(-(n-0.5)*dy,(col_on_grid+0.5)*dx)

            thn_i_j = thn(-(row_on_grid+0.5)*dy,(col_on_grid-0.5)*dx)
            thn_i_jm1 = thn(-(row_on_grid+1+0.5)*dy,(col_on_grid-0.5)*dx)

            thn_ip1_j = thn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)
            thn_ip1_jm1 = thn(-(row_on_grid+1+0.5)*dy,(col_on_grid+0.5)*dx)
        elif row_on_grid==(n-1):
            thn_i_jm1 = thn(-0.5*dy,(col_on_grid-0.5)*dx)
            thn_ip1_jm1 = thn(-0.5*dy,(col_on_grid+0.5)*dx)

            thn_i_jp1 = thn(-(row_on_grid-0.5)*dy,(col_on_grid-0.5)*dx)
            thn_i_j = thn(-(row_on_grid+0.5)*dy,(col_on_grid-0.5)*dx)

            thn_ip1_jp1 = thn(-(row_on_grid-0.5)*dy,(col_on_grid+0.5)*dx)
            thn_ip1_j = thn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)
        else:
            thn_i_jp1 = thn(-(row_on_grid-0.5)*dy,(col_on_grid-0.5)*dx)
            thn_i_j = thn(-(row_on_grid+0.5)*dy,(col_on_grid-0.5)*dx)
            thn_i_jm1 = thn(-(row_on_grid+1+0.5)*dy,(col_on_grid-0.5)*dx)

            thn_ip1_jp1 = thn(-(row_on_grid-0.5)*dy,(col_on_grid+0.5)*dx)
            thn_ip1_j = thn(-(row_on_grid+0.5)*dy,(col_on_grid+0.5)*dx)
            thn_ip1_jm1 = thn(-(row_on_grid+1+0.5)*dy,(col_on_grid+0.5)*dx)

        if is_ths:
            ths_i_j = 1-thn_i_j
            ths_ip1_j = 1-thn_ip1_j
            ths_i_jp1 = 1- thn_i_jp1
            ths_ip1_jp1 = 1-thn_ip1_jp1
            ths_i_jm1 = 1-thn_i_jm1
            ths_ip1_jm1 = 1-thn_ip1_jm1

            return ths_i_j, ths_ip1_j, ths_i_jp1, ths_ip1_jp1, ths_i_jm1, ths_ip1_jm1
        
        else:
            return thn_i_j, thn_ip1_j, thn_i_jp1, thn_ip1_jp1, thn_i_jm1, thn_ip1_jm1

    def get_block_matrices(self, is_ths: bool=False):
        dx = self.dx
        dy = self.dy
        n = self.n
        xi = self.xi
        # Sizes for a single phase. 
        L = np.zeros((2*n*n,2*n*n))   # 32-by-32 
        D = np.zeros((n*n,2*n*n))     # 16-by-32
        XI = np.zeros((2*n*n,2*n*n))  # 32-by-32
        G = np.zeros((2*n*n,n*n))     # 32-by-16

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

            thn_i_j, thn_ip1_j, thn_i_jp1, thn_ip1_jp1, thn_i_jm1, thn_ip1_jm1 = self.get_thn_vals(n,row_on_grid,col_on_grid, is_ths)
            
            thn_iph_jph = 0.25*(thn_i_j+thn_i_jp1+thn_ip1_jp1+thn_ip1_j)
            thn_iph_jmh = 0.25*(thn_i_j+thn_ip1_j+thn_i_jm1+thn_ip1_jm1)
            thn_iph_j = 0.5*(thn_i_j+thn_ip1_j)
            thn_i_jph = 0.5*(thn_i_j+thn_i_jp1)
            thn_i_jmh = 0.5*(thn_i_j+thn_i_jm1)

            thn_im1_j, thn_i_j, _, _, _, _ = self.get_thn_vals(n,row_on_grid,col_on_grid-1, is_ths)
            thn_imh_j = 0.5*(thn_im1_j+thn_i_j)
            thn_ip1_jph = 0.5*(thn_ip1_j+thn_ip1_jp1)
            # print(f'Thn(i+0.5, j+0.5) is {thn_iph_jph}. Thn(i+0.5,j-0.5) is {thn_iph_jmh}')

            # Note: u[0][0] is the first u(i+1/2,j) value.
            XI[row][row] = xi*thn_iph_j*(1-thn_iph_j)
            XI[row+n*n][row+n*n] = xi*thn_ip1_jph*(1-thn_ip1_jph)

            L[row][row] = 1/(dx*dx)*(-thn_ip1_j-thn_i_j)+1/(dy*dy)*(-thn_iph_jph-thn_iph_jmh)   # Coeff of u_(i+0.5,j)

            # No ghost cell handling needed for this:
            L[row][n*n+(n*row_on_grid)+col_on_grid] = 1/(dx*dy)*(-thn_ip1_j + thn_iph_jph)      # Coeff of v_(i+1,j+1/2)
            
            ################# Handles ghost cell values on the vertical edges ###################
            # Handle ghost cell values on the left edges before the left boundary
            if col_on_grid==0:
                L[row][(n-1)+row_on_grid*(n)] = 1/(dx*dx)*(thn_i_j)
            else:
                L[row][row-1] += 1/(dx*dx)*(thn_i_j)                        # Coeff of u_(i-0.5,j)

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

                if row_on_grid==(n-1):
                    L[row][n*n+n-1] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)                                       # Coeff of v_(i,j-1/2)
                else:
                    L[row][n*n+((row_on_grid+2)*n)-1] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)                     # Coeff of v_(i,j-1/2)

            else:  
                L[row][n*n+(n*row_on_grid)+(col_on_grid-1)] = 1/(dy*dx)*(thn_i_j-thn_iph_jph)               # Coeff of v_(i,j+1/2)

                if row_on_grid==(n-1):
                    L[row][n*n+(col_on_grid-1)] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)                           # Coeff of v_(i,j-1/2)
                else:
                    L[row][n*n+(n*(row_on_grid+1))+(col_on_grid-1)] = 1/(dy*dx)*(thn_iph_jmh-thn_i_j)       # Coeff of v_(i,j-1/2)

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

            thn_i_j, thn_ip1_j, thn_i_jp1, thn_ip1_jp1, thn_i_jm1, _ = self.get_thn_vals(n,row_on_grid,col_on_grid+1, is_ths)
            thn_im1_j, _, thn_im1_jp1, _, _, _ = self.get_thn_vals(n,row_on_grid,col_on_grid, is_ths)
            
            thn_imh_jph = 0.25*(thn_im1_j+thn_im1_jp1+thn_i_j+thn_i_jp1)
            thn_iph_jph = 0.25*(thn_i_j+thn_ip1_j+thn_i_jp1+thn_ip1_jp1)

            thn_i_jph = 0.5*(thn_i_j+thn_i_jp1)
            thn_i_jmh = 0.5*(thn_i_j+thn_i_jm1)
            thn_imh_j = 0.5*(thn_i_j+thn_im1_j)
            thn_iph_j = 0.5*(thn_i_j+thn_ip1_j)

            # Gradient in x-direction
            G[row][row] = (1/dx)*thn_imh_j  # Coeff of P(i,j)

            # Coeff of P(i-1,j)
            if col_on_grid==0:
                G[row][row_on_grid*n+(n-1)] = -(1/dx)*thn_imh_j  
            else:
                G[row][row-1] = -(1/dx)*thn_imh_j

            # Gradient in y-direction
            G[nrow][row] = -(1/dy)*thn_i_jph   # Coeff of P(i,j)

            # Coeff of P(i,j+1)
            if row_on_grid==0:
                G[nrow][n*(n-1)+col_on_grid] = (1/dy)*thn_i_jph
            else:
                G[nrow][row-n] = (1/dy)*thn_i_jph

            # Divergence Operator
            # Coeff of u_(i+0.5,j)
            if col_on_grid==(n-1):
                D[row][n*row_on_grid] = 1/dx*thn_iph_j   
            else:
                D[row][row+1] = 1/dx*thn_iph_j   

            # Coeff of u_(i-0.5,j)
            D[row][row] = -1/dx*thn_imh_j 

            # Coeff of v_(i,j+0.5)
            D[row][nrow] = 1/dy*thn_i_jph    

            # Coeff of v_(i,j-0.5)
            if row_on_grid==(n-1):
                D[row][nrow-n*(n-1)] = -1/dy*thn_i_jmh        
            else:       
                D[row][nrow+n] = -1/dy*thn_i_jmh    

            # Laplacian Operator
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
            # nrow-n*n == row. This is the same as n*row_on_grid+col_on_grid 
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

        return L, D, XI, G
    
    def get_big_A_matrix(self, w):
        dx = self.dx
        dy = self.dy
        n = self.n

        zero_block = np.zeros((n*n,n*n))     # 16-by-16

        L_n, D_n, XI_n, G_n = self.get_block_matrices(is_ths=False)
        L_s, D_s, XI_s, G_s = self.get_block_matrices(is_ths=True)
        L = scipy.linalg.block_diag(L_n, L_s)
        D = np.hstack((D_n,D_s))
        G = np.vstack((G_n,G_s))

        w_thn = np.zeros((2*n*n, 2*n*n))   # 32-by-32
        w_ths = np.zeros((2*n*n, 2*n*n))   # 32-by-32
        for row in range(n*n):
            if row < n:
                row_on_grid = 0
                col_on_grid = row
            else:
                row_on_grid = row//n
                col_on_grid = row%n

            w_thn[row][row] = w*thn(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)      
            w_thn[row+n*n][row+n*n] = w*thn(-row_on_grid*dy,(col_on_grid+0.5)*dx)

            w_ths[row][row] = w*ths(-(row_on_grid+0.5)*dy,(col_on_grid)*dx)      
            w_ths[row+n*n][row+n*n] = w*ths(-row_on_grid*dy,(col_on_grid+0.5)*dx)

        w_thn_xi  = w_thn + XI_n
        w_ths_xi  = w_ths + XI_s

        XI_first_row = np.hstack((w_thn_xi, -XI_n))
        XI_second_row = np.hstack((-XI_s, w_ths_xi))
        XI = np.vstack((XI_first_row, XI_second_row))
        A_block = XI + L

        A_G = np.hstack((A_block,G))
        D_zero = np.hstack((-D,zero_block))
        A = np.vstack((A_G,D_zero))

        inv_A = np.linalg.inv(A_block)
        # Compute the Schur Complement: S = -D*A^-1*G
        S_intermediate = np.matmul(-D,inv_A)
        S = np.matmul(S_intermediate,G)
        return A, S