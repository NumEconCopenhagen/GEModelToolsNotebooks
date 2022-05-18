import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,EVa_plus,EVa,a,c):
    """ solve backwards with EVa from previous iteration (here EVa_plus) """

    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in nb.prange(par.Nz):
        
            # i. EGM
            c_endo = (par.beta_grid[i_fix]*EVa_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            
            # ii. interpolation to fixed grid
            m = (1+r)*par.a_grid + w*par.z_grid[i_z]
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        Va = (1+r)*c[i_fix]**(-par.sigma)
        EVa[i_fix] = z_trans[i_fix]@Va

