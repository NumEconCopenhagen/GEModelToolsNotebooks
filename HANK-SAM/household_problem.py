import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

###################
# solve backwards #
###################

@nb.njit
def solve_hh_backwards(par,z_trans,rh,wh,EVa_plus,EVa,a,c):
    """ solve backwards with EVa from previous iteration (here EVa_plus) """

    for i_fix in range(par.Nfix):

        # a. solution step
        for i_z in range(par.Nz):
      
            # i. EGM
            c_endo = (par.beta_grid[i_fix]*EVa_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            
            # ii. interpolation to fixed grid
            y = par.phi**par.u_grid[i_z]*wh*par.z_grid[i_z]
            m = (1+rh)*par.a_grid + y

            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        Va = (1+rh)*c[i_fix]**(-par.sigma)
        EVa[i_fix] = z_trans[i_fix]@Va

################
# fill_z_trans #
################

import math
inv_sqrt2 = 1/math.sqrt(2) # precompute

@nb.njit(fastmath=True)
def _norm_cdf(z):  
    """ raw normal cdf """

    return 0.5*math.erfc(-z*inv_sqrt2)

@nb.njit(fastmath=True)
def norm_cdf(z,mean,std):
    """ normal cdf with scaling """

    # a. check
    if std <= 0:
        if z > mean: return 1
        else: return 0

    # b. scale
    z_scaled = (z-mean)/std

    # c. return
    return _norm_cdf(z_scaled)

@nb.njit(fastmath=True)
def fill_z_trans(par,z_trans,EU,UE):
    """ transition matrix for z """
    
    # a. logaritn
    log_e_grid = np.log(par.e_grid)

    # b. unemployment transition
    u_trans = np.array([[1.0-EU,EU],[UE,1.0-UE]])

    # c. transition matrix
    for i_fix in nb.prange(par.Nfix):
        for i_e in nb.prange(par.Ne):
            for i_u in nb.prange(2):
                for i_e_plus in nb.prange(par.Ne):
                    for i_u_plus in nb.prange(2):

                        if i_e_plus == par.Ne-1:
                            L = 1.0
                        else:
                            midpoint = log_e_grid[i_e_plus] + (log_e_grid[i_e_plus+1]-log_e_grid[i_e_plus])/2
                            L = norm_cdf(midpoint,par.rho_e*log_e_grid[i_e],par.sigma_psi)

                        if i_e_plus == 0:
                            R = 0.0
                        else:
                            midpoint = log_e_grid[i_e_plus] - (log_e_grid[i_e_plus]-log_e_grid[i_e_plus-1])/2
                            R = norm_cdf(midpoint,par.rho_e*log_e_grid[i_e],par.sigma_psi)

                        i_z = i_u*par.Ne + i_e
                        i_z_plus = i_u_plus*par.Ne+i_e_plus
                        u_trans_now = u_trans[i_u,i_u_plus]
                        z_trans[i_fix,i_z,i_z_plus] = u_trans_now*(L-R)                            