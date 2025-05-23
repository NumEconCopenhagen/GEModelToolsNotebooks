# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

from steady_state import tauchen_trans_nb

@nb.njit
def calc_gamma(N,upsilon,z_grid,z_ergodic):
    """ calculate gamma """

    return z_grid**(upsilon*np.log(N))/np.sum(z_ergodic*z_grid**(1+upsilon*np.log(N)))
    
#@nb.njit(parallel=True)        
#def solve_hh_backwards(par,z_trans,ra,Z,UniformT,vbeg_a_plus,vbeg_a,a,c,muc,z,G,N,z_scale,ss=False):
#    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

@nb.njit       
def solve_hh_backwards(par,z_trans,ra,w,L,tau,chi,z_scale,vbeg_a_plus,vbeg_a,a,c,z):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # update idiosyncratic earnings 
    gamma = np.ones(par.Nz)
    
    if par.use_tauchen:

        sigma_psi_ = par.sigma_psi + par.upsilon*np.log(L)
        z_trans_ = tauchen_trans_nb(par.z_log_grid,0.0,par.rho_z,sigma_psi_)

        for i_fix in nb.prange(par.Nfix):
            z_trans[i_fix] = z_trans_

    else:
        
        z_trans[:] = par.z_trans_ss 

        if np.abs(par.upsilon) > 0:
            gamma [:] = calc_gamma(L,par.upsilon,par.z_grid,par.z_ergodic)

    # a. solve step
    for i_fix in range(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
        
            z[i_fix,i_z,:] = gamma[i_z]*z_scale*par.z_grid[i_z]
            income = (1-tau)*w*L*z[i_fix,i_z,:]+chi

            # i. EGM
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            
            # ii. interpolation to fixed grid
            m = (1+ra)*par.a_grid + income
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0) # enforce borrowing constraint
            c[i_fix,i_z] = m-a[i_fix,i_z]

        # b. expectation step
        v_a = (1+ra)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a