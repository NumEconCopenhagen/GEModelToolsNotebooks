import numpy as np
from numba import njit
from scipy.stats import norm

# create grids

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def create_grids(model):
    """ create grids """

    # note: only fills out already allocated arrays

    par = model.par
    ss = model.ss

    # a. beta
    par.beta_grid[:] = np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nbeta)

    # b. a
    par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
    
    # c. z - steady state
    par.z_grid_ss[:],par.z_trans_ss[:,:],par.z_ergodic_ss[:],_,_ = log_rouwenhorst(par.rho_z,par.sigma_z,par.Nz)

    # d. z - path
    for t in range(par.transition_T):
        par.z_grid_path[t,:] = par.z_grid_ss
        par.z_trans_path[t,:,:] = par.z_trans_ss
