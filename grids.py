import numpy as np
from numba import njit
from scipy.stats import norm

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def create_grids(model):
    """ create grids """

    # note: only fills out already allocated arrays

    # model specific: par.beta_grid
    # always:
    #   par.endo1_grid
    #   par.z_grid_ss (par.Nz,) # grid values
    #   par.z_trans_ss (par.Nz,par.Nz) # transition matrix
    #   par.z_ergodic_ss (par.Nz,) # ergodic distribution
    #   par.z_grid_path (par.transition,par.Nz) # grid along transition path
    #   par.z_transition_path (par.transition,par.Nz) # transition matrix along transition path

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
