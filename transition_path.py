import math
import numpy as np
import numba as nb

from GEModelTools import lag, simulate_hh_path

# local
from household_problem import solve_hh_path

###############
# 2. evaluate #
###############

@nb.njit
def evaluate_transition_path_distribution(par,sol,sim,ss,path,jac_hh,use_jac_hh=True):
    """ evaluate household path """

    # a. solve and simulate households
    path.A_hh[:] = np.repeat(np.nan,par.transition_T)
    path.C_hh[:] = np.repeat(np.nan,par.transition_T)

    if use_jac_hh:            
    
        path.A_hh[:] = ss.A_hh 
        path.A_hh[:] += jac_hh.A_r@(path.r-ss.r) 
        path.A_hh[:] += jac_hh.A_w@(path.w-ss.w) 

        path.C_hh[:] = ss.C_hh 
        path.C_hh[:] += jac_hh.C_r@(path.r-ss.r) 
        path.C_hh[:] += jac_hh.C_w@(path.w-ss.w) 
    
    else:
        
        solve_hh_path(par,sol,path)
        simulate_hh_path(par,sol,sim)
    
        for t in range(par.transition_T):

            # i. distribution                
            if t == 0: # steady state
                D_lag = sim.D
            else:
                D_lag = sim.path_D[t-1]

            # ii. aggregate
            path.A_hh[t] = np.sum(sol.path_a[t]*D_lag)
            path.C_hh[t] = np.sum(sol.path_c[t]*D_lag)
            
@nb.njit
def evaluate_transition_path(par,sol,sim,ss,path,jac_hh,use_jac_hh=True):
    """ evaluate transition path """

    K_lag = lag(ss.K,path.K[:-1])

    ####################
    # I. implied paths #
    ####################

    path.L[:] = 1.0
    path.rk[:] = par.alpha*path.Z*(K_lag/path.L)**(par.alpha-1.0)
    path.r[:] = path.rk[:] - par.delta
    path.w[:] = (1.0-par.alpha)*path.Z*(path.rk/(par.alpha*path.Z))**(par.alpha/(par.alpha-1.0))

    path.Y[:] = path.Z*K_lag**(par.alpha)*path.L**(1-par.alpha)
    path.C[:] = path.Y - (path.K-K_lag) - par.delta*K_lag

    path.A[:] = path.K

    #########################
    # II. household problem #
    #########################
    
    evaluate_transition_path_distribution(par,sol,sim,ss,path,jac_hh,use_jac_hh=use_jac_hh)

    ######################
    # III. check targets #
    ######################

    path.clearing_A[:] = path.A-path.A_hh
    path.clearing_C[:] = path.C-path.C_hh