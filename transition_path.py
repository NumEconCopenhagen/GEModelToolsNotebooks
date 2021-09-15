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
def evaluate_transition_path(par,sol,sim,ss,path,jac_hh,threads=1,use_jac_hh=True):
    """ evaluate transition path """

    for thread in nb.prange(threads):

        # unpack
        Z = path.Z[:,thread]
        K = path.K[:,thread]
        L = path.L[:,thread]

        rk = path.rk[:,thread]
        r = path.r[:,thread]
        w = path.w[:,thread]
        
        Y = path.Y[:,thread]
        C = path.C[:,thread]
        A = path.A[:,thread]

        A_hh = path.A_hh[:,thread]
        C_hh = path.C_hh[:,thread]

        clearing_A = path.clearing_A[:,thread]
        clearing_C = path.clearing_C[:,thread]

        # lag
        K_lag = lag(ss.K,K[:-1])

        ####################
        # I. implied paths #
        ####################

        L[:] = 1.0
        rk[:] = par.alpha*Z*(K_lag/L)**(par.alpha-1.0)
        r[:] = rk-par.delta
        w[:] = (1.0-par.alpha)*Z*(rk/(par.alpha*Z))**(par.alpha/(par.alpha-1.0))

        Y[:] = Z*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag

        A[:] = K

        #########################
        # II. household problem #
        #########################
        
        # a. solve and simulate households
        A_hh[:] = np.repeat(np.nan,par.transition_T)
        C_hh[:] = np.repeat(np.nan,par.transition_T)

        if use_jac_hh:            
        
            A_hh[:] = ss.A_hh 
            A_hh[:] += jac_hh.A_r@(r-ss.r) 
            A_hh[:] += jac_hh.A_w@(w-ss.w) 

            C_hh[:] = ss.C_hh 
            C_hh[:] += jac_hh.C_r@(r-ss.r) 
            C_hh[:] += jac_hh.C_w@(w-ss.w) 
        
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
                A_hh[t] = np.sum(sol.path_a[t]*D_lag)
                C_hh[t] = np.sum(sol.path_c[t]*D_lag)

        ######################
        # III. check targets #
        ######################

        clearing_A[:] = A[:]-A_hh
        clearing_C[:] = C[:]-C_hh