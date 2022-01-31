import math
import numpy as np
import numba as nb

from GEModelTools import lag, simulate_hh_path, prepare_simulation_1d_1d

# local
from household_problem import solve_hh_path

###############
# 2. evaluate #
###############
    
@nb.njit
def evaluate_path(par,sol,sim,ss,path,jac_hh,threads=1,use_jac_hh=False):
    """ evaluate transition path """

    # par, sol, sim, ss, path, jac_hh are namespaces
    # threads specifies have many parallel versions of the model to solve
    #   path.VARNAME have shape len(inputs_endo)*par.transtion_T
    #   path.VARNAME[:,0] is always used outside of this function
    # use_jac_hh specifies whether to get the household behavior from the steady state + jacobian
    # note: if threads > 1 then we should have use_jac_hh = False

    for thread in nb.prange(threads):

        # unpack
        Z = path.Z[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]

        rk = path.rk[thread,:]
        r = path.r[thread,:]
        w = path.w[thread,:]
        
        Y = path.Y[thread,:]
        C = path.C[thread,:]
        A = path.A[thread,:]

        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]

        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]

        # lag
        K_lag = lag(ss.K,K[:-1]) # first element is steady state

        ####################
        # I. implied paths #
        ####################

        # VARNAME is used for reading
        # VARNAME[:] is used for writing in-place

        # a. exogenous (implicite Z)
        L[:] = 1.0

        # b. implied prices (remember K is input -> K_lag is known)
        rk[:] = par.alpha*Z*(K_lag/L)**(par.alpha-1.0)
        r[:] = rk-par.delta
        w[:] = (1.0-par.alpha)*Z*(rk/(par.alpha*Z))**(par.alpha/(par.alpha-1.0))

        # c. production and consumption
        Y[:] = Z*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag

        # d. stocks equal capital
        A[:] = K

        #########################
        # II. household problem #
        #########################

        # note: 
        # A_hh is aggregate stocks *from the household side*
        # C_hh is aggregate consumption *from the household side*
        
        # a. solve and simulate households
        if use_jac_hh:            
        
            A_hh[:] = ss.A_hh # start fra steady state
            A_hh[:] += jac_hh.A_r@(r-ss.r) # effect from r
            A_hh[:] += jac_hh.A_w@(w-ss.w) # effect from w

            C_hh[:] = ss.C_hh 
            C_hh[:] += jac_hh.C_r@(r-ss.r) 
            C_hh[:] += jac_hh.C_w@(w-ss.w) 
        
        else:
            
            solve_hh_path(par,sol,ss,path)
            prepare_simulation_1d_1d(par,sol,sol.path_a,par.a_grid)
            simulate_hh_path(par,sol,sim)
        
            for t in range(par.transition_T):
                A_hh[t] = np.sum(sol.path_a[t]*sim.path_D[t])
                C_hh[t] = np.sum(sol.path_c[t]*sim.path_D[t])

        ######################
        # III. check targets #
        ######################

        clearing_A[:] = A-A_hh
        clearing_C[:] = C-C_hh