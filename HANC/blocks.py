import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    # par, ini, ss, path are namespaces
    # ncols specifies have many versions of the model to evaluate at once
    #   path.VARNAME have shape=(len(unknowns)*par.T,par.T)
    #   path.VARNAME[0,t] for t in [0,1,...mpar.T] is always used outside of this function

    for thread in nb.prange(ncols):
        
        # unpack
        Gamma = path.Gamma[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]

        rk = path.rk[thread,:]
        r = path.r[thread,:]
        w = path.w[thread,:]
        
        Y = path.Y[thread,:]
        C = path.C[thread,:]

        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]

        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]

        #################
        # implied paths #
        #################

        # lags and leads of unknowns and shocks
        K_lag = lag(ini.K,K) # copy, same as [ss.K,K[0],K[1],...,K[-2]]
        
        # example: K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

        # VARNAME is used for reading values
        # VARNAME[:] is used for writing in-place

        # a. exogenous
        L[:] = 1.0

        # b. implied prices (remember K is input -> K_lag is known)
        rk[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
        r[:] = rk-par.delta
        w[:] = (1.0-par.alpha)*Gamma*(rk/(par.alpha*Gamma))**(par.alpha/(par.alpha-1.0))

        # c. production and consumption
        Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag

@nb.njit
def block_post(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    for thread in nb.prange(ncols):

        # unpack
        Gamma = path.Gamma[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]

        rk = path.rk[thread,:]
        r = path.r[thread,:]
        w = path.w[thread,:]
        
        Y = path.Y[thread,:]
        C = path.C[thread,:]

        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]

        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = K-A_hh
        clearing_C[:] = C-C_hh            