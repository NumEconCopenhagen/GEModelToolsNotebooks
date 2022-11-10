import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    # par, ini, ss, path are namespaces
    # nncols specifies have many versions of the model to evaluate at once
    #   path.VARNAME have shape=(len(unknowns)*par.T,par.T)
    #   path.VARNAME[0,t] for t in [0,1,...,par.T] is always used outside of this function

    for ncol in nb.prange(ncols):
        
        # unpack
        A = path.A[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        Gamma = path.Gamma[ncol,:]
        I = path.I[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        r = path.r[ncol,:]
        rk = path.rk[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]

        #################
        # implied paths #
        #################

        # lags and leads of unknowns and shocks
        K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
        
        # example: K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

        # VARNAME is used for reading values
        # VARNAME[:] is used for writing in-place

        # a. exogenous
        L[:] = 1.0

        # b. implied prices (remember K is input -> K_lag is known)
        rk[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
        r[:] = rk-par.delta
        w[:] = (1.0-par.alpha)*Gamma*(rk/(par.alpha*Gamma))**(par.alpha/(par.alpha-1.0))

        # c. production and investment
        Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)
        I[:] = (K-K_lag)+par.delta*K_lag

        # d. total assets
        A[:] = K[:]

@nb.njit
def block_post(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    for ncol in nb.prange(ncols):

        # unpack
        A = path.A[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        Gamma = path.Gamma[ncol,:]
        I = path.I[ncol,:]
        K = path.K[ncol,:]
        L = path.L[ncol,:]
        r = path.r[ncol,:]
        rk = path.rk[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = A-A_hh
        clearing_Y[:] = Y-C_hh-I