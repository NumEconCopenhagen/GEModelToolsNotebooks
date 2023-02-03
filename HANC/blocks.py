import numpy as np
import numba as nb

from GEModelTools import lag, lead

# lags and leads of unknowns and shocks
# K_lag = lag(ini.K,K) # copy, same as [ini.K,K[0],K[1],...,K[-2]]
# K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

@nb.njit
def capital_accumulation(par,ini,ss,K,K_lag,I):

    # a. capital for production
    K_lag[:] = lag(ini.K,K)

    # b. investment
    I[:] = (K-K_lag)+par.delta*K_lag    

@nb.njit
def production_firm(par,ini,ss,Gamma,K_lag,L,rk,w,Y):

    # a. implied prices (remember K and L are inputs)
    rk[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
    w[:] = (1.0-par.alpha)*Gamma*(K_lag/L)**par.alpha
    
    # b. production and investment
    Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)

@nb.njit
def mutual_fund(par,ini,ss,K,rk,A,r):

    # a. total assets
    A[:] = K

    # b. return
    r[:] = rk-par.delta

@nb.njit
def market_clearing(par,ini,ss,A,A_hh,L,L_hh,Y,C_hh,I,clearing_A,clearing_L,clearing_Y):

    clearing_A[:] = A-A_hh
    clearing_L[:] = L-L_hh
    clearing_Y[:] = Y-C_hh-I
