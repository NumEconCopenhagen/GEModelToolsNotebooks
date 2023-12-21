import numpy as np
import numba as nb

from GEModelTools import prev,next,lag,lead

# @nb.njit
# def prev(x,t,inivalue):
#     if t > 0:
#         return x[t-1]
#     else:
#         return np.repeat(inivalue,x.shape[1])
    
# @nb.njit
# def next(x,t,ssvalue):
#     if t+1 < x.shape[0]:
#         return x[t+1]
#     else:
#         return np.repeat(ssvalue,x.shape[1])

@nb.njit
def government(par,ini,ss,eta,G,pB,tau,B):

    for t in range(par.T):
        
        B_lag = prev(B,t,ini.B)
        tau[t] = ss.tau + eta[t] + par.phi*(B_lag-ss.B) 
        B[t] = (B_lag + G[t] - tau[t])/pB[t]

    # WARNING: This would NOT work
    #  B_lag = lag(ini.B,B)
    #  tau[:] = ss.tau + par.phi*(B_lag-ss.B)
    #  B[:] = (B_lag + G - tau)/pB
    # Explanation: lag() creates COPY not a view into the lagged value

@nb.njit
def market_clearing(par,ini,ss,B,A_hh,clearing_B):

    clearing_B[:] = B-A_hh