import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def production(par,ini,ss,Gamma,w,Y,N,s):

    N[:] = Y/Gamma
    s[:] = w/Gamma

@nb.njit
def taylor(par,ini,ss,istar,pi,Y,i):

    i[:] = istar + par.phi*pi + par.phi_y*(Y-ss.Y)

@nb.njit
def fisher(par,ini,ss,i,pi,r):

    i_lag = lag(ini.i,i)
    r[:] = (1+i_lag)/(1+pi)-1

@nb.njit
def government(par,ini,ss,G,r,B,tau):
    
    B[:] = ss.B
    tau[:] = r*B + G

@nb.njit
def intermediary_goods(par,ini,ss,r,s,Y,pi,NKPC_res,adjcost,d):

    # a. Phillips curve
    r_plus = lead(r,ss.r)
    pi_plus = lead(pi,ss.pi)
    Y_plus = lead(Y,ss.Y)

    LHS = np.log(1+pi)
    RHS = par.kappa*(s-1/par.mu) + 1/(1+r_plus)*Y_plus/Y*np.log(1+pi_plus)
    NKPC_res[:] = LHS-RHS

    # b. adjustment costs and dividends
    adjcost[:] = par.mu/(par.mu-1)/(2*par.kappa)*np.log(1+pi)**2*Y
    d[:] = (1-s)*Y-adjcost

@nb.njit
def market_clearing(par,ini,ss,A,B,N,Y,G,adjcost,N_hh,A_hh,C_hh,r,w,clearing_N,clearing_A,clearing_Y):

    A[:] = B[:]

    if par.RA:

        C = Y-G-adjcost # derive consumption from ressource constraint
        C_plus = lead(C,ss.C_hh)
        r_plus = lead(r,ss.r)

        clearing_N[:] = N**par.nu - w/par.varphi_RA*C**(-par.sigma) # Euler equation
        clearing_A[:] = C**(-par.sigma) - par.beta_RA*(1+r_plus)*C_plus**(-par.sigma) # FOC for labor supply
        clearing_Y[:] = 0.0 # from using ressource constraint

    else:

        clearing_N[:] = N-N_hh
        clearing_A[:] = A-A_hh
        clearing_Y[:] = Y-(C_hh+G+adjcost)