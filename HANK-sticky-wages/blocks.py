import numpy as np
import numba as nb

from GEModelTools import lag, lead, prev, next
   
@nb.njit
def production(par,ini,ss,Gamma,pi_w,L,w,pi,Y):

    Gamma_lag = lag(ini.Gamma,Gamma)

    w[:] = Gamma
    Y[:] = Gamma*L
    pi[:] = (1+pi_w)/(Gamma/Gamma_lag)-1

@nb.njit
def central_bank(par,ini,ss,pi,i,r):

    # a. central bank
    for t in range(par.T):
        i_lag = prev(i,t,ini.i)
        i[t] = (1+i_lag)**par.rho_i*((1+ss.r)*(1+pi[t])**(par.phi_pi))**(1-par.rho_i)-1
 
    # b. Fisher
    pi_plus = lead(pi,ss.pi)
    r[:] = (1+i)/(1+pi_plus)-1
        
@nb.njit
def mutual_fund(par,ini,ss,r,q,ra):

    for k in range(par.T):
        t = par.T-1-k
        q_plus = next(q,t,ss.q)
        q[t] = (1+par.delta*q_plus)/(1+r[t])
    
    q_lag = lag(ini.q,q)
    ra[:] = (1+par.delta*q)/q_lag-1

@nb.njit
def government(par,ini,ss,G,chi,q,Y,B,tau):

    for t in range(par.T):
        
        B_lag = prev(B,t,ini.B)
        tau[t] = ss.tau + par.omega*ss.q*(B_lag-ss.B)/ss.Y
        B[t] = ((1+par.delta*q[t])*B_lag + G[t] + chi[t] - tau[t]*Y[t])/q[t]

@nb.njit
def NKWC(par,ini,ss,pi_w,L,tau,w,C_hh,NKWC_res):

    # a. phillips curve
    pi_w_plus = lead(pi_w,ss.pi_w)

    LHS = pi_w
    RHS = par.kappa*(par.varphi*L**par.nu - 1/par.mu*(1-tau)*w*C_hh**(-par.sigma)) + par.beta*pi_w_plus
    NKWC_res[:] = LHS-RHS

@nb.njit
def market_clearing(par,ini,ss,G,q,B,Y,C_hh,A_hh,A,clearing_A,clearing_Y,Z_hh,z_res):
        
    # a. aggregates
    A[:] = q*B
    z_res[:] = Z_hh - 1. 

    # b. market clearing
    clearing_A[:] = A-A_hh
    clearing_Y[:] = Y-C_hh-G