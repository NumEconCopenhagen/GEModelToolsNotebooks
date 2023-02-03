import numpy as np
import numba as nb

from GEModelTools import lag, lead, prev, next

@nb.njit
def production_firm(par,ini,ss,w,rk,Y,N,Kd,s):
    
    KN_ratio = par.alpha/(1-par.alpha)*w/rk
    
    K_fac = par.Theta*KN_ratio**(par.alpha-1)
    N_fac = par.Theta*KN_ratio**(par.alpha)

    Kd[:] = Y/K_fac
    N[:] = Y/N_fac

    s[:] = rk/K_fac + w/N_fac

@nb.njit
def capital_firm(par,ini,ss,r,rk,Ip,Q,invest_res,I,K,psi,Div_k):

    for k in range(par.T):

        t = par.T-1-k
        
        Q_plus = next(Q,t,ss.Q)
        rk_plus = next(rk,t,ss.rk)

        Q[t] = (rk_plus+(1-par.delta_K)*Q_plus)/(1+r[t])

    # check investment residual            
    Ip_lag = lag(ini.Ip,Ip)
    r_plus = lead(r,ss.r)
    Ip_plus = lead(Ip,ss.I)

    Sp = par.phi_K/2*(Ip/Ip_lag-1.0)**2
    Spderiv = par.phi_K*(Ip/Ip_lag-1.0)
    Spderiv_plus = par.phi_K*(Ip_plus/Ip-1.0)

    LHS = 1.0+Sp+Ip/Ip_lag*Spderiv
    RHS = Q+1.0/(1.0+r_plus)*(Ip_plus/Ip)**2*Spderiv_plus

    invest_res[:] = RHS-LHS 

    # accumulate capital
    I[:] = lag(ini.Ip, Ip)
    for t in range(par.T):
        K_lag = prev(K,t,ini.K)
        K[t] = (1-par.delta_K)*K_lag + I[t]

    # dividends
    I_lag = lag(ini.I, I)
    S = par.phi_K/2*(I/I_lag-1.0)**2
    psi[:] = I*S
    Div_k[:] = rk*K-I-psi

@nb.njit
def price_setters(par,ini,ss,s,Pi,Y,NKPC_res,Div_int):

    Pi_plus = lead(Pi,ss.Pi)
    Y_plus = lead(Y,ss.Y)

    LHS = Pi
    RHS = par.kappa * (s-(par.e_p-1)/par.e_p) + par.beta_mean * Y_plus/Y * Pi_plus
    
    NKPC_res[:] = LHS-RHS

    Div_int[:] = (1-s)*Y
    
@nb.njit
def mutual_fund(par,ini,ss,Div_k,Div_int,r,Div,q,p_eq,rl,ra):

    Div[:] = Div_k+Div_int

    r_lag = lag(ini.r, r)
    rl[:] = r_lag - par.xi

    for t_ in range(par.T):

        t = (par.T-1) - t_

        # q
        q_plus = next(q,t,ss.q)
        q[t] = (1+par.delta_q*q_plus) / (1+r[t])

        # p_eq
        p_eq_plus = next(p_eq,t,ss.p_eq)
        Div_plus = next(Div,t,ss.Div)
        p_eq[t] = (Div_plus+p_eq_plus) / (1+r[t])

    A_lag = ini.A
    term_L = (1+rl[0])*ini.L+par.xi*ini.L

    term_B = (1+par.delta_q*q[0])*ini.B
    term_eq = p_eq[0]+Div[0]

    ra[0] = (term_B+term_eq-term_L)/A_lag-1
    ra[1:] = r[:-1]

@nb.njit
def government(par,ini,ss,eg,w,N,q,G,tau,B):

    G[:] = ss.G*(1+eg)
    for t in range(par.T):

        B_lag = prev(B,t,ini.B)

        tau_no_shock = ss.tau + par.phi_tau*ss.q*(B_lag-ss.B)/ss.Y
        delta_tau = (1-par.phi_G)*ss.G*eg[t]/(w[t]*N[t])
        
        tau[t] = tau_no_shock + delta_tau
        B[t] = ((1+par.delta_q*q[t])*B_lag+G[t]-tau[t]*w[t]*N[t])/q[t]

@nb.njit
def hh_income(par,ini,ss,N,w,tau,Z):
    Z[:] = (1-tau)*w*N

@nb.njit
def union(par,ini,ss,tau,w,UCE_hh,N,Pi_w,NKPC_w_res):

    Pi_w_plus = lead(Pi_w,ss.Pi_w)

    LHS = Pi_w
    RHS = par.kappa_w * ( par.nu*N**(1/par.frisch) - (par.e_w-1)/par.e_w*(1-tau)*w*UCE_hh ) + par.beta_mean * Pi_w_plus
    NKPC_w_res[:] = LHS-RHS

@nb.njit
def taylor(par,ini,ss,em,Pi,i):

    for t in range(par.T):
        i_lag = prev(i,t,ini.i)
        i[t] = par.rho_m*i_lag+(1-par.rho_m)*(ss.r+par.phi_pi*Pi[t])+em[t]

@nb.njit
def fisher(par,ini,ss,Pi,i,r,fisher_res):

    Pi_plus = lead(Pi,ss.Pi)
    fisher_res[:] = 1+i-(1+r)*(1+Pi_plus)

@nb.jit
def real_wage(par,ini,ss,w,Pi_w,Pi,w_res):

    w_lag = lag(ini.w,w)
    w_res[:] = np.log(w/w_lag)-(Pi_w-Pi)

@nb.jit
def market_clearing(par,ini,ss,Y,L_hh,C_hh,G,I,psi,Kd,K,q,B,p_eq,A_hh,qB,A,L,clearing_Y,clearing_K,clearing_A,clearing_L):

    # Y
    L[:] = L_hh
    L_lag = lag(ini.L, L)
    clearing_Y[:] = Y - (C_hh + G + I + psi + par.xi*L_lag)
    
    # K
    clearing_K[:] = Kd-K

    # A
    qB[:] = q*B
    A[:] = p_eq+qB-L
    clearing_A[:] = A_hh - A

    # L
    clearing_L[:] = L_hh - L