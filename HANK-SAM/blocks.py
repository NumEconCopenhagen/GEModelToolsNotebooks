import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def production(par,ini,ss,shock_TFP,delta,w,px,Vj,errors_Vj):

    Vj_plus = lead(Vj,ss.Vj)
    delta_plus = lead(delta,ss.delta)

    LHS = Vj
    RHS = shock_TFP*px-w + (1-delta_plus)*par.beta*Vj_plus
    
    errors_Vj[:] = LHS-RHS

@nb.njit
def labor_market(par,ini,ss,vt,ut,S,theta,delta,lambda_v,lambda_u,v,u,entry,errors_ut):

    theta[:] = vt/S

    lambda_v[:] = par.A*theta**(-par.alpha)
    lambda_u[:] = par.A*theta**(1-par.alpha)

    u[:] = (1-lambda_u)*ut
    v[:] = (1-lambda_v)*vt

    u_lag = lag(ini.u,u)
    v_lag = lag(ini.v,v)

    entry[:] = vt -(1-delta)*v_lag
    errors_ut[:] = ut - (u_lag + delta*(1-u_lag))

@nb.njit
def entry(par,ini,ss,lambda_v,Vj,errors_entry):
        
    LHS = -par.kappa + lambda_v*Vj
    RHS = 0

    errors_entry[:] = LHS-RHS

@nb.njit
def price_setters(par,ini,ss,shock_TFP,u,px,Pi,errors_Pi):

        LHS = 1-par.epsilon_p + par.epsilon_p*px

        Pi_plus = lead(Pi,ss.Pi)        
        shock_TFP_plus = lead(shock_TFP,ss.shock_TFP)
        u_plus = lead(u,ss.u)

        RHS = par.phi*(Pi-ss.Pi)*Pi - par.beta*par.phi*((Pi_plus-ss.Pi)*Pi_plus*(shock_TFP_plus*u_plus)/(shock_TFP*u))

        errors_Pi[:] = LHS-RHS

@nb.njit
def central_bank(par,ini,ss,Pi,R,RealR,q,RealR_ex_post):

    for t in range(par.T):

        R_lag = ss.R if t == 0 else R[t-1]
        R[t] = ss.R*(R_lag/ss.R)**(par.rho_R)*(Pi[t]/ss.Pi)**(par.delta_pi*(1-par.rho_R))
            
        if t < par.T-1:
            RealR[t] = R[t]/Pi[t+1]
        else:
            RealR[t] = R[t]/ss.Pi

    # iv. arbitrage
    for k in range(par.T):
        t = par.T-1-k
        q_plus = q[t+1] if t < par.T-1 else ss.q
        q[t] = (1+par.delta_q*q_plus)/RealR[t]

    q_lag = lag(ini.q,q)
    RealR_ex_post[:] = (1+par.delta_q*q)/q_lag

@nb.njit
def government(par,ini,ss,w,u,q,tau,B,qB,UI,Yt_hh):

    UI[:] = par.phi*w*u
    Yt_hh[:] = w*(1-u) + UI

    for t in range(par.T):
        
        B_lag = B[t-1] if t > 0 else ini.B
        tau[t] = ss.tau + par.omega*ss.q*(B_lag-ss.B)/ss.Yt_hh

        expenses = UI[t] 
        B[t] = ((1+par.delta_q*q[t])*B_lag + expenses - tau[t]*Yt_hh[t])/q[t]

    qB[:] = q*B
    
@nb.njit
def market_clearing(par,ini,ss,qB,S,A_hh,S_hh,errors_assets,errors_search):

    errors_assets[:] = qB-A_hh
    errors_search[:] = S-S_hh