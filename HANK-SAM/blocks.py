import numpy as np
import numba as nb

from GEModelTools import lag, lead, bound, bisection
   
@nb.njit
def NKPC_eq(x,par,ss,w,Gamma,Y,Y_plus,Pi_plus):  
    
    LHS_NKPC = (1-par.epsilon) + par.epsilon*w/Gamma
        
    RHS_NKPC_cur = par.theta*(x-ss.Pi)*x
    RHS_NKPC_fut = par.beta_mean*par.theta*(Pi_plus-ss.Pi)*Pi_plus*Y_plus/Y

    NKPC = LHS_NKPC - (RHS_NKPC_cur-RHS_NKPC_fut)
    
    return NKPC    

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    # par, sol, sim, ss, path are namespaces
    # ncols specifies have many versions of the model to evaluate at once
    #   path.VARNAME have shape=(len(unknowns)*par.T,par.T)
    #   path.VARNAME[0,t] for t in [0,1,...mpar.T] is always used outside of this function

    for ncol in range(ncols):

        # unpack
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        d = path.d[ncol,:]
        EU = path.EU[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        N = path.N[ncol,:]
        NKPC = path.NKPC[ncol,:]
        Pi_w = path.Pi_w[ncol,:]
        Pi = path.Pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        U = path.U[ncol,:]
        UE = path.UE[ncol,:]
        w = path.w[ncol,:]
        wh = path.wh[ncol,:]
        WPC = path.WPC[ncol,:]
        Y = path.Y[ncol,:]
        NKPC = path.NKPC[ncol,:]
        
        ####################
        # I. implied paths #
        ####################

        # a. solve NKPC                
        for t_ in range(par.T):

            t = (par.T-1)-t_

            Y_plus = Y[t+1] if t < par.T-1 else ss.Y
            Pi_plus = Pi[t+1] if t < par.T-1 else ss.Pi
            Pi[t] = bisection(NKPC_eq,0.9,1.1,args=(par,ss,w[t],Gamma[t],Y[t],Y_plus,Pi_plus))

        # b. monetary policy
        i[:] = ((1+ss.r)*Pi**par.varepsilon_pi)-1.0
        i_lag = lag(ini.i,i)
        r[:] = ((1+i_lag)/Pi)-1.0

        # c. firm behavior
        N[:] = Y/Gamma
        d[:] = Y-w*N
        
        # d. labor market
        U[:] = ss.U*(N/ss.N)**par.varepsilon_U

        U_lag = lag(ini.U,U)
        
        if par.U_residual == 'UE':

            EU[:] = bound(ss.EU*(N/ss.N)**par.varepsilon_EU,0.0,1.0)
            UE[:] = bound(1.0-(U-EU*(1-U_lag))/U_lag,0.0,1.0)

        elif par.U_residual == 'EU':
        
            UE[:] = bound(ss.UE*(N/ss.N)**par.varepsilon_UE,0.0,1.0)
            EU[:] = bound((U-UE*U_lag)/(1-U_lag),0.0,1.0)

        else:

            raise ValueError('par.U_residual should be UE or EU')

        # e. government
        for t in range(par.T):

            # i. lag
            B_lag = B[t-1] if t > 0 else ini.B
            
            # ii. tau
            x = bound((t-par.t_B)/par.Delta_B,0,1)
            omega = 3*x**2-2*x**3
            tau_tilde = ss.tau*(B_lag/ss.B)**par.varepsilon_B

            tau[t] = (1.0-omega)*ss.tau + omega*tau_tilde
            
            # iii. government debt
            B[t] = (1+r[t])*B_lag+G[t]-tau[t]*w[t]*N[t]-d[t]
        
        # household income
        wh[:] = (1-tau)*w*N / (par.phi*U+(1-U))

        # e. aggregates
        A[:] = B

@nb.njit
def block_post(par,ini,ss,path,ncols=1):
    """ evaluate transition path """

    for ncol in range(ncols):

        # unpack
        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        d = path.d[ncol,:]
        EU = path.EU[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]        
        i = path.i[ncol,:]
        N = path.N[ncol,:]
        NKPC = path.NKPC[ncol,:]
        Pi_w = path.Pi_w[ncol,:]
        Pi = path.Pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        U = path.U[ncol,:]
        UE = path.UE[ncol,:]
        w = path.w[ncol,:]
        wh = path.wh[ncol,:]
        WPC = path.WPC[ncol,:]
        Y = path.Y[ncol,:]


        # lagged
        w_lag = lag(ini.w,w)
        
        Pi_w[:] = w/w_lag*Pi
        Pi_w_plus = lead(Pi_w,ss.Pi_w)
        N_plus = lead(N,ss.N)

        # b. WPC
        v_prime = par.nu*N**(1/par.varphi)
        u_prime = C_hh**(-par.sigma)
        LHS_WPC = (1-par.epsilon_w)*(1-tau)*w + par.epsilon_w*v_prime/u_prime
        
        RHS_WPC_cur = par.theta_w*(Pi_w-ss.Pi_w)*Pi_w
        RHS_WPC_fut = par.beta_mean*par.theta_w*(Pi_w_plus-ss.Pi_w)*Pi_w_plus*N_plus/N

        WPC[:] =  LHS_WPC - (RHS_WPC_cur-RHS_WPC_fut)

        # c. market clearing
        clearing_A[:] = A-A_hh
        clearing_Y[:] = Y-C_hh-G