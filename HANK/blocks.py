import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C = path.C[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_NE = path.clearing_NE[ncol,:]
        d = path.d[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        NE_hh = path.NE_hh[ncol,:]
        NE = path.NE[ncol,:]
        nkpc_res = path.nkpc_res[ncol,:]
        pi = path.pi[ncol,:]
        psi = path.psi[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]

        #################
        # implied paths #
        #################

        # a. firms
        NE[:] = Y/Z

        psi[:] = par.mu/(par.mu-1)/(2*par.kappa)*np.log(1+pi)**2*Y
        d[:] = Y-w*NE-psi

        # b. monetary policy
        i[:] = istar + par.phi*pi + par.phi_y*(Y-ss.Y)
        i_lag = lag(ss.i,i)
        r[:] = (1+i_lag)/(1+pi)-1

        # c. government
        B[:] = ss.B
        tau[:] = ss.r*B + par.tau_r_fac*(r-ss.r)*B
        G[:] = tau-r*B
        
        # d. aggregates
        A[:] = B[:] = ss.B
        C[:] = Y-G-psi

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        A_hh = path.A_hh[ncol,:]
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C = path.C[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_NE = path.clearing_NE[ncol,:]
        d = path.d[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        NE_hh = path.NE_hh[ncol,:]
        NE = path.NE[ncol,:]
        nkpc_res = path.nkpc_res[ncol,:]
        pi = path.pi[ncol,:]
        psi = path.psi[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]
        
        #################
        # check targets #
        #################

        # a. phillips curve
        r_plus = lead(r,ss.r)
        pi_plus = lead(pi,ss.pi)
        Y_plus = lead(Y,ss.Y)

        nkpc_res[:] = par.kappa*(w/Z-1/par.mu) + 1/(1+r_plus)*Y_plus/Y*np.log(1+pi_plus) - np.log(1+pi)

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_C[:] = C-C_hh
        clearing_NE[:] = NE-NE_hh