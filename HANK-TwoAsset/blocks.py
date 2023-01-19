import numpy as np
import numba as nb

from GEModelTools import lag, lead

@nb.njit
def block_pre(par,ini,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    for ncol in range(ncols):

        # unpack
        B = path.B[ncol,:]
        Div_int = path.Div_int[ncol,:]
        Div_k = path.Div_k[ncol,:]
        Div = path.Div[ncol,:]
        eg = path.eg[ncol,:]
        G = path.G[ncol,:]
        I = path.I[ncol,:]
        invest_res = path.invest_res[ncol,:]
        Ip = path.Ip[ncol,:]
        K = path.K[ncol,:]
        N = path.N[ncol,:]
        p_eq = path.p_eq[ncol,:]
        p_int = path.p_int[ncol,:]
        p_k = path.p_k[ncol,:]
        Pi_increase = path.Pi_increase[ncol,:]
        Pi = path.Pi[ncol,:]
        psi = path.psi[ncol,:]
        q = path.q[ncol,:]
        Q = path.Q[ncol,:]
        qB = path.qB[ncol,:]
        r = path.r[ncol,:]
        ra = path.ra[ncol,:]
        rk = path.rk[ncol,:]
        rl = path.rl[ncol,:]
        s = path.s[ncol,:]
        tau = path.tau[ncol,:]
        valuation_res = path.valuation_res[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        Z = path.Z[ncol,:]
        NKPC_res = path.NKPC_res[ncol,:]

        #################
        # a. production #
        #################

            # inputs: N,Ip,s,r,Q
            # outputs: Y,w,rk,S,Div

        # Ip -> I and K
        I[:] = lag(ini.Ip, Ip)
        for t in range(par.T):
            K_lag = K[t-1] if t > 0 else ini.K
            K[t] = (1-par.delta_K)*K_lag + I[t]

        # N,K -> Y
        Y[:] = par.Theta*K**par.alpha*N**(1-par.alpha)

        # N,K,s -> w,r
        w[:] = s*(1-par.alpha)*par.Theta*K**par.alpha*N**(-par.alpha)
        rk[:] = s*par.alpha*par.Theta*K**(par.alpha-1)*N**(1-par.alpha)

        # check valuation residual
        Q_plus = lead(Q,ss.Q)
        r_plus = lead(r,ss.r)
        rk_plus = lead(rk,ss.rk)
        rk_plus2 = lead(rk_plus,ss.rk)

        Q_alt = 1.0/(1.0+r_plus)*(rk_plus2+(1.0-par.delta_K)*Q_plus)
        valuation_res[:] = Q_alt - Q[:]

        # check investment residual
        Ip_plus = lead(Ip,ss.I)
        I_plus = lead(I,ss.I)

        Sp = par.phi_K/2*(Ip/I-1.0)**2
        Spderiv = par.phi_K*(Ip/I-1.0)
        Spderiv_plus = par.phi_K*(Ip_plus/I_plus-1.0)

        LHS = 1.0+Sp+Ip/I*Spderiv
        RHS = Q+1.0/(1.0+r_plus)*(Ip_plus/I_plus)**2*Spderiv_plus
        invest_res[:] = RHS-LHS 

        # Y,w,N,I -> psi,Div,Div_k,Div_int
        I_lag = lag(ini.I, I)
        S = par.phi_K/2*(I/I_lag-1.0)**2
        psi[:] = I*S

        Div[:] = Y-w*N-I-psi
        Div_k[:] = rk*K-I-psi
        Div_int[:] = (1-s)*Y


        ###########
        # b. NKPC #
        ###########

            # input: s, Pi
            # output: 

        for t_ in range(par.T):

            t = (par.T-1)-t_
            
            kappa = (1-par.xi_p)*(1-par.xi_p/(1+ss.r))/par.xi_p*par.e_p/(par.v_p+par.e_p-1)

            gap = 0
            for k in range(t_+1):
                gap += 1/(1+ss.r)**k*(s[t+k]-(par.e_p-1)/par.e_p)

            Pi_increase[t] = kappa*gap

        for t in range(par.T):

            Pi_lag = Pi[t-1] if t > 0 else ini.Pi
            NKPC_res[t] = (Pi[t]-Pi_lag)-Pi_increase[t]


        ##################
        # c. mutual fund #
        ##################

            # inputs: Div,r
            # outputs: rl,q,ra

        r_lag = lag(ini.r, r)
        rl[:] = r_lag - par.xi

        for t_ in range(par.T):
            t = (par.T-1) - t_

            # q
            q_plus = q[t+1] if t < par.T-1 else ss.q
            q[t] = (1+par.delta_q*q_plus) / (1+r[t])

            # p_eq
            p_eq_plus = p_eq[t+1] if t < par.T-1 else ss.p_eq
            Div_plus = Div[t+1] if t < par.T-1 else ss.Div
            p_eq[t] = (Div_plus+p_eq_plus) / (1+r[t])

            # p_k
            Div_k_plus = Div_k[t+1] if t < par.T-1 else ss.Div_k
            p_k_plus = p_k[t+1] if t < par.T-1 else ss.p_k
            p_k[t] = (p_k_plus+Div_k_plus) / (1+r[t])

            # p_int
            Div_int_plus = Div_int[t+1] if t < par.T-1 else ss.Div_int
            p_int_plus = p_int[t+1] if t < par.T-1 else ss.p_int
            p_int[t] = (p_int_plus+Div_int_plus) / (1+r[t])

        A_lag = ini.A
        term_L = (1+rl[0])*ini.L+par.xi*ini.L

        term_B = (1+par.delta_q*q[0])*ini.B
        term_eq = p_eq[0]+Div[0]

        ra[0] = (term_B+term_eq-term_L)/A_lag-1
        ra[1:] = r[:-1]


        ###################
        # d. fiscal block #
        ###################

            # inputs: q, w, eg
            # outputs: tau, Z, G

        G[:] = ss.G * (1 + eg)
        for t in range(par.T):

            B_lag = B[t-1] if t > 0 else ini.B
            tau_no_shock = par.phi_tau*ss.q*(B_lag-ss.B) / ss.Y + ss.tau
            delta_tau = ((1-par.phi_G)*ss.G*eg[t]) / w[t] / N[t]
            tau[t] = delta_tau + tau_no_shock
            B_no_shock = (ss.G + (1 + par.delta_q*q[t])*B_lag-tau_no_shock*w[t]*N[t]) / q[t]
            delta_B = par.phi_G*ss.G*eg[t] / q[t]
            B[t] = delta_B + B_no_shock
            qB[t] = q[t]*B[t]
            Z[t] = (1-tau[t])*w[t]*N[t]


@nb.njit
def block_post(par, ini, ss, path, ncols=1):
    """ evaluate transition path-after household block """

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_L = path.clearing_L[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        em = path.em[ncol,:]
        fisher_res = path.fisher_res[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        I = path.I[ncol,:]
        L = path.L[ncol,:]
        N = path.N[ncol,:]
        p_eq = path.p_eq[ncol,:]
        Pi_w_increase = path.Pi_w_increase[ncol,:]
        Pi_w = path.Pi_w[ncol,:]
        Pi = path.Pi[ncol,:]
        psi = path.psi[ncol,:]
        qB = path.qB[ncol,:]
        r = path.r[ncol,:]
        s_w = path.s_w[ncol,:]
        tau = path.tau[ncol,:]
        w_res = path.w_res[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        NKPC_w_res = path.NKPC_w_res[ncol,:]
        C_hh = path.C_hh[ncol,:]
        L_hh = path.L_hh[ncol,:]
        A_hh = path.A_hh[ncol,:]
        UCE_hh = path.UCE_hh[ncol,:]

        ##########################
        # a. wage phillips curve #
        ##########################

        kappa_w = (1-par.xi_w)*(1-par.xi_w*par.beta_mean)/par.xi_w*par.e_w/(par.v_w+par.e_w-1)

        for t_ in range(par.T):

            t = (par.T-1) - t_

            s_w[t] = par.nu*N[t]**(1/par.frisch) / ((1-tau[t])*w[t]*UCE_hh[t])

            gap_w = 0
            for k in range(t_+1):
                gap_w += par.beta_mean**k * (s_w[t+k]-(par.e_w-1)/par.e_w)
                
            Pi_w_increase[t] = kappa_w*gap_w

        for t in range(par.T):
            Pi_lag = Pi[t-1] if t > 0 else ss.Pi
            NKPC_w_res[t] = (Pi_w[t]-Pi_lag) - Pi_w_increase[t]


        ####################
        # b. Taylor+Fisher #
        ####################
        
            # inputs: Pi
            # outputs: i

        for t in range(par.T):
            i_lag = i[t-1] if t > 0 else ini.i
            i[t] = par.rho_m*i_lag + (1-par.rho_m)*(ss.r+par.phi_pi*Pi[t]) + em[t]

        Pi_plus = lead(Pi,ss.Pi)
        fisher_res[:] = 1+i-(1+r)*(1+Pi_plus)


        ####################
        # c. wage residaul #
        ####################

        w_lag = lag(ini.w,w)
        w_res[:] = np.log(w/w_lag)-(Pi_w-Pi)


        ######################
        # d. market clearing #
        ######################

        # Y
        L[:] = L_hh
        L_lag = lag(ini.L, L)
        clearing_Y[:] = Y - (C_hh + G + I + psi + par.xi*L_lag)

        # A
        A[:] = p_eq+qB-L
        clearing_A[:] = A_hh - A

        # L
        clearing_L[:] = L_hh - L