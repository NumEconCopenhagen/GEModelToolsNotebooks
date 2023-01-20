import time
import numpy as np
from scipy import optimize

from consav import elapsed
from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. beta
    par.beta_grid[:] = np.linspace(par.beta_mean-par.beta_delta,par.beta_mean + par.beta_delta,par.Nfix)

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. e
    sigma = np.sqrt(par.sigma_e**2*(1-par.rho_e**2))
    par.z_grid[:],ss.z_trans[0,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_e,sigma,n=par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_fix in range(par.Nfix):
        ss.Dz[i_fix,:] = e_ergodic/par.Nfix
        ss.Dbeg[i_fix,:,0] = ss.Dz[i_fix,:]
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            
            e = par.z_grid[i_z]
            Z = (1-ss.tau)*ss.w*ss.N
            Ze = Z*e

            for i_a in range(par.Na):

                a_lag = par.a_grid[i_a]
                m = (1+ss.ra)*a_lag + Ze
                c = m
                v_a[i_fix,i_z,i_a] = (1+ss.ra)*c**(-par.sigma)

        for i_a in range(par.Na):
            ss.vbeg_a[i_fix,:,i_a] = ss.z_trans[i_fix]@v_a[i_fix,:,i_a]

def evaluate_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.Y = 1.0  # normalization
    ss.N = 1.0  # normalization

    ss.r = par.r_ss_target
    ss.K = par.K_Y_ratio*ss.Y
    ss.G = par.G_Y_ratio*ss.Y
    ss.qB = par.qB_Y_ratio*ss.Y

    par.e_p = par.mu_p/(par.mu_p-1)
    par.e_w = par.e_p

    # zero inflation
    ss.Pi = 0.0
    ss.Pi_w = 0.0
    ss.Pi_w_increase = 0.0
    ss.Pi_increase = 0.0

    # shocks
    ss.eg = 0.0
    ss.em = 0.0

    # b. central bank
    ss.i = ss.r

    # c. mutal fund
    ss.ra = ss.r
    ss.q = 1.0/(1.0 + ss.r-par.delta_q)
    ss.B = par.qB_Y_ratio*ss.Y/ss.q

    # d. intermediate goods
    ss.rk = ss.r+par.delta_K
    ss.s = (par.e_p-1)/par.e_p
    par.alpha = ss.rk*ss.K/ss.s
    par.Theta = ss.Y*ss.K**(-par.alpha)*ss.N**(par.alpha-1)
    ss.w = ss.s*(1-par.alpha)/ss.N

    ss.Div_int = (1-ss.s)*ss.Y
    ss.p_int = ss.Div_int/ss.r

    # e. capital firms
    ss.Q = 1.0
    ss.psi = 0.0
    ss.I = par.delta_K*ss.K
    ss.Ip = ss.I
    ss.Div_k = ss.rk*ss.K-ss.I
    ss.p_k = ss.Div_k/ss.r

    # f. all firms
    ss.Div = ss.Y-ss.w*ss.N-ss.I
    assert np.isclose(ss.Div-ss.Div_int-ss.Div_k,0.0)
    ss.p_eq = ss.Div/ss.r

    # g. unions
    ss.s_w = (par.e_w-1)/par.e_w

    # h. government
    ss.tau = (ss.G + (1 + par.delta_q*ss.q)*ss.B-ss.q*ss.B)/(ss.w*ss.N)

    # i. households
    ss.Z = (1-ss.tau)*ss.w*ss.N

    assert par.Nfix == 1

    model.solve_hh_ss()
    model.simulate_hh_ss()

    v_prime_N_unscaled = ss.N**(1/par.frisch)
    u_prime_e = ss.UCE_hh
    par.nu = ss.s_w*(1-ss.tau)*ss.w*u_prime_e/v_prime_N_unscaled

    # j. clearing
    ss.clearing_Y = ss.Y-(ss.C_hh + ss.G + ss.I + ss.psi)
    
    ss.A = ss.p_eq + ss.qB
    ss.clearing_A = ss.A_hh-ss.A
    
def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    par.beta_mean = x[0]
    evaluate_ss(model,do_print=do_print)

    if do_print:
        print(f' beta = {par.beta_mean:16.12f} -> {ss.clearing_A = :16.12f} [{elapsed(t0)}]')
    
    return ss.clearing_A

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    if do_print: print('find beta for market clearing')

    t0 = time.time()
    res = optimize.root(objective_ss,par.beta_mean,method='hybr',tol=par.tol_ss,args=(model,do_print))
    
    # b. final evaluation
    if do_print: print('\nfinal evaluation')
    objective_ss([par.beta_mean],model,do_print=do_print)

    # check targets
    if do_print:

        print(f'\nsteady state found in {elapsed(t0)}')
        print(f' beta   = {par.beta_mean:6.4}')
        print(f' nu     = {par.nu:6.4f}')
        print('')
        print(f'Discrepancy in Y = {ss.clearing_Y:12.8f}')
        print(f'Discrepancy in A = {ss.clearing_A:12.8f}')