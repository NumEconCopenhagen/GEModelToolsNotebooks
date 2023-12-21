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

    # b. a and l
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)
    par.l_grid[:] = equilogspace(par.l_min,par.l_max,par.Nl)
    
    # c. e
    sigma = np.sqrt(par.sigma_e**2*(1-par.rho_e**2))
    par.z_grid[:],ss.z_trans[0,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_e,sigma,n=par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    ss.Dbeg = get_Dbeg(model,e_ergodic)

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Nl,par.Na))
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            
            e = par.z_grid[i_z]
            Z = (1-ss.tau)*ss.w*ss.N
            Ze = Z*e

            for i_a in range(par.Na):

                a_lag = par.a_grid[i_a]
                d = ss.ra*a_lag
                m = (1 + ss.rl)*par.l_grid + Ze + d
                c = m
                v_a[i_fix,i_z,:,i_a] = (1 + ss.rl)*c**(-par.sigma)

        for i_a in range(par.Na):
            ss.vbeg_l[i_fix,:,:,i_a] = ss.z_trans[i_fix] @ v_a[i_fix,:,:,i_a]

def get_Dbeg(model,e_ergodic):
    """ initiate Dbeg that is optimized along the illiquid asset grid """

    par = model.par
    ss = model.ss

    A_target = ss.A

    # a. find closest but smaller grid value to target
    i_a_target = np.abs(par.a_grid-A_target).argmin() # find grid value which is closest to the target
    if par.a_grid[i_a_target] > A_target:
        i_a_target += -1  # select grid value that is smaller than target

    assert i_a_target <= par.Na, 'illiquid asset target outside of grid'
    
    # b. find weights between grid value and target,
    # s.t. w*a_grid[i]+(1-w)*a_grid[i+1] = a_target
    i_a_weight = (A_target-par.a_grid[i_a_target + 1])/(par.a_grid[i_a_target]-par.a_grid[i_a_target + 1])

    # c. fill Dbeg
    Dbeg = np.zeros_like(ss.Dbeg)
    Dz = np.zeros((par.Nfix,par.Nz))
    
    for i_fix in range(par.Nfix):
        
        Dz[i_fix,:] = e_ergodic/par.Nfix
        
        for i_z in range(par.Nz):
            for i_l in range(par.Nl):

                # distribute population shares along the relevant grids for the illiquid asset target
                Dbeg[i_fix,i_z,i_l,i_a_target] = Dz[i_fix,i_z]/par.Nl*i_a_weight
                Dbeg[i_fix,i_z,i_l,i_a_target + 1] = Dz[i_fix,i_z]/par.Nl*(1-i_a_weight)
    
    # d. check
    Dbeg_sum = np.sum(Dbeg)
    assert np.isclose(Dbeg_sum,1.0),f'sum(ss.Dbeg) = {Dbeg_sum:12.8f},should be 1.0'

    return Dbeg

def evaluate_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. exogenous and targets
    ss.Y = 1.0  # normalization
    ss.N = 1.0  # normalization

    ss.r = par.r_ss_target
    ss.L = par.L_Y_ratio*ss.Y
    ss.Kd = ss.K = par.K_Y_ratio*ss.Y
    ss.G = par.G_Y_ratio*ss.Y
    ss.qB = par.qB_Y_ratio*ss.Y
    ss.A = par.A_Y_ratio*ss.Y

    par.mu_p = ss.Y/(ss.Y-ss.r*(ss.A+ss.L-ss.qB-ss.K))
    par.e_p = par.mu_p/(par.mu_p-1)
    par.e_w = par.e_p

    # zero inflation
    ss.Pi = 0.0
    ss.Pi_w = 0.0

    # shocks
    ss.eg = 0.0
    ss.em = 0.0

    # b. central bank
    ss.i = ss.r

    # c. mutal fund
    ss.ra = ss.r
    ss.rl = ss.r-par.xi
    ss.q = 1.0/(1.0 + ss.r-par.delta_q)
    ss.B = par.qB_Y_ratio*ss.Y/ss.q

    # d. production firms and price setters
    ss.rk = ss.r+par.delta_K
    ss.s = (par.e_p-1)/par.e_p
    par.alpha = ss.rk*ss.K/ss.s
    par.Theta = ss.Y*ss.K**(-par.alpha)*ss.N**(par.alpha-1)
    ss.w = ss.s*(1-par.alpha)/ss.N
    ss.Div_int = (1-ss.s)*ss.Y

    # e. capital firms
    ss.Q = 1.0
    ss.psi = 0.0
    ss.I = par.delta_K*ss.K
    ss.Ip = ss.I
    ss.Div_k = ss.rk*ss.K-ss.I

    # f. all firms
    ss.Div = ss.Y-ss.w*ss.N-ss.I
    assert np.isclose(ss.Div-ss.Div_int-ss.Div_k,0.0)
    ss.p_eq = ss.Div/ss.r

    # h. government
    ss.tau = (ss.G + (1 + par.delta_q*ss.q)*ss.B-ss.q*ss.B)/(ss.w*ss.N)

    # i. households
    ss.Z = (1-ss.tau)*ss.w*ss.N
    par.A_target = ss.A

    model.solve_hh_ss()
    model.simulate_hh_ss()

    v_prime_N_unscaled = ss.N**(1/par.frisch)
    u_prime_e = ss.UCE_hh
    par.nu = (par.e_w-1)/par.e_w*(1-ss.tau)*ss.w*u_prime_e/v_prime_N_unscaled

    # j. clearing
    ss.clearing_Y = ss.Y-(ss.C_hh + ss.G + ss.I + ss.psi + par.xi*ss.L_hh)

    ss.A = ss.p_eq + ss.qB - ss.L
    ss.clearing_A = ss.A_hh-ss.A

    ss.clearing_L = ss.L_hh-ss.L    
    
def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    t0 = time.time()

    par = model.par
    ss = model.ss

    par.beta_mean = x[0]
    evaluate_ss(model,do_print=do_print)

    if do_print: print(f' beta = {par.beta_mean:16.12f} -> {ss.clearing_Y = :16.12f} [{elapsed(t0)}]')
    
    return ss.clearing_Y

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
        print(f'Discrepancy in L = {ss.clearing_L:12.8f}')