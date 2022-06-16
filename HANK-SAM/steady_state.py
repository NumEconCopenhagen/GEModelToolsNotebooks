import numpy as np

from consav.grids import equilogspace
from consav.markov import tauchen, find_ergodic

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    # a. beta
    par.beta_grid[:] = np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nbeta)

    # b. a
    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)

    # c. z
    log_e_grid,_,_,_,_ = tauchen(0,par.rho_e,par.sigma_psi,n=par.Ne)       
    par.e_grid[:] = np.exp(log_e_grid)
    par.z_grid[:] = np.tile(par.e_grid,2)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    # a. transition matrix
    model.fill_z_trans_ss()

    # b. ergodic
    for i_beta in range(par.Nbeta):
        ss.Dz[i_beta,:] = find_ergodic(ss.z_trans[i_beta])/par.Nbeta
        ss.Dbeg[i_beta,:,0] = ss.Dz[i_beta,:]
        ss.Dbeg[i_beta,:,1:] = 0.0
        
    # c. impose mean-one for z
    par.z_grid[:] = par.z_grid/np.sum(par.z_grid*ss.Dz)

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    Va = np.zeros((par.Nfix,par.Nz,par.Na))

    # a. raw value        
    y = par.phi**par.u_grid*ss.wh*par.z_grid
    c = m = (1+ss.r)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)

    # b. expectation
    for i_beta in range(par.Nbeta):
        ss.vbeg_a[i_beta] = ss.z_trans[i_beta]@v_a

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Gamma = 1.0
    ss.N = 1.0
    ss.Pi = ss.Pi_w = 1.0
    
    # targets
    ss.tau = par.tau_target
    ss.EU = par.EU_target
    ss.UE = par.UE_target

    ss.r = par.r_ss_target
    ss.i = ((1.0+ss.r)*ss.Pi)-1.0

    # b. firms
    ss.Y = ss.Gamma*ss.N
    ss.w = (par.epsilon-1)/par.epsilon
    ss.d = ss.Y-ss.w*ss.N
    
    # c. labor markets
    ss.U = ss.EU/(ss.EU+ss.UE)
    
    # d. household problem
    ss.wh = (1-ss.tau)*ss.w*ss.N / (par.phi*ss.U+(1-ss.U))

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    ss.B = ss.A = ss.A_hh

    # e. government
    ss.G = ss.d + ss.tau*ss.w*ss.N - ss.r*ss.B

    # f. resource constraint
    ss.C = ss.Y-ss.G

    if do_print:
        
        C = ss.C/ss.Y
        G = ss.G/ss.Y
        print(f'GDP by spending: C = {C:.3f}, G = {G:.3f}')

        assert np.isclose(1.0,C+G)

        print(f'Implied B = {ss.B:6.3f}')
        print(f'Discrepancy in C = {ss.C-ss.C_hh:12.8f}')

    # h. union
    v_prime_N_unscaled = ss.N**(1/par.varphi)
    u_prime = ss.C**(-par.sigma)
    par.nu = (par.epsilon_w-1)/par.epsilon_w*(1-ss.tau)*ss.w*u_prime/v_prime_N_unscaled # WPC
    if do_print: print(f'Implied nu = {par.nu:6.3f}')