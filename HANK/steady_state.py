# find steady state

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

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. e
    sigma = np.sqrt(par.sigma_e**2*(1-par.rho_e**2))
    par.z_grid[:],ss.z_trans[0,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_e,sigma,n=par.Ne)

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

    va = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_z in range(par.Nz):

        e = par.z_grid[i_z]
        T = ss.d*e - ss.tau*e
        ne = 1.0*e

        c = (1+ss.r)*par.a_grid + ss.w*ne + T

        va[0,i_z,:] = c**(-par.sigma)

    ss.vbeg_a[0] = ss.z_trans[0]@va[0]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    # a. fixed
    ss.Z = 1.0
    ss.NE = 1.0
    ss.pi = 0.0
    
    # b. targets
    ss.r = par.r_target_ss
    ss.A = ss.B = par.B_target_ss
    ss.G = par.G_target_ss

    # c.. monetary policy
    ss.istar = ss.r
    ss.i = ss.istar + par.phi*ss.pi

    # d. firms
    ss.Y = ss.Z*ss.NE
    ss.w = ss.Z/par.mu
    ss.psi = 0.0
    ss.d = ss.Y-ss.w*ss.NE-ss.psi
    
    # e. government
    ss.tau = ss.r*ss.B + ss.G

    # f. household 
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    ss.A_hh = np.sum(ss.a*ss.D)
    ss.C_hh = np.sum(ss.c*ss.D)
    ss.NE_hh = np.sum(ss.ne*ss.D)

    # g. market clearing
    ss.C = ss.Y-ss.G-ss.psi

def objective_ss(x,model,do_print=False):
    """ objective function for finding steady state """

    par = model.par
    ss = model.ss

    par.beta = x[0]
    par.varphi = x[1]

    evaluate_ss(model,do_print=do_print)
    
    return np.array([ss.A_hh-ss.B,ss.NE_hh-ss.NE])

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()
    res = optimize.root(objective_ss,[par.beta, par.varphi],method='hybr',tol=par.tol_ss,args=(model))

    # final evaluation
    objective_ss(res.x,model)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f' beta   = {res.x[0]:8.4f}')
        print(f' varphi = {res.x[1]:8.4f}')
        print('')
        print(f'Discrepancy in B = {ss.A-ss.A_hh:12.8f}')
        print(f'Discrepancy in C = {ss.C-ss.C_hh:12.8f}')
        print(f'Discrepancy in N = {ss.NE-ss.NE_hh:12.8f}')
