# find steady state

import time
import numpy as np
from root_finding import brentq

from EconModel import jit

from consav.grids import equilogspace
from consav.markov import tauchen, find_ergodic
from consav.misc import elapsed

import household_problem

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    par.a_grid[:] = equilogspace(0.0,par.a_max,par.Na)
    par.z_grid[:] = np.ones(par.Nz) # not used
    
    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        ss.Dbeg[i_fix,:,0] = np.array([1-ss.u,ss.u])   
        ss.Dbeg[i_fix,:,1:] = 0.0      
    
    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    model.set_hh_initial_guess()

def find_ss(model,do_print=False,fix_RealR=False):
    """ find the steady state"""

    t0 = time.time()
    
    find_ss_SAM(model,do_print=do_print)
    find_ss_HANK(model,do_print=do_print)

    if do_print: print(f'steady state found in {elapsed(t0)}')

def find_ss_SAM(model,do_print=False):
    """ find the steady state - SAM """

    par = model.par
    ss = model.ss
    
    # a. shocks
    ss.shock_TFP = 1.0

    # b. fixed
    ss.delta = par.delta_ss
    ss.lambda_u = par.lambda_u_ss
    ss.w = par.w_ss
    ss.theta = par.theta_ss
    ss.px = (par.epsilon_p-1)/par.epsilon_p

    # c. direct implications
    par.A = ss.lambda_u/ss.theta**(1-par.alpha)
    ss.lambda_v = par.A*ss.theta**(-par.alpha)

    # d. labor market dynamics
    ss.u = ss.delta*(1-ss.lambda_u)/(ss.lambda_u+ss.delta*(1-ss.lambda_u))
    ss.ut = ss.u/(1-ss.lambda_u)
    ss.vt = ss.ut*ss.theta
    ss.v = (1-ss.lambda_v)*ss.vt
    ss.entry = ss.vt-(1-ss.delta)*ss.v
    ss.S = ss.vt/ss.theta

    # e. job and vacancy bellmans
    ss.Vj = (ss.px*ss.shock_TFP-ss.w)/(1-par.beta*(1-ss.delta))
    par.kappa = ss.lambda_v*ss.Vj

    if do_print:
        print(f'{par.A = :6.4f}')
        print(f'{par.kappa = :6.4f}')
        print(f'{ss.w = :6.4f}')
        print(f'{ss.delta = :6.4f}')
        print(f'{ss.lambda_u = :6.4f}')
        print(f'{ss.lambda_v = :6.4f}')
        print(f'{ss.theta = :6.4f}')
        print(f'{ss.u = :6.4f}')
        print(f'{ss.ut = :6.4f}')
        print(f'{ss.S = :6.4f}')


def find_ss_HANK(model,do_print=False):
    """ find the steady state - HANK """

    par = model.par
    ss = model.ss

    # a. shocks
    pass

    # b. fixed
    ss.qB = par.qB_share_ss*ss.w
    ss.Pi = 1.0
    
    # c. equilibrium  
    ss.UI = par.phi*ss.w*ss.u
    ss.Yt_hh = ss.w*(1-ss.u) + ss.UI

    def asset_market_clearing(R):
        
        # o. set
        ss.RealR_ex_post = ss.RealR = R
        ss.q = 1/(ss.RealR-par.delta_q)
        ss.B = ss.qB/ss.q

        ss.tau = ((1+par.delta_q*ss.q)*ss.B+ss.UI-ss.q*ss.B)/ss.Yt_hh

        # oo. solve + simulate
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False)

        # ooo. difference
        ss.A_hh = np.sum(ss.a*ss.D)        
        diff = ss.qB - ss.A_hh
        
        return diff

    # i. initial values
    R_max = 1.0/par.beta
    R_min = R_max - 0.05
    R_guess = (R_min+R_max)/2

    diff = asset_market_clearing(R_guess)
    if do_print: print(f'guess:\n     R = {R_guess:12.8f} -> B-A_hh = {diff:12.8f}')

    # ii. find bracket
    if diff > 0:
        dR = R_max-R_guess
    else:
        dR = R_min-R_guess

    if do_print: print(f'find bracket to search in:')
    fac = 0.95
    it = 0
    max_iter = 50
    R = R_guess
    while True:
    
        oldR = R 
        olddiff = diff
        R = R_guess + dR*(1-fac)
        diff = asset_market_clearing(R)
        
        if do_print: print(f'{it:3d}: R = {R:12.8f} -> B-A_hh = {diff:12.8f}')

        if np.sign(diff)*np.sign(olddiff) < 0: 
            break
        else:
            fac *= 0.50 
            it += 1
            if it > max_iter: raise ValueError('could not find bracket')

    if oldR < R:
        a,b = oldR,R
        fa,fb = olddiff,diff
    else:
        a,b = R,oldR
        fa,fb = diff,olddiff                
    
    # iii. search
    if do_print: print(f'brentq:')
    
    brentq(asset_market_clearing,a,b,fa=fa,fb=fb,xtol=par.tol_R,rtol=par.tol_R,
        do_print=do_print,varname='R',funcname='B-A_hh')

    ss.C_hh = np.sum(ss.c*ss.D)

    # d. R
    ss.R = ss.RealR*ss.Pi

    if do_print:
        print(f'{ss.qB = :6.4f}')
        print(f'{ss.RealR = :6.4f}')