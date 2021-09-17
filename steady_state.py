# find steady state

import numpy as np

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    sol = model.sol
    sim = model.sim
    ss = model.ss

    # a. exogenous and targets
    ss.L = 1.0 # normalization
    ss.r = par.r_ss_target
    ss.w = par.w_ss_target

    assert (1+ss.r)*par.beta_mean < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    # b. stock and capital stock from household behavior
    model.solve_hh_ss(do_print=do_print) # give us sol.a and sol.c (steady state policy functions)
    model.simulate_hh_ss(do_print=do_print) # give us sim.D (steady state distribution)
    if do_print: print('')

    ss.K = ss.A = ss.A_hh = np.sum(sim.D*sol.a)
    
    # c. back technology and depreciation rate
    ss.Z = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
    ss.rk = par.alpha*ss.Z*(ss.K/ss.L)**(par.alpha-1)
    par.delta = ss.rk - ss.r

    # d. remaining
    ss.Y = ss.Z*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.C = ss.Y - par.delta*ss.K
    ss.C_hh = np.sum(sim.D*sol.c)

    # e. print
    if do_print:

        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Z = {ss.Z:6.3f}')
        print(f'Implied delta = {par.delta:6.3f}') # check is positive
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
        print(f'Discrepancy in A-A_hh = {ss.A-ss.A_hh:12.8f}') # = 0 by construction
        print(f'Discrepancy in C-C_hh = {ss.C-ss.C_hh:12.8f}') # != 0 due to numerical error 

    # remember: we should at some point have called
    #   model.solve_hh_ss(do_print=do_print)
    #   model.simulate_hh_ss(do_print=do_print)