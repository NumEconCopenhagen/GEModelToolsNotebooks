# find steady state

import numpy as np

def find_ss(model,do_print=False):
    """ find the steady state """

    par = model.par
    sol = model.sol
    sim = model.sim
    ss = model.ss

    # a. 
    ss.L = 1.0 # normalization
    ss.r = par.r_ss_target
    ss.w = par.w_ss_target

    assert (1+ss.r)*par.beta_mean < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)
    if do_print: print('')

    ss.K = ss.A = ss.A_hh = np.sum(sim.D*sol.a)
    
    ss.Z = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
    ss.rk = par.alpha*ss.Z*(ss.K/ss.L)**(par.alpha-1)
    par.delta = ss.rk - ss.r

    ss.Y = ss.Z*ss.K**par.alpha*ss.L**(1-par.alpha)
    ss.C = ss.Y - par.delta*ss.K
    ss.C_hh = np.sum(sim.D*sol.c)

    if do_print:

        print(f'Implied K = {ss.K:6.3f}')
        print(f'Implied Y = {ss.Y:6.3f}')
        print(f'Implied Z = {ss.Z:6.3f}')
        print(f'Implied K/Y = {ss.K/ss.Y:6.3f}')
        print(f'Implied delta = {par.delta:6.3f}')
        print(f'Discrepance in C-C_hh = {ss.C-ss.C_hh:12.8f}')
