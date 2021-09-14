
# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec
from GEModelTools import find_i_and_w_1d_1d

@nb.njit(parallel=True)        
def solve_hh_backwards(par,r,Va_p,Va,a,c,m):
    """ solve backwards with Va_p from previous iteration """

    # a. post-decision
    marg_u_plus = np.zeros((par.Nbeta,par.Nz,par.Na))
    for i_beta in nb.prange(par.Nbeta):
        marg_u_plus[i_beta] = (par.beta_grid[i_beta]*par.z_trans_ss)@Va_p[i_beta]

    # b. EGM loop
    for i_z in nb.prange(par.Nz):
        for i_beta in nb.prange(par.Nbeta):
        
            # i. EGM
            c_endo = marg_u_plus[i_beta,i_z]**(-1/par.sigma)
            m_endo = c_endo + par.a_grid
            
            # ii. interpolation
            interp_1d_vec(m_endo,par.a_grid,m[i_beta,i_z],a[i_beta,i_z])
            a[i_beta,i_z,:] = np.fmax(a[i_beta,i_z,:],0.0) # enforce borrowing constraint
            c[i_beta,i_z] = m[i_beta,i_z]-a[i_beta,i_z]

            # iii. envelope condition
            Va[i_beta,i_z] = (1+r)*c[i_beta,i_z]**(-par.sigma)

@nb.njit
def solve_hh_ss(par,sol,ss):
    """ solve household problem in steady state """

    it = 0

    # a. construct grid for m 
    y = ss.w*par.z_grid_ss
    for i_beta in range(par.Nbeta):
        for i_z in range(par.Nz):
            sol.m[i_beta,i_z,:] = (1+ss.r)*par.a_grid + y[i_z]

    # b. initial guess
    sol.a[:,:,:] = 0.90*sol.m # pure guess
    sol.c[:,:,:] = sol.m - sol.a 
    sol.Va[:,:,:] = (1+ss.r)*sol.c**(-par.sigma)

    # c. iterate
    while True:

        # i. save
        a_old = sol.a.copy()

        # ii. egm
        solve_hh_backwards(par,ss.r,sol.Va,sol.Va,sol.a,sol.c,sol.m)

        # ii. check
        if np.max(np.abs(sol.a-a_old)) < par.tol_solve: 
            return it
        
        # iv. increment
        it += 1
        if it > par.max_iter_solve: 
            raise ValueError('solve_hh_ss(), too many iterations')

@nb.njit
def solve_hh_path(par,sol,path):
    """ solve household problem along the transition path """

    # solve Bellman equations backwards along transition path
    for k in range(par.transition_T):

        t = (par.transition_T-1)-k

        # i. next-period
        if t == par.transition_T-1:
            Va_p = sol.Va
        else:
            Va_p = sol.path_Va[t+1]

        # ii. solve       
        y = path.w[t]*par.z_grid_ss
        for i_beta in range(par.Nbeta):
            for i_z in range(par.Nz):
                sol.path_m[t,i_beta,i_z] = (1+path.r[t])*par.a_grid + y[i_z]

        # iii. time iteration
        solve_hh_backwards(par,path.r[t],Va_p,sol.path_Va[t],sol.path_a[t],sol.path_c[t],sol.path_m[t])

        # iv. find indices and weights
        find_i_and_w_1d_1d(sol.path_a[t],par.a_grid,sol.path_i[t],sol.path_w[t])

    return k
