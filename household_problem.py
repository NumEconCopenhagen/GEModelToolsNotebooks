
# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,r,z_trans_plus,Va_plus,Va,a,c,m):
    """ solve backwards with Va_plus from previous iteration """

    # a. post-decision
    marg_u_plus = np.zeros((par.Nbeta,par.Nz,par.Na))
    for i_beta in nb.prange(par.Nbeta):
        marg_u_plus[i_beta] = (par.beta_grid[i_beta]*z_trans_plus)@Va_plus[i_beta]

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

    # called by .solve_hh_ss(), which also prepares simulation
    # by calculating sol.i and sol.w
    
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
        solve_hh_backwards(par,ss.r,par.z_trans_ss,sol.Va,sol.Va,sol.a,sol.c,sol.m)

        # ii. check
        if np.max(np.abs(sol.a-a_old)) < par.tol_solve: 
            return it
        
        # iv. increment
        it += 1
        if it > par.max_iter_solve: 
            raise ValueError('solve_hh_ss(), too many iterations')

@nb.njit
def solve_hh_path(par,sol,ss,path):
    """ solve household problem along the transition path """

    # called by .solve_hh_path(), which also prepares for simulation
    # by calculating sol.path_i and sol.path_w

    # a. construct transition paths
    for t in range(par.transition_T):
        par.z_trans_path[t] = par.z_trans_ss # could be a function period t variables

    # b. solve Bellman equations backwards along transition path
    for k in range(par.transition_T):

        t = (par.transition_T-1)-k

        # i. next-period
        if t == par.transition_T-1:
            Va_plus = sol.Va
            z_trans_plus = par.z_trans_ss
        else:
            Va_plus = sol.path_Va[t+1]
            z_trans_plus = par.z_trans_path[t+1]

        # ii. solve       
        y = path.w[0,t]*par.z_grid_ss
        for i_beta in range(par.Nbeta):
            for i_z in range(par.Nz):
                sol.path_m[t,i_beta,i_z] = (1+path.r[0,t])*par.a_grid + y[i_z]

        # iii. time iteration
        solve_hh_backwards(par,path.r[0,t],z_trans_plus,Va_plus,sol.path_Va[t],sol.path_a[t],sol.path_c[t],sol.path_m[t])

    return k
