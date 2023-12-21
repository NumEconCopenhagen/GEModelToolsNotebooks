# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d

@nb.njit(parallel=True)
def solve_hh_backwards(par,z_trans,Z,ra,rl,vbeg_l_plus,vbeg_l,l,c,a,uce):
    """ solve backwards with vbeg_l from previous iteration """

    # unpack
    A_target = par.A_target
    ra_ss = par.r_ss_target

    # solve
    for i_fix in nb.prange(par.Nfix):
        for i_z in nb.prange(par.Nz):

            e = par.z_grid[i_z]  # productivity
            Ze = Z*e # labor income

            for i_a_lag in range(par.Na): # end-of-previous-period

                c_endo = (par.beta_grid[i_fix]*vbeg_l_plus[i_fix,i_z,:,i_a_lag])**(-1 / par.sigma)
                m_endo = c_endo + par.l_grid

                for i_l_lag in range(par.Nl): # end-of-period

                    a_lag = par.a_grid[i_a_lag]

                    # interpolation to fixed grid
                    d = ra_ss/(1+ra_ss)*(1+ra)*a_lag + par.chi*((1+ra)*a_lag-(1+ra_ss)*A_target)
                    m = (1+rl) * par.l_grid[i_l_lag] + Ze + d

                    # liquid assets
                    l[i_fix,i_z,i_l_lag,i_a_lag] = interp_1d(m_endo,par.l_grid,m)
                    l[i_fix,i_z,i_l_lag,i_a_lag] = np.fmax(l[i_fix,i_z,i_l_lag,i_a_lag],0.0)  # enforce borrowing constraint

                    # consumption
                    c[i_fix,i_z,i_l_lag,i_a_lag] = m-l[i_fix,i_z,i_l_lag,i_a_lag]
                    c[i_fix,i_z,i_l_lag,i_a_lag] = np.fmax(c[i_fix,i_z,i_l_lag,i_a_lag],0.0) # enforce non-negative consumption

                    # illiquid assets
                    a[i_fix,i_z,i_l_lag,i_a_lag] = (1+ra)*a_lag-d

                    # productivity weighted marg. util.
                    uce[i_fix,i_z,i_l_lag,i_a_lag] = e*c[i_fix,i_z,i_l_lag,i_a_lag]**(-par.sigma)

        # b. expectation step
        v_l_a = (1+rl)*c[i_fix]**(-par.sigma)

        for i_z_lag in nb.prange(par.Nz):
            for i_l_lag in nb.prange(par.Nl):
                for i_a_lag in nb.prange(par.Na):

                    vbeg_l[i_fix,i_z_lag,i_l_lag,i_a_lag] = 0.0

                    for i_z in range(par.Nz): # after realization
                        vbeg_l[i_fix,i_z_lag,i_l_lag,i_a_lag] += z_trans[i_fix,i_z_lag,i_z]*v_l_a[i_z,i_l_lag,i_a_lag]