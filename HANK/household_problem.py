# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit        
def solve_hh_backwards(par,z_trans,w,r,d,tau,vbeg_a_plus,vbeg_a,a,c,ne):
    """ solve backwards with vbeg_a_plus from previous iteration """
    
    for i_fix in range(par.Nfix):
        
        # a. solution step
        for i_z in range(par.Nz):

            # i. prepare
            e = par.z_grid[i_z]
            T = d*e - tau*e
            fac = (w*e/par.varphi)**(1/par.nu)

            # ii. use focs
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z,:])**(-1/par.sigma)
            ne_endo = fac*(par.beta*vbeg_a_plus[i_fix,i_z,:])**(1/par.nu)*e

            # iii. re-interpolate
            f_endo = c_endo + par.a_grid - w*ne_endo - T
            f_exo = (1+r)*par.a_grid

            interp_1d_vec(f_endo,c_endo,f_exo,c[i_fix,i_z,:])
            interp_1d_vec(f_endo,ne_endo,f_exo,ne[i_fix,i_z,:])

            # iv. saving
            a[i_fix,i_z,:] = f_exo + w*ne[i_fix,i_z,:] + T - c[i_fix,i_z,:]

            # v. refinement at constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < par.a_min:
                    
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a] = par.a_min

                    # ii. solve foc for n
                    ni = ne[i_fix,i_z,i_a]/e
                    for i in range(30):

                        ci = (1+r)*par.a_grid[i_a] + w*e*ni + T - par.a_min # from binding constraint

                        error = ni - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < 1e-11:
                            break
                        else:
                            dc_dn = w*e
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*dc_dn
                            ni = ni - error/derror
                    else:
                        
                        raise ValueError('solution could not be found')

                    # iii. save
                    c[i_fix,i_z,i_a] = ci
                    ne[i_fix,i_z,i_a] = ni*e
                    
                else:

                    break

        # b. expectation step
        va = c[i_fix,:,:]**(-par.sigma)
        vbeg_a[i_fix] = (1+r)*z_trans[i_fix]@va