
import numpy as np
import numba as nb

from EconModel import EconModelClass
from GEModelTools import GEModelClass, lag, lead

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.linear_interp import interp_1d_vec

class AiygariModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','sim','ss','path']
        
        # b. household
        self.grids_hh = ['a'] # grids in household problem
        self.pols_hh = ['a'] # household policy functions
        self.inputs_hh = ['r','w'] # inputs to household problem
        self.outputs_hh = ['a','c'] # outputs of household problem
        self.intertemps_hh = ['Va'] # intertemporal variables in household problem

        # c. GE
        self.shocks = ['Gamma'] # exogenous inputs
        self.unknowns = ['K'] # endogenous inputs
        self.targets = ['clearing_A'] # targets

        # d. all variables
        self.varlist = [
            'A_hh',
            'C_hh',
            'C',
            'clearing_A',
            'clearing_C',
            'Gamma',
            'K',
            'L',
            'r',
            'rk',
            'w',
            'Y',
        ]

        # functions
        self.solve_hh_backwards = solve_hh_backwards
        self.block_pre = block_pre
        self.block_post = block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 # number of fixed discrete states (here discount factor)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. macros
        pass

        # b. preferences
        par.sigma = 2.0 # CRRA coefficient
        par.beta_mean = 0.9875 # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = 0.00000 # discount factor, width, range is [mean-width,mean+width]
        par.Nbeta = 1 # discount factor, number of states

        # c. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # d. production and investment
        par.alpha = 0.36 # cobb-douglas
        par.delta = np.nan # depreciation [determined in ss]

        # e. calibration
        par.r_ss_target = 0.01 # target for real interest rate
        par.w_ss_target = 1.0 # target for real wage

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. shocks
        par.jump_Gamma = -0.01 # initial jump
        par.rho_Gamma = 0.8 # AR(1) coefficient
        par.std_Gamma = 0.01 # std. of innovation

        # h. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.beta_grid = np.zeros(par.Nbeta)
        
        # b. solution
        self.allocate_GE() # should always be called here

    def create_grids(self):
        """ create grids """

        # note: only fills out already allocated arrays

        # model specific: par.beta_grid
        # always:
        #   par.endo1_grid
        #   par.z_grid_ss (par.Nz,) # grid values
        #   par.z_trans_ss (par.Nz,par.Nz) # transition matrix
        #   par.z_ergodic_ss (par.Nz,) # ergodic distribution
        #   par.z_grid_path (par.transition,par.Nz) # grid along transition path
        #   par.z_transition_path (par.transition,par.Nz) # transition matrix along transition path for t to t+1

        par = self.par
        ss = self.ss

        # a. beta
        par.beta_grid[:] = np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nbeta)

        # b. a
        par.a_grid[:] = equilogspace(0.0,ss.w*par.a_max,par.Na)
        
        # c. z - steady state
        par.z_grid_ss[:],par.z_trans_ss[:,:],par.z_ergodic_ss[:],_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,par.Nz)

        # d. initial guess
        y = ss.w*par.z_grid_ss

        m = np.zeros((par.Nbeta,par.Nz,par.Na))
        for i_beta in range(par.Nbeta):
            for i_z in range(par.Nz):
                m[i_beta,i_z,:] = (1+ss.r)*par.a_grid + y[i_z]

        a = 0.90*m # pure guess
        c = m - a
        
        ss.Va[:,:,:] = (1+ss.r)*c**(-par.sigma)

    def find_ss(self,do_print=False):
        """ find the steady state """

        par = self.par
        ss = self.ss

        # a. exogenous and targets
        ss.L = 1.0 # normalization
        ss.r = par.r_ss_target
        ss.w = par.w_ss_target

        assert (1+ss.r)*par.beta_mean < 1.0, '(1+r)*beta < 1, otherwise problems might arise'

        # b. stock and capital stock from household behavior
        self.solve_hh_ss(do_print=do_print) # give us sol.a and sol.c (steady state policy functions)
        self.simulate_hh_ss(do_print=do_print) # give us sim.D (steady state distribution)
        if do_print: print('')

        ss.K = ss.A_hh = np.sum(ss.D*ss.a)
        
        # c. back technology and depreciation rate
        ss.Gamma = ss.w / ((1-par.alpha)*(ss.K/ss.L)**par.alpha)
        ss.rk = par.alpha*ss.Gamma*(ss.K/ss.L)**(par.alpha-1)
        par.delta = ss.rk - ss.r

        # d. remaining
        ss.Y = ss.Gamma*ss.K**par.alpha*ss.L**(1-par.alpha)
        ss.C = ss.Y - par.delta*ss.K
        ss.C_hh = np.sum(ss.D*ss.c)

        # e. print
        if do_print:

            print(f'Implied K = {ss.K:6.3f}')
            print(f'Implied Y = {ss.Y:6.3f}')
            print(f'Implied Gamma = {ss.Gamma:6.3f}')
            print(f'Implied delta = {par.delta:6.3f}') # check is positive
            print(f'Implied K/Y = {ss.K/ss.Y:6.3f}') 
            print(f'Discrepancy in K-A_hh = {ss.K-ss.A_hh:12.8f}') # = 0 by construction
            print(f'Discrepancy in C-C_hh = {ss.C-ss.C_hh:12.8f}') # != 0 due to numerical error 

    def create_grids_path(self):
        """ create grids for solving backwards along the transition path """

        # note: can use the full path of all inputs to the household block

        par = self.par
        path = self.path

        for t in range(par.T):
            par.z_grid_path[t,:] = par.z_grid_ss
            par.z_trans_path[t,:,:] = par.z_trans_ss

#####################
# household problem #
#####################

@nb.njit(parallel=True)        
def solve_hh_backwards(par,r,w,z_grid,z_trans_plus,Va_plus,Va,a,c):
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
            m = (1+r)*par.a_grid + w*z_grid[i_z]
            interp_1d_vec(m_endo,par.a_grid,m,a[i_beta,i_z])
            a[i_beta,i_z,:] = np.fmax(a[i_beta,i_z,:],0.0) # enforce borrowing constraint
            c[i_beta,i_z] = m-a[i_beta,i_z]

            # iii. envelope condition
            Va[i_beta,i_z] = (1+r)*c[i_beta,i_z]**(-par.sigma)

################
# other blocks #
################

@nb.njit
def block_pre(par,ss,path,ncols=1):
    """ evaluate transition path - before household block """

    # par, sol, sim, ss, path are namespaces
    # ncols specifies have versions of the model to evaluate at once
    #   path.VARNAME have shape=(len(unknowns)*par.T,par.T)
    #   path.VARNAME[0,t] for t in [0,1,...mpar.T] is always used outside of this function

    for thread in nb.prange(ncols):
        
        # unpack
        Gamma = path.Gamma[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]

        rk = path.rk[thread,:]
        r = path.r[thread,:]
        w = path.w[thread,:]
        
        Y = path.Y[thread,:]
        C = path.C[thread,:]

        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]

        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]

        #################
        # implied paths #
        #################

        # lags and leads of unknowns and shocks
        K_lag = lag(ss.K,K) # copy, same as [ss.K,K[0],K[1],...,K[-2]]
        
        # example: K_lead = lead(K,ss.K) # copy, same as [K[1],K[1],...,K[-1],ss.K]

        # VARNAME is used for reading values
        # VARNAME[:] is used for writing in-place

        # a. exogenous
        L[:] = 1.0

        # b. implied prices (remember K is input -> K_lag is known)
        rk[:] = par.alpha*Gamma*(K_lag/L)**(par.alpha-1.0)
        r[:] = rk-par.delta
        w[:] = (1.0-par.alpha)*Gamma*(rk/(par.alpha*Gamma))**(par.alpha/(par.alpha-1.0))

        # c. production and consumption
        Y[:] = Gamma*K_lag**(par.alpha)*L**(1-par.alpha)
        C[:] = Y-(K-K_lag)-par.delta*K_lag

@nb.njit
def block_post(par,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    # par, sol, sim, ss, path are namespaces
    # ncols specifies have many versions of the model to evaluate at once
    #   path.VARNAME have shape=(len(unknowns)*par.T,par.T)
    #   path.VARNAME[0,t] for t in [0,1,...mpar.T] is always used outside of this function

    for thread in nb.prange(ncols):

        # unpack
        Gamma = path.Gamma[thread,:]
        K = path.K[thread,:]
        L = path.L[thread,:]

        rk = path.rk[thread,:]
        r = path.r[thread,:]
        w = path.w[thread,:]
        
        Y = path.Y[thread,:]
        C = path.C[thread,:]

        A_hh = path.A_hh[thread,:]
        C_hh = path.C_hh[thread,:]

        clearing_A = path.clearing_A[thread,:]
        clearing_C = path.clearing_C[thread,:]

        ###########
        # targets #
        ###########

        clearing_A[:] = K-A_hh
        clearing_C[:] = C-C_hh            