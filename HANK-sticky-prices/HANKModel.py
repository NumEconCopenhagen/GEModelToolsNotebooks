
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class HANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['w','r','d','tau'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','ell','n'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['Gamma','istar','G'] # exogenous inputs
        self.unknowns = ['Y','w','pi'] # endogenous inputs
        self.targets = ['NKPC_res','clearing_N','clearing_A'] # targets
        self.blocks = [
            'blocks.production',
            'blocks.taylor',
            'blocks.fisher',
            'blocks.government',
            'blocks.intermediary_goods',
            'hh',
            'blocks.market_clearing']
        
        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.RA = False # representative agent
        par.Nfix = 1
        par.r_target_ss = 0.01 # target real interest rate in steady state 
        # note: high, but else might need more periods with RANK

        # a. preferences
        par.beta = 0.98 # discount factor (guess, calibrated in ss)
        par.varphi = 0.80 # disutility of labor (guess, calibrated in ss)

        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution
        par.nu = 1.0 # inverse Frisch elasticity

        par.beta_RA = np.nan # discount factor for representative agent
        par.varphi_RA = np.nan # disutility for representative agent

        # c. income parameters
        par.rho_z = 0.965 # AR(1) parameter
        par.sigma_psi = np.sqrt(0.50**2*(1-par.rho_z**2)) # std. of psi
        par.Nz = 7 # number of productivity states

        # d. price setting
        par.mu = 1.2 # mark-up
        par.kappa = 0.05 # slope of Phillips curve

        # e. government
        par.phi = 1.5 # Taylor rule coefficient on inflation
        par.phi_y = 0.0 # Taylor rule coefficient on output
        
        par.G_target_ss = 0.0 # government spending
        par.B_target_ss = 4.0 # bond supply

        # f. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 150.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. shocks
        par.jump_istar = -0.0025 # initial jump
        par.rho_istar = 0.60 # AR(1) coefficeint
        par.std_istar = 0.0025 # std.

        par.jump_G = 0.01
        par.rho_G = 0.90
        par.std_G = 0.000

        par.jump_Gamma = 0.01 
        par.rho_Gamma = 0.90 
        par.std_Gamma = 0.00

        # h. misc.
        par.T = 1000 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system
        
        par.py_hh = False # call solve_hh_backwards in Python-mode
        par.py_blocks = False # call blocks in Python-mode
        par.full_z_trans = False # let z_trans vary over endogenous states

    def allocate(self):
        """ allocate model """

        par = self.par

        # b. solution
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        

# Reperesentative agent model
class RANKModelClass(HANKModelClass):

    # same settings
    # same allocate 

    def setup(self):

        super().setup() # calls setup from HANKModelClass
        self.par.RA = True

    # note: the heterogenous households are still there, but does not matter for transition path
    # this is a simple though not computationally efficient implementation