import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem
import blocks

class HANKSAMModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['rh','wh',] # direct inputs
        self.inputs_hh_z = ['EU','UE'] # transition matrix inputs
        self.outputs_hh = ['a','c'] # outputs
        self.intertemps_hh = ['EVa'] # intertemporal variables

        # c. GE
        self.shocks = ['G','Z'] # exogenous shocks
        self.unknowns = ['Y','w'] # endogenous unknown
        self.targets = ['WPC','clearing_A'] # targets = 0
        
        # d. all variables
        self.varlist = [
            'A_hh','A','B','C_hh','C',
            'clearing_A','clearing_C',
            'd','EU','G','Z','i','N','NKPC','Pi_w','Pi',
            'r','rh','tau','U','UE','w','wh','WPC','Y',
        ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.fill_z_trans = household_problem.fill_z_trans
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. preferences
        par.sigma = 2.0 # CRRA cieffucuebt
        par.varphi = 0.5 # Frisch elasticity
        par.nu = np.nan # disutility of labor scale factor [determined in steady state]
        
        par.Nbeta = 2 # number of discount factor groups
        par.beta_mean = 0.94**(1/4) # discount factor mean, range is [mean-width,mean+width]
        par.beta_delta = 0.005 # discount factor width, range is [mean-width,mean+width]

        # c. income parameters
        par.rho_e = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of persistent shock in low risk state
        par.Ne = 7 # number of productivity states

        par.EU_target = 0.025 # E -> U prob.
        par.UE_target = 0.500 # U -> E prob.
        par.phi = 0.50 # unemployment benefit share

        par.U_residual = 'EU' # variable to take residual adjustment
        par.varepsilon_U = 0.0 # elasticity of U wrt. N 
        par.varepsilon_EU = 0.0 # elasticity of EU wrt. N (if U_residual == UE)
        par.varepsilon_UE = 0.0 # elasticity of EU wrt. N (if U_residual == EU)

        # d. price setting
        par.epsilon = 10.0 # market power, firms
        par.epsilon_w = 10.0 # market power, unions
        par.theta = 100.0 # Rotemberg costs, price setting
        par.theta_w = 500.0 # Rotemberg costs, wage setting

        # e. government
        par.varepsilon_pi = 1.5 # taylor parameter
        par.tau_target = 0.25 # tax rate

        par.varepsilon_B = 2.0 # elasticity of the tax rate wrt. B/Y
        par.t_B = 40 # initial period with potential tax change
        par.Delta_B = 150 # interval for phasing in tax change

        # f. calibration
        par.r_ss_target = 1.02**(1/4)-1.0 # target for real interest rate
           
        # g. grids
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # h. shocks
        par.jump_Z = 0.0 # initial jump in %
        par.jump_G = 0.01
        par.rho_Z = 0.00 # AR(1) coefficeint
        par.rho_G = 0.75

        # i. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-8 # tolerance when solving eq. system
        
        par.simT = 2_000 # length of simulation 

    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.Nfix = par.Nbeta
        par.beta_grid = np.zeros(par.Nbeta)
        
        par.e_grid = np.zeros(par.Ne)
        par.Nz = 2*par.Ne # double up due to unemployment

        # for easier selection later on
        par.u_grid = np.hstack([np.zeros(par.Ne,dtype=np.int_),np.ones(par.Ne,dtype=np.int_)])
                    
        # b. solution
        sol_shape = (par.Nbeta,par.Nz,par.Na)
        self.allocate_GE(sol_shape)

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss