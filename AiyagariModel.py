
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import GEModelTools

class AiygariModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','sol','sim','ss','path','jac_hh']
        
        # typically constant accross models 

        # b. other attributes (to save them)
        self.other_attrs = [
            'grids_hh','pols_hh','inputs_hh','varlist_hh'
            'inputs_exo','inputs_endo','targets','varlist','jac']

        # used when copying and saving the model

        # household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions

        self.inputs_hh = ['r','w'] # inputs to household problem
        self.outputs_hh = ['a','c'] # output of household problem

        self.varlist_hh = ['a','m','c','Va'] # variables in household problem

        # GE
        self.inputs_exo = ['Z'] # exogenous inputs
        self.inputs_endo = ['K'] # endogenous inputs

        self.targets = ['clearing_A'] # targets
        
        self.varlist = [ # all variables
            'A_hh',
            'A',
            'C_hh',
            'C',
            'clearing_A',
            'clearing_C',
            'K',
            'L',
            'r',
            'rk',
            'w',
            'Y',
            'Z',
        ]

        # c. folder to save in
        self.savefolder = 'saved'
        
        # d. list not-floats in namespaces for safe type inference
        self.not_floats = ['Nbeta']

    def setup(self):
        """ set baseline parameters """

        par = self.par

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
        par.Nz = 7 # number of productivity states

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
        par.jump_Z = -0.01 # initial jump in %
        par.rho_Z = 0.8 # AR(1) coefficient

        # h. misc.
        par.transition_T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-10 # tolerance when solving
        par.tol_simulate = 1e-10 # tolerance when simulating
        par.tol_broyden = 1e-8 # tolerance when solving eq. system
        
    def allocate(self):
        """ allocate model """

        par = self.par

        # a. grids
        par.beta_grid = np.zeros(par.Nbeta)
        
        # b. solution
        sol_shape = (par.Nbeta,par.Nz,par.Na) # (Nfix,Nz,Nendo1)
        self.allocate_GE(sol_shape) # should always be called here