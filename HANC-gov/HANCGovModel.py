import numpy as np

from EconModel import EconModelClass
from GEModelTools import GEModelClass

import steady_state
import household_problem

class HANCGovModelClass(EconModelClass,GEModelClass):    

    def settings(self):
        """ fundamental settings """

        # a. namespaces (typically not changed)
        self.namespaces = ['par','ini','ss','path','sim']

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['pB','tau'] # direct inputs
        self.inputs_hh_z = [] # transition matrix input
        self.outputs_hh = ['a','c','u'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['G','eta'] # exogenous shocks
        self.unknowns = ['pB'] # endogenous unknowns
        self.targets = ['clearing_B'] # targets = 0

        # d. all variables
        self.blocks = [ # list of strings to block-functions
            'blocks.government',
            'hh', # household block
            'blocks.market_clearing']

        # d. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix = 1 # number of fixed discrete states (none here)
        par.Nz = 7 # number of stochastic discrete states (here productivity)

        # a. preferences
        par.beta = 0.96 # discount factor
        par.sigma = 2.0 # CRRA coefficient

        # b. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.30*(1.0-par.rho_z**2.0)**0.5 # std. of persistent shock

        # c. government
        par.G_ss = 0.10 # government spending
        par.phi = 0.10 # adjustment of tax rate to high debt

        # f. grids         
        par.a_max = 500.0 # maximum point in grid for a
        par.Na = 300 # number of grid points

        # g. misc.
        par.max_iter_solve = 50_000 # maximum number of iterations when solving household problem
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating household problem
        
        par.tol_solve = 1e-12 # tolerance when solving household problem
        par.tol_simulate = 1e-12 # tolerance when simulating household problem

        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

        par.py_hh = False # call solve_hh_backwards in Python-mode
        par.py_blocks = False # call blocks in Python-mode

    def allocate(self):
        """ allocate model """

        self.allocate_GE() # should always be called here

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss

    def v_ss(self):
        """ social welfare in steady state """

        par = self.par
        ss = self.ss
        
        return np.sum([par.beta**t*ss.U_hh for t in range(par.T)])

    def v_path(self):
        """ social welfare in transition path """

        par = self.par
        path = self.path

        return np.sum([par.beta**t*path.U_hh[t] for t in range(par.T)])