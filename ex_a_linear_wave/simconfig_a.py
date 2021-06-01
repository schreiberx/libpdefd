#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd
import argparse

from libpdefd.array_matrix.libpdefd_array import *



class SimConfig:
    
    def __init__(self, num_dims = 1):

        """
        Number of dimensions
        """
        self.num_dims = num_dims
        
        """
        Resolution of simulation in number of cells
        """
        base_res = 256//(2**num_dims)
        #base_res = 16//(2**num_dims)
        self.cell_res = np.array([int(base_res*(1+0.2*(i+1))) for i in range(num_dims)])
        
        """
        Use grid staggering
        """
        self.use_staggering = True
        
        """
        Domain start/end coordinate
        """
        self.domain_start = np.array([0 for _ in range(self.num_dims)])
        self.domain_end = np.array([320//(1+0.5*(i+1)) for i in range(self.num_dims)])
        self.domain_size = self.domain_end - self.domain_start
        
        """
        Boundary condition: 'periodic', 'dirichlet0' or 'neumann0', 'symmetric'
        """
        #self.boundary_conditions_rho[0] = "symmetric"
        #self.boundary_conditions_vel[0] = "dirichlet0"
        self.boundary_conditions_rho = ["symmetric" for _ in range(self.num_dims)]
        self.boundary_conditions_vel = ["dirichlet0" for _ in range(self.num_dims)]
        
        
        """
        Center of initial condition (relative to domain)
        """
        self.initial_condition_center = np.array([0.25 for i in range(self.num_dims)])
        
        """
        GridInfo1D layout: 'auto' or 'manual'
        """
        self.grid_setup = "auto"
        
        """
        Minimum order of spatial approximation
        """
        self.min_spatial_approx_order = 2
        
        
        """
        Use symlog for plot
        """
        self.use_symlog = False
        if self.boundary_conditions_rho[0][0] == "neumann0" or self.boundary_conditions_vel[0] == "neumann0":
            self.use_symlog = True
            self.use_symlog = False
        
        
        """
        Output frequency of output
        """
        self.output_freq = 10
        
        
        """
        Visualization dimensions for 2D plots
        """
        self.vis_dim_x = 0
        self.vis_dim_y = 1
        
        
        """
        Slice to extract if the dimension is not visualized
        """
        self.vis_slice  = [self.cell_res[i]//2 for i in range(self.num_dims)]
        
        
        """
        Number of time steps
        """
        self.num_timesteps = 10000
        
        
        """
        Test run
        """
        self.test_run = False
        
        self.parseargs()
        self.setup()
        
        
    def parseargs(self):
    
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-freq', dest="output_freq", type=int, help="Output frequency")
        parser.add_argument('--num-timesteps', dest="num_timesteps", type=int, help="Number of time steps")
        parser.add_argument('--test-run', dest="test_run", type=str, help="Test run")
        
        args = parser.parse_args()
        dict_args = vars(args)
        for key in dict_args:
            value = dict_args[key]
            if value != None:
                if isinstance(dict_args[key], list):
                    value = np.array(value)
        
                self.__setattr__(key, value)


    def setup(self):
        
        """
        Guess time step size
        """
        
        self.dt = np.min(self.domain_size/(self.cell_res+1))
        self.dt *= 0.5
        
        def str2bool(v):
            """
            https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
            """
            if isinstance(v, bool):
                return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
        
        self.test_run = str2bool(self.test_run)
        
        if self.test_run:
            self.num_timesteps = 10
        
