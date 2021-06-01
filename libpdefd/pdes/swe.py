#! /usr/bin/env python3
"""
Shallow-Water Equation
"""

import sys
import numpy as np
import libpdefd
import argparse



class SimConfig:
    def __init__(self):
        self.init_phase = True

        parser = argparse.ArgumentParser()
        
        """
        Setup default values
        """
        
        """
        Sim settings
        """
        self.sim_g = 9.81        # Gravity
        self.sim_h0 = 1e3       # Average water height
        self.sim_hpert_scaling = 100    # Scaling of perturbation
        self.sim_f = 1e-3        # Coriolis effect
        
        self.sim_visc = 0       # Viscosity on velocities
        self.sim_visc_order = 2
        
        self.dt_scaling = 1.0
        
        
        """
        Visualization Variable: hpert, hpert_diff_hpert0, u, v, vort
        """
        self.vis_variable = "vort"
        
        
        """
        SWE type:
            linear_c_grid
            linear_a_grid
            nonlinear_c_grid
            nonlinear_a_grid
        """
        self.swe_type = "linear_a_grid"
        
        
        """
        Minimum order of spatial approximation
        """
        self.min_spatial_approx_order = 2
        
        
        """
        Different variants for nonlinear discretization.
        Choose '1' or '2'
        Variant 1 seems to be more stable with non-periodic boundary conditions
        """
        self.swe_nonlinear_discretization = 1
        
        
        """
        Resolution of simulation in number of cells
        """
        self.base_res = 128
        self.cell_res = np.array([self.base_res for i in range(2)])
        
        
        """
        Center of initial condition (relative to domain)
        """
        self.initial_condition_default_center = np.array([0.25 for i in range(2)])
        
        
        """
        Visualization dimensions for 2D plots
        """
        self.vis_dim_x = 0
        self.vis_dim_y = 1
        
        
        """
        Domain start/end coordinate
        """
        self.domain_start = np.array([0 for _ in range(2)])
        self.domain_end = np.array([np.pi*2.0 for i in range(2)])
        
        self.sim_domain_aspect = 1
        
        """
        Boundary condition: 'periodic', 'dirichlet' or 'neumann'
        """
        self.boundary_conditions_hpert = ["periodic" for _ in range(2)]
        self.boundary_conditions_u = ["periodic" for _ in range(2)]
        self.boundary_conditions_v = ["periodic" for _ in range(2)]
        self.boundary_conditions_q = ["periodic" for _ in range(2)]
        
        """
        Grid setup: 'auto' or 'manual'
        """
        self.grid_setup = "auto"

        self.boundary_dirichlet_values = {
            'hpert': [-self.sim_hpert_scaling, self.sim_hpert_scaling],
            'u': [0, 0],
            'v': [0, 0],
        }

        """
        Variables initialized by update()
        """
        self.domain_size = None
        self.vis_slice = None

        
        """
        Output frequency of output
        """
        self.output_freq = 10
        
        
        """
        Number of time steps
        """
        self.num_timesteps = 10000
        
        
        """
        Test run
        """
        self.test_run = False
        parser.add_argument('--test-run', dest="test_run", type=str, help="Test run")

        """
        Update class members with program parameters
        """
        args = parser.parse_args()
        dict_args = vars(args)
        for key in dict_args:
            value = dict_args[key]
            if value != None:
                if isinstance(dict_args[key], list):
                    value = np.array(value)

                self.__setattr__(key, value)
        

        
        """
        Activate variable guard to avoid setting variables which don't exist
        """     
        self.init_phase = False

    def __setattr__(self, name, value):

        if name != 'init_phase':
            if not self.init_phase:
                if not name in self.__dict__:
                    raise Exception("Attribute '"+name+"' does not exist!")

        self.__dict__[name] = value

    
        
    def update(self):
        self.domain_size = self.domain_end - self.domain_start
        
        """
        Slice to extract if the dimension is not visualized
        """
        if self.vis_slice == None:
            self.vis_slice  = [self.cell_res[i]//2 for i in range(2)]

        libpdefd.core.operator.set_default_param('operator_diff__min_approx_order', self.min_spatial_approx_order)

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
        

"""
Differential operators
"""
class SWEBase:
    def __init__(self, simconfig):
        self.simconfig = simconfig
    
    
    """
    Setup boundaries
    """
    def get_boundaries(
            self,
            boundary_condition : str,
            dim_id : int,
            variable_id : str
    ):
        
        # Use periodic boundaries
        if boundary_condition == "periodic":
            boundary_left = libpdefd.BoundaryPeriodic()
            boundary_right = libpdefd.BoundaryPeriodic()
            
        elif boundary_condition == "dirichlet":
            self.simconfig.boundary_dirichlet_values['hpert'][0]
            if self.simconfig.benchmark_name in ["geostrophic_balance", "geostrophic_balance_symmetric"]:
                if variable_id == "hpert" and dim_id == 1:
                    boundary_left = libpdefd.BoundaryDirichlet(-1*self.simconfig.sim_hpert_scaling)
                    boundary_right = libpdefd.BoundaryDirichlet(1*self.simconfig.sim_hpert_scaling)
                    
                elif variable_id == "u" and dim_id == 1:
                    boundary_left = libpdefd.BoundaryDirichlet(0)
                    boundary_right = libpdefd.BoundaryDirichlet(0)
                else:
                    boundary_left = libpdefd.BoundaryDirichlet(0)
                    boundary_right = libpdefd.BoundaryDirichlet(0)
            else:
                boundary_left = libpdefd.BoundaryDirichlet(0)
                boundary_right = libpdefd.BoundaryDirichlet(0)
            
        elif boundary_condition == "neumann":
            boundary_left = libpdefd.BoundaryNeumannExtrapolated(0)
            boundary_right = libpdefd.BoundaryNeumannExtrapolated(0)
            
        else:
            raise Exception("Boundary condition '"+boundary_condition+"' is not supported")
    
        boundaries = [boundary_left, boundary_right]
        return boundaries
    

    """
    Setup grid
    """
    def _setup_grid_info(
            self,
            grid_nd : list,         # update 1D grid info
            staggered_dim : int,    # Dimension in which to apply staggering
            boundary_conditions : list, # List of boundary conditions
            variable_id : str       # Variable ID as a string (u, v)
    ):
        for i in range(2):
            boundaries = self.get_boundaries(boundary_conditions[i], i, variable_id)
        
            """
            Setup grids for each variable
            """
            if self.simconfig.grid_setup == "auto":
                if staggered_dim == 666:
                    staggered = True
                else:
                    staggered = (i==staggered_dim)
                grid_nd[i].setup_autogrid(self.simconfig.domain_start[i], self.simconfig.domain_end[i], self.simconfig.cell_res[i]+1, boundaries=boundaries, staggered=staggered)
            
            elif self.simconfig.grid_setup == "manual":
                if "_c_" in self.simconfig.swe_type:
                    raise Exception("TODO: Implement this here, but it's supported in libpdefd")
            
                x = np.linspace(0, 1, self.simconfig.cell_res[1]+1, endpoint=True)
                
                x = np.tanh(x*2.0-1.0)
                x /= np.abs(x[0])
                
                x = x*0.5+0.5
                
                x *= self.simconfig.domain_size
                x += self.simconfig.domain_start
                
                grid_nd[i].setup_manualgrid(x, boundaries=boundaries)
            
            else:
                raise Exception("GridInfo1D setup '"+self.simconfig.grid_setup+"' not supported")
    
    
    def setup(
            self
    ):
        if self.simconfig.swe_type in ["linear_a_grid", "nonlinear_a_grid"]:
            staggered_dim_hpert = -1
            staggered_dim_u = -1
            staggered_dim_v = -1
            staggered_dim_q = -1
            
        elif self.simconfig.swe_type in ["linear_c_grid", "nonlinear_c_grid"]:
            staggered_dim_hpert = -1
            staggered_dim_u = 0
            staggered_dim_v = 1
            staggered_dim_q = 666
            
        else:
            raise Exception("Unknown grid staggering")
        
        self.var_names_prognostic = ['hpert', 'u', 'v']
        
        hpert_grid_ = [libpdefd.GridInfo1D("hpert_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info(hpert_grid_, boundary_conditions=self.simconfig.boundary_conditions_hpert, staggered_dim=staggered_dim_hpert, variable_id="hpert")
        self.hpert_grid = libpdefd.GridInfoND(hpert_grid_, name="hpert")
        
        u_grid_ = [libpdefd.GridInfo1D("u_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info(u_grid_, boundary_conditions=self.simconfig.boundary_conditions_u, staggered_dim=staggered_dim_u, variable_id="hpert")
        self.u_grid = libpdefd.GridInfoND(u_grid_, name="u")
        
        v_grid_ = [libpdefd.GridInfo1D("v_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info(v_grid_, boundary_conditions=self.simconfig.boundary_conditions_v, staggered_dim=staggered_dim_v, variable_id="v")
        self.v_grid = libpdefd.GridInfoND(v_grid_, name="v")
        
        q_grid_ = [libpdefd.GridInfo1D("q_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info(q_grid_, boundary_conditions=self.simconfig.boundary_conditions_q, staggered_dim=staggered_dim_q, variable_id="q")
        self.q_grid = libpdefd.GridInfoND(q_grid_, name="q")
        
        """
        Grid Infos
        """
        self._grid_info_nd_list = [self.hpert_grid, self.u_grid, self.v_grid, self.q_grid]
        self.grid_info_nd_set = libpdefd.GridInfoNDSet(self._grid_info_nd_list)
        
        self.op_visc_u = libpdefd.OperatorDiffND(
                diff_dim = 0,
                diff_order = self.simconfig.sim_visc_order,
                src_grid = self.u_grid,
                dst_grid = self.u_grid,
            )
        
        self.op_visc_u += libpdefd.OperatorDiffND(
                diff_dim = 1,
                diff_order = self.simconfig.sim_visc_order,
                src_grid = self.u_grid,
                dst_grid = self.u_grid,
            )
        
        self.op_visc_u.bake()
        
        self.op_visc_v = libpdefd.OperatorDiffND(
                diff_dim = 0,
                diff_order = self.simconfig.sim_visc_order,
                src_grid = self.v_grid,
                dst_grid = self.v_grid,
            )
            
        self.op_visc_v += libpdefd.OperatorDiffND(
                diff_dim = 1,
                diff_order = self.simconfig.sim_visc_order,
                src_grid = self.v_grid,
                dst_grid = self.v_grid,
            )
        
        self.op_visc_v.bake()

    def getGridInfoNDSet(self):
        return self.grid_info_nd_set
    

    def getMeshNDSet(self):
        """
        Meshes
        """
        _mesh_nd_list = [libpdefd.MeshND(i) for i in self._grid_info_nd_list]
        mesh_nd_set = libpdefd.MeshNDSet(_mesh_nd_list)
        
        return mesh_nd_set


    def get_prog_variable_index_by_name(self, name):
        try:
            return self.var_names_prognostic.index(name)
        except:
            return None


    def get_vort_from_uv_to_q(self, u, v):
        vort = self.op_vort_du_dy_to_q(u) - self.op_vort_dv_dx_to_q(v)
        return vort

    
    def get_variable_set(self):
        """
        Setup variables
        """
        hpert_var = libpdefd.VariableND(self.hpert_grid, "hpert")
        u_var = libpdefd.VariableND(self.u_grid, "u")
        v_var = libpdefd.VariableND(self.v_grid, "v")
    
        return libpdefd.VariableNDSet([hpert_var, u_var, v_var])
    
    
    def get_variable_set_all(self):
        return self.get_variable_set()


    def print(self):
                
        print("")
        print(self.hpert_grid)
        print(self.hpert_grid[0])
        print(self.hpert_grid[1])
        
        print("")
        print(self.u_grid)
        print(self.u_grid[0])
        print(self.u_grid[1])
        
        print("")
        print(self.v_grid)
        print(self.v_grid[0])
        print(self.v_grid[1])
        
        print("")
        print(self.q_grid)
        print(self.q_grid[0])
        print(self.q_grid[1])



class SimPDE_SWELinearA(SWEBase):
    def __init__(self, simconfig):
        SWEBase.__init__(self, simconfig)
    
    
    def setup(
            self
    ):
        SWEBase.setup(self)


        """
        dhpert_dt = -h0*d/dx(u) - h0*d/dy(v)
        
        dhpert_dt = -op_diff_du_dx_to_hpert(u)*sim_h0 - op_diff_dv_dy_to_hpert(v)*sim_h0
        """
        self.op_diff_du_dx_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        self.op_diff_dv_dy_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        
        
        """
        du_dt = -g * dhpert/dx + f*v 
        
        du_dt = - op_diff_dhpert_dx_to_u(hpert)*g + op_interpolate_v_to_q(op_interpolate_q_to_u(v))*f 
        """
        self.op_diff_dhpert_dx_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.u_grid,
        ).bake()
        
        
        """
        dv_dt = -g * dhpert/dy - f*u 
        
        dv_dt = - op_diff_dhpert_dy_to_v(hpert)*g - op_interpolate_u_to_q(op_interpolate_q_to_v(u))*f 
        """
        self.op_diff_dhpert_dy_to_v = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.v_grid,
        ).bake()
        
        
        """
        Vorticity related visualization operators
        """
        self.op_vort_du_dy_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_vort_dv_dx_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()


    def dU_dt(self, U):
        hpert = U[0]
        u = U[1]
        v = U[2]

        self.du_dt = -self.op_diff_dhpert_dx_to_u(hpert)*self.simconfig.sim_g
        self.dv_dt = -self.op_diff_dhpert_dy_to_v(hpert)*self.simconfig.sim_g
        
        if self.simconfig.sim_f != 0:
            self.du_dt += v*self.simconfig.sim_f
            self.dv_dt += -u*self.simconfig.sim_f
            
        self.dhpert_dt = -self.op_diff_du_dx_to_hpert(u) - self.op_diff_dv_dy_to_hpert(v)
        
        if self.simconfig.sim_visc != 0:
            #dhpert_dt += visc_hpert(hpert)*sim_visc
            self.du_dt += self.op_visc_u(u)*self.simconfig.sim_visc
            self.dv_dt += self.op_visc_v(v)*self.simconfig.sim_visc

        retval = libpdefd.VariableNDSet_Empty_Like(U)
        retval[0] = self.dhpert_dt
        retval[1] = self.du_dt
        retval[2] = self.dv_dt
        return retval




class SimPDE_SWELinearC(SWEBase):
    """
    Sadourny, R. (1975). The Dynamics of Finite-Difference Models of the Shallow-Water Equations.
    Energy conserving scheme
    Linear parts only
    """
    def __init__(self, simconfig):
        SWEBase.__init__(self, simconfig)
    
    def setup(
            self
    ):
        SWEBase.setup(self)

        """
        dhpert_dt = -h0*d/dx(u) - h0*d/dy(v)
        
        dhpert_dt = -op_diff_du_dx_to_hpert(u)*sim_h0 - op_diff_dv_dy_to_hpert(v)*sim_h0
        """
        
        self.op_diff_du_dx_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        self.op_diff_dv_dy_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        
        
        """
        du_dt = -g * dhpert/dx + f*v 
        
        du_dt = - op_diff_dhpert_dx_to_u(hpert)*g + op_interpolate_v_to_q(op_interpolate_q_to_u(v))*f 
        """
        
        self.op_diff_dhpert_dx_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.u_grid,
        ).bake()
        
        self.op_interpolate_v_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_interpolate_q_to_u = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.q_grid,
            dst_grid = self.u_grid,
        ).bake()
        
        
        
        """
        dv_dt = -g * dhpert/dy - f*u 
        
        dv_dt = - op_diff_dhpert_dy_to_v(hpert)*g - op_interpolate_u_to_q(op_interpolate_q_to_v(u))*f 
        """
        
        self.op_diff_dhpert_dy_to_v = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.v_grid,
        ).bake()
        
        self.op_interpolate_u_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_interpolate_q_to_v = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.q_grid,
            dst_grid = self.v_grid,
        ).bake()
        
        
        """
        Vorticity related visualization operators
        """
        
        self.op_vort_du_dy_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_vort_dv_dx_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()


    def dU_dt(self, U):
        hpert = U[0]
        u = U[1]
        v = U[2]

        """
        Sadourny, R. (1975). The Dynamics of Finite-Difference Models of the Shallow-Water Equations.
        Energy conserving scheme
        """
        
        self.du_dt = -self.op_diff_dhpert_dx_to_u(hpert)*self.simconfig.sim_g
        self.dv_dt = -self.op_diff_dhpert_dy_to_v(hpert)*self.simconfig.sim_g
        
        if self.simconfig.sim_f != 0:
            self.du_dt += self.op_interpolate_q_to_u(self.op_interpolate_v_to_q(v)*self.simconfig.sim_f)
            self.dv_dt += -self.op_interpolate_q_to_v(self.op_interpolate_u_to_q(u)*self.simconfig.sim_f)
            
        self.dhpert_dt = -self.op_diff_du_dx_to_hpert(u) - self.op_diff_dv_dy_to_hpert(v)
        
        if self.simconfig.sim_visc != 0:
            #dhpert_dt += visc_hpert(hpert)*sim_visc
            self.du_dt += self.op_visc_u(u)*self.simconfig.sim_visc
            self.dv_dt += self.op_visc_v(v)*self.simconfig.sim_visc

        retval = libpdefd.VariableNDSet_Empty_Like(U)
        retval[0] = self.dhpert_dt
        retval[1] = self.du_dt
        retval[2] = self.dv_dt
        return retval



class SimPDE_SWENonlinearA(SWEBase):
    """
    Sadourny, R. (1975). The Dynamics of Finite-Difference Models of the Shallow-Water Equations.
    Energy conserving scheme
    Nonlinear parts, skipping C grid alignment
    """
    def __init__(self, simconfig):
        SWEBase.__init__(self, simconfig)
    
    def setup(
            self
    ):
        SWEBase.setup(self)

        """
        U = self.op_interpolate_hpert_to_u(hpert)*u
        V = self.op_interpolate_hpert_to_v(hpert)*v
        """
        
        """
        H = total_h*self.simconfig.sim_g + (self.op_interpolate_u_to_hpert(u*u) + self.op_interpolate_v_to_hpert(v*v))*0.5
        """
        
        """
        total_h_pv = self.op_interpolate_v_to_q(self.op_interpolate_h_to_v(total_h))
        """
        
        """
        q = (self.op_diff_dv_dx_to_q(v) - self.op_diff_du_dy_to_q(u) + self.simvars.sim_f) / total_h_pv
        """
        self.op_diff_dv_dx_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_diff_du_dy_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        """
        self.du_dt = self.op_interpolate_q_to_u(q*self.op_interpolate_v_to_q(V)) - self.op_diff_dH_dx_to_u(H)
        """
        self.op_diff_dH_dx_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.u_grid,
        ).bake()
        
        """
        self.dv_dt = -self.op_interpolate_q_to_v(q*self.op_interpolate_u_to_q(U)) - self.op_diff_dH_dy_to_v(H)
        """
        self.op_diff_dH_dy_to_v = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.v_grid,
        ).bake()
        
        """
        self.dhpert_dt = - self.op_diff_du_dx_to_hpert(U) - self.op_diff_dv_dy_to_hpert(V)
        """
        self.op_diff_du_dx_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        self.op_diff_dv_dy_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        """
        Vorticity related visualization operators
        """
        self.op_vort_du_dy_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_vort_dv_dx_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()

    def dU_dt(self, Uset):
        hpert = Uset[0]
        u = Uset[1]
        v = Uset[2]

    
        P = (hpert + self.simconfig.sim_h0)*self.simconfig.sim_g
        
        """
        Sardouny 1974, p. 683, right column
        """
        U = P*u
        V = P*v
        H = P + (u*u + v*v)*0.5
        
        P_to_q = P
        
        q = (self.op_diff_dv_dx_to_q(v) - self.op_diff_du_dy_to_q(u) + self.simconfig.sim_f) / P_to_q
        
        self.du_dt = q*V - self.op_diff_dH_dx_to_u(H)
        self.dv_dt = -q*U - self.op_diff_dH_dy_to_v(H)
        
        self.dhpert_dt = - self.op_diff_du_dx_to_hpert(U) - self.op_diff_dv_dy_to_hpert(V)
        
        self.dhpert_dt /= self.simconfig.sim_g
    
        if self.simconfig.sim_visc != 0:
            #dhpert_dt += visc_hpert(hpert)*sim_visc
            self.du_dt += self.op_visc_u(u)*self.simconfig.sim_visc
            self.dv_dt += self.op_visc_v(v)*self.simconfig.sim_visc

        retval = libpdefd.VariableNDSet_Empty_Like(Uset)
        retval[0] = self.dhpert_dt
        retval[1] = self.du_dt
        retval[2] = self.dv_dt
        return retval



class SimPDE_SWENonlinearC(SWEBase):
    """
    Sadourny, R. (1975). The Dynamics of Finite-Difference Models of the Shallow-Water Equations.
    Energy conserving scheme
    Full nonlinear equation
    """
    
    def __init__(self, simconfig):
        SWEBase.__init__(self, simconfig)
    
    def setup(
            self
    ):
        SWEBase.setup(self)
        
        """
        U = self.op_interpolate_hpert_to_u(hpert)*u
        V = self.op_interpolate_hpert_to_v(hpert)*v
        """
        self.op_interpolate_hpert_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.hpert_grid,
            dst_grid = self.u_grid,
        ).bake()
        self.op_interpolate_hpert_to_v = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.hpert_grid,
            dst_grid = self.v_grid,
        ).bake()
        
        """
        H = total_h*self.simconfig.sim_g + (self.op_interpolate_u_to_hpert(u*u) + self.op_interpolate_v_to_hpert(v*v))*0.5
        """
        self.op_interpolate_u_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.u_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        self.op_interpolate_v_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.v_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        """
        total_h_pv = self.op_interpolate_v_to_q(self.op_interpolate_h_to_v(total_h))
        """
        self.op_interpolate_h_to_v = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.hpert_grid,
            dst_grid = self.v_grid,
        ).bake()
        self.op_interpolate_v_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        """
        q = (self.op_diff_dv_dx_to_q(v) - self.op_diff_du_dy_to_q(u) + self.simvars.sim_f) / total_h_pv
        """
        self.op_diff_dv_dx_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()
        self.op_diff_du_dy_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        """
        self.du_dt = self.op_interpolate_q_to_u(q*self.op_interpolate_v_to_q(V)) - self.op_diff_dH_dx_to_u(H)
        """
        self.op_interpolate_v_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()
        self.op_interpolate_q_to_u = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.q_grid,
            dst_grid = self.u_grid,
        ).bake()
        self.op_diff_dH_dx_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.u_grid,
        ).bake()
        
        """
        self.dv_dt = -self.op_interpolate_q_to_v(q*self.op_interpolate_u_to_q(U)) - self.op_diff_dH_dy_to_v(H)
        """
        self.op_interpolate_u_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        self.op_interpolate_q_to_v = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            src_grid = self.q_grid,
            dst_grid = self.v_grid,
        ).bake()
        self.op_diff_dH_dy_to_v = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.hpert_grid,
            dst_grid = self.v_grid,
        ).bake()
        
        """
        self.dhpert_dt = - self.op_diff_du_dx_to_hpert(U) - self.op_diff_dv_dy_to_hpert(V)
        """
        self.op_diff_du_dx_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        self.op_diff_dv_dy_to_hpert = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.hpert_grid,
        ).bake()
        
        """
        Vorticity related visualization operators
        """
        self.op_vort_du_dy_to_q = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            src_grid = self.u_grid,
            dst_grid = self.q_grid,
        ).bake()
        
        self.op_vort_dv_dx_to_q = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            src_grid = self.v_grid,
            dst_grid = self.q_grid,
        ).bake()


    def dU_dt(self, Uset):
        hpert = Uset[0]
        u = Uset[1]
        v = Uset[2]

        """
        From SWEEEEEEEEEEEET
        
        /*
         * Sadourny energy conserving scheme
         *
         * Note, that this grid does not follow the formulation
         * in the paper of Robert Sadourny, but looks as follows:
         *
         *              ^
         *              |
         *       ______v0,1_____
         *       |             |
         *       |             |
         *       |             |
         *  u0,0 |->  H/P0,0   |u1,0 ->
         *(0,0.5)|             |
         *       |      ^      |
         *   q0,0|______|______|
         * (0,0)      v0,0
         *           (0.5,0)
         *
         * V_t + q N x (P V) + grad( g P + 1/2 V*V) = 0
         * P_t + div(P V) = 0
         */
         
        PlaneData U(i_h.planeDataConfig); // U flux
        PlaneData V(i_h.planeDataConfig); // V flux
        PlaneData H(i_h.planeDataConfig); // Bernoulli potential
        
        PlaneData total_h = i_h + simVars.sim.h0;

        /*
         * U and V updates
         */
        U = op.avg_b_x(total_h)*i_u;
        V = op.avg_b_y(total_h)*i_v;

        H = simVars.sim.gravitation*total_h + 0.5*(op.avg_f_x(i_u*i_u) + op.avg_f_y(i_v*i_v));

        // Potential vorticity
        PlaneData total_h_pv = total_h;
        total_h_pv = op.avg_b_x(op.avg_b_y(total_h));

        PlaneData q = (op.diff_b_x(i_v) - op.diff_b_y(i_u) + simVars.sim.plane_rotating_f0) / total_h_pv;

        // u, v tendencies
        // Energy conserving scheme
        o_u_t = op.avg_f_y(q*op.avg_b_x(V)) - op.diff_b_x(H);
        o_v_t = -op.avg_f_x(q*op.avg_b_y(U)) - op.diff_b_y(H);

        // P UPDATE
        // standard update
        o_h_t = -op.diff_f_x(U) - op.diff_f_y(V);
        """
        
        P = (hpert + self.simconfig.sim_h0)*self.simconfig.sim_g
        
        """
        Sardouny 1974, p. 683, right column
        """
        U = self.op_interpolate_hpert_to_u(P)*u
        V = self.op_interpolate_hpert_to_v(P)*v
        H = P + (self.op_interpolate_u_to_hpert(u*u) + self.op_interpolate_v_to_hpert(v*v))*0.5
        
        P_to_q = self.op_interpolate_v_to_q(self.op_interpolate_h_to_v(P))
        
        q = (self.op_diff_dv_dx_to_q(v) - self.op_diff_du_dy_to_q(u) + self.simconfig.sim_f) / P_to_q
        
        self.du_dt = self.op_interpolate_q_to_u(q*self.op_interpolate_v_to_q(V)) - self.op_diff_dH_dx_to_u(H)
        self.dv_dt = -self.op_interpolate_q_to_v(q*self.op_interpolate_u_to_q(U)) - self.op_diff_dH_dy_to_v(H)
        
        self.dhpert_dt = - self.op_diff_du_dx_to_hpert(U) - self.op_diff_dv_dy_to_hpert(V)
        
        self.dhpert_dt /= self.simconfig.sim_g
    
        if self.simconfig.sim_visc != 0:
            #dhpert_dt += visc_hpert(hpert)*sim_visc
            self.du_dt += self.op_visc_u(u)*self.simconfig.sim_visc
            self.dv_dt += self.op_visc_v(v)*self.simconfig.sim_visc

        retval = libpdefd.VariableNDSet_Empty_Like(Uset)
        retval[0] = self.dhpert_dt
        retval[1] = self.du_dt
        retval[2] = self.dv_dt
        return retval


