#! /usr/bin/env python3
"""
Compressible Navier-Stokes Equation
"""

import sys
import numpy as np
import libpdefd
import argparse



class SimConfig:
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        
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
        
        self.init_phase = True
        
        """
        Setup default values at 0 meter altitude
        """
        
        """
        Specific gas constant in J kg^-1 K^-2
        """
        self.const_R = float(287)
        parser.add_argument('--const-r', dest="const_R", type=float, help="Specific gas constant")
        
        """
        Specific heat capacity in J kg^-1 K
        """
        self.const_c_p = float(1003.0) 
        parser.add_argument('--const-c-p', dest="const_c_p", type=float, help="Specific heat capacity")

        """
        Abbreviation of R/c_p
        
        Will be automatically initialized
        """
        self.kappa = None # dimensionless
        
        
        """
        Density at surface level in kg/m^3
        """
        self.const_rho0 = float(1.225)
        parser.add_argument('--const-rho0', dest="const_rho0", type=float, help="Density at surface level")
        
        """
        Pressure at surface level in N/m^3
        """
        self.const_p0 = float(101.3)*1e3
        parser.add_argument('--const-p0', dest="const_p0", type=float, help="Pressure at surface level")
        
        """
        Temperature at surface level in K (Kelvin)

        Will be computed with the ideal gas law based on rho and p
        """
        self.const_t0 = None
        self.compute_t0_ideal_gas()
        parser.add_argument('--const-t0', dest="const_t0", type=float, help="Temperature at surface level")
        
        

        self.const_g = 9.81       # Gravity
        parser.add_argument('--const-g', dest="const_g", type=float, help="Gravity")
        
        self.const_viscosity = float(0)
        parser.add_argument('--const-viscosity', dest="const_viscosity", type=float, help="Viscosity")

        self.const_viscosity_order = 2
        parser.add_argument('--viscosity-order', dest="viscosity_order", type=float, help="Viscosity order")
        
        self.dt_scaling = float(0.5)
        parser.add_argument('--dt-scaling', dest="dt_scaling", type=float, help="Scaling of time step size")
        

        """
        Benchmark name
        """
        self.benchmark_name = None
        parser.add_argument('--benchmark-name', dest="benchmark_name", type=str, help="Name of benchmark")

        """
        Visualization Variable: rho, u, v, t, p, tracer
        """
        self.vis_variable = "rho"
        parser.add_argument('--vis-variable', dest="vis_variable", type=str, help="Variable name to visualize")
        
        """
        Navier Stokes implementation:
        """
        o = """
 + nonlinear_a_grid__p_rho
 + nonlinear_a_grid__rho_t
 + nonlinear_a_grid__p_t   
 + nonlinear_c_grid__p_rho
 + nonlinear_c_grid__rho_t
 + nonlinear_c_grid__p_t    
"""

        self.ns_type = "nonlinear_a_grid__p_rho"
        parser.add_argument('--ns-type', dest="ns_type", type=str, help="Type of Navier Stokes equation\n"+o)
        
        
        """
        Minimum order of spatial approximation
        """
        self.min_spatial_approx_order = 2
        parser.add_argument('--min-spatial-approx-order', dest="min_spatial_approx_order", type=int, help="Minimum spatial approximation order")
        
        """
        Resolution of simulation in number of cells
        """
        base_res = 128
        self.cell_res = np.array([base_res for _ in range(2)])
        parser.add_argument('--cell-res', '-X', dest="cell_res", nargs="+", type=int, help="Resolution in each dimension (values separated by space)")
        
        
        
        """
        Center of initial condition (relative to domain)
        """
        self.initial_condition_default_center = [0.25 for _ in range(2)]
        parser.add_argument('--initial-condition-default-center', dest="initial_condition_default_center", nargs="+", type=float, help="Center of initial condition (values separated by space)")
        
        
        """
        Domain start/end coordinate
        """
        self.domain_start = np.array([0 for _ in range(2)])
        self.domain_end = np.array([np.pi*2.0 for i in range(2)])
        parser.add_argument('--domain-start', dest="domain_start", nargs="+", type=float, help="Start coordinate of domain (values separated by space)")
        parser.add_argument('--domain-end', dest="domain_end", nargs="+", type=float, help="End coordinate of domain (values separated by space)")
        
        self.sim_domain_aspect = 1
        
        """
        Boundary condition: 'periodic', 'dirichlet' or 'neumann'
        """
        self.boundary_conditions_rho = [["periodic" for _ in range(2)] for _i in range(2)]
        self.boundary_conditions_u = [["periodic" for _ in range(2)] for _i in range(2)]
        self.boundary_conditions_w = [["periodic" for _ in range(2)] for _i in range(2)]
        self.boundary_conditions_p = [["periodic" for _ in range(2)] for _i in range(2)]
        self.boundary_conditions_t = [["periodic" for _ in range(2)] for _i in range(2)]

        self.boundary_conditions_rho_u = [["periodic" for _ in range(2)] for _i in range(2)]
        self.boundary_conditions_rho_w = [["periodic" for _ in range(2)] for _i in range(2)]
        self.boundary_conditions_rho_t = [["periodic" for _ in range(2)] for _i in range(2)]
        
        """
        Grid setup: 'auto' or 'manual'
        """
        self.grid_setup = "auto"
        
        """
        How often to print output
        """
        self.verbosity = 0
        parser.add_argument('--verbosity', '-v', dest="verbosity", type=int, help="Verbosity")
        
        """
        How often to print output
        """
        self.gui = False
        parser.add_argument('--gui', '-g', dest="gui", type=str2bool, help="Activate GUI")
        
        """
        Visualization dimensions for 2D plots
        """
        self.vis_dim_x = 0
        self.vis_dim_y = 1
        parser.add_argument('--vis-dim-x', dest="vis_dim_x", type=int, help="Dimension to visualize along x axis")
        parser.add_argument('--vis-dim-y', dest="vis_dim_y", type=int, help="Dimension to visualize along y axis")
        
        
        """
        Contour information for plots
        """
        self.plot_contour_info = None
        parser.add_argument('--plot-contour-info', dest="plot_contour_info", nargs="+", type=int, help="Plot contour info (3 parameters used for numpy.arange()")
        
        """
        Time to sleep between time steps in seconds
        """
        self.timestep_sleep = 0
        parser.add_argument('--timestep-sleep', '-s', dest="timestep_sleep", type=float, help="Time to sleep between time steps in seconds")
        
        """
        How often to print output
        """
        self.output_text_freq = 0
        parser.add_argument('--output-text-freq', '-o', dest="output_text_freq", type=int, help="Output each n-th time step")
        
        """
        How often to do file output in time steps
        """
        self.output_plot_timesteps_interval = 0
        parser.add_argument('--output-plot-timesteps-interval', dest="output_plot_timesteps_interval", type=int, help="Write state to file each n-th time step")
        
        """
        How often to do file output in seconds
        """
        self.output_plot_simtime_interval = 0
        parser.add_argument('--output-plot-simtime-interval', dest="output_plot_simtime_interval", type=float, help="Write state to file each given time interval")
        
        """
        Output filename using placeholders 'VARNAME' for prog variable and 'TIMESTEP' for time step nummer
        """
        self.output_plot_filename = ""
        parser.add_argument('--output-plot-filename', dest="output_plot_filename", type=str, help="Output filename using placeholders 'VARNAME' for prog variable and 'TIMESTEP' for time step nummer, e.g. output_plot_VARNAME_TIMESTEP.pdf")
        
        """
        Output filename using placeholders 'VARNAME' for prog variable and 'TIMESTEP' for time step nummer
        """
        self.output_pickle_filename = ""
        parser.add_argument('--output-pickle-filename', dest="output_pickle_filename", type=str, help="Output filename using placeholders 'VARNAME' for prog variable and 'TIMESTEP' for time step nummer, e.g. output_pickle_VARNAME_TIMESTEP.pdf")
        
        """
        Maximum number of time steps to run
        """
        self.number_of_timesteps = None
        parser.add_argument('--number-of-timesteps', dest="number_of_timesteps", type=int, help="Number of time steps")
        
        """
        Time integrator
        """
        self.time_integrator = "rk4"
        parser.add_argument('--time-integrator', dest="time_integrator", type=str, help="Time integrator")
        
        """
        Maximum simulation time to run simulation for
        """
        self.sim_time = 60*60*45
        parser.add_argument('--sim-time', dest="sim_time", type=float, help="Total simulation time")
        
        """
        Timestep size
        """
        self.dt = None
        parser.add_argument('--dt', dest="dt", type=float, help="Time step size")
        
        """
        Variables initialized by update()
        """
        self.domain_size = None
        self.vis_slice = None
        
        """
        Gravitational field applied to 'w' if it's not 'None'
        """
        self.gravity_field = None
        
        """
        Activate variable guard to avoid setting variables which don't exist
        """    
        self.init_phase = False
        
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
        
        self.update()


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
        self.vis_slice  = [self.cell_res[i]//2 for i in range(2)]
        
        
        """
        Just convert to np.array in case it's not yet converted
        """
        self.cell_res = np.array(self.cell_res)
        
        """
        Some constants
        
        p = rho * T * R
        """        
        self.kappa = self.const_R / self.const_c_p
    
    
    def compute_p0_ideal_gas(self):
        self.const_p0 = self.const_rho0 * self.const_t0 * self.const_R
    
    def compute_rho0_ideal_gas(self):
        self.const_rho0 = self.const_p0 / (self.const_t0 * self.const_R)
    
    def compute_t0_ideal_gas(self):
        self.const_t0 = self.const_p0 / (self.const_rho0 * self.const_R)
    
    
    def print_config(self):
        print("SimConfig")
        print(" + const_rho0: "+str(self.const_rho0))
        print(" + const_p0: "+str(self.const_p0))
        print(" + const_t0: "+str(self.const_t0))
        print(" + const_R: "+str(self.const_R))
        print(" + const_c_p: "+str(self.const_c_p))
        print(" + kappa: "+str(self.kappa))
        print(" + const_g: "+str(self.const_g))
        print(" + const_viscosity: "+str(self.const_viscosity))
        print(" + const_viscosity_order: "+str(self.const_viscosity_order))
        print(" + dt_scaling: "+str(self.dt_scaling))
        print(" + vis_variable: "+str(self.vis_variable))
        print(" + ns_type: "+str(self.ns_type))
        print(" + min_spatial_approx_order: "+str(self.min_spatial_approx_order))
        print(" + cell_res: "+str(self.cell_res))
        print(" + initial_condition_default_center: "+str(self.initial_condition_default_center))
        print(" + vis_dim_x: "+str(self.vis_dim_x))
        print(" + vis_dim_y: "+str(self.vis_dim_y))
        print(" + domain_start: "+str(self.domain_start))
        print(" + domain_end: "+str(self.domain_end))
        print(" + sim_domain_aspect: "+str(self.sim_domain_aspect))
        print(" + boundary_conditions_rho: "+str([[str(i) for i in k] for k in self.boundary_conditions_rho]))
        print(" + boundary_conditions_u: "+str([[str(i) for i in k] for k in self.boundary_conditions_u]))
        print(" + boundary_conditions_w: "+str([[str(i) for i in k] for k in self.boundary_conditions_w]))
        print(" + boundary_conditions_p: "+str([[str(i) for i in k] for k in self.boundary_conditions_p]))
        print(" + boundary_conditions_t: "+str([[str(i) for i in k] for k in self.boundary_conditions_t]))
        print(" + boundary_conditions_rho_u: "+str([[str(i) for i in k] for k in self.boundary_conditions_rho_u]))
        print(" + boundary_conditions_rho_w: "+str([[str(i) for i in k] for k in self.boundary_conditions_rho_w]))
        print(" + boundary_conditions_rho_t: "+str([[str(i) for i in k] for k in self.boundary_conditions_rho_t]))
        print(" + timestep_sleep: "+str(self.timestep_sleep))
        print(" + grid_setup: "+str(self.grid_setup))
        print(" + plot_contour_info: "+str(self.plot_contour_info))
        print(" + output_text_freq: "+str(self.output_text_freq))
        print(" + gui: "+str(self.gui))
        print(" + output_plot_timesteps_interval: "+str(self.output_plot_timesteps_interval))
        print(" + output_plot_simtime_interval: "+str(self.output_plot_simtime_interval))
        print(" + output_plot_filename: "+str(self.output_plot_filename))
        print(" + output_pickle_filename: "+str(self.output_pickle_filename))
        print(" + number_of_timesteps: "+str(self.number_of_timesteps))
        print(" + sim_time: "+str(self.sim_time))
        print(" + domain_size: "+str(self.domain_size))
        print(" + vis_slice: "+str(self.vis_slice))
        pass



"""
Differential operators
"""
class SimPDE_Base:
    def __init__(self, simconfig):
        self.simconfig = simconfig
    
    
    """
    Setup grid
    """
    def _setup_grid_info_info(
            self,
            grid_info_nd,
            staggered_dim,
            boundary_conditions,
            variable_id
    ):
        for i in range(2):
            """
            Setup grids for each variable
            """
            if self.simconfig.grid_setup == "auto":
                if staggered_dim == 666:
                    staggered = True
                else:
                    staggered = (i==staggered_dim)
                grid_info_nd[i].setup_autogrid(self.simconfig.domain_start[i], self.simconfig.domain_end[i], self.simconfig.cell_res[i]+1, boundaries=boundary_conditions[i], staggered=staggered)
            
            elif self.simconfig.grid_setup == "manual":
                if "_c_" in self.simconfig.ns_type:
                    raise Exception("TODO: Implement this here, but it's supported in libpdefd")
            
                x = np.linspace(0, 1, self.simconfig.cell_res[1]+1, endpoint=True)
                
                x = np.tanh(x*2.0-1.0)
                x /= np.abs(x[0])
                
                x = x*0.5+0.5
                
                x *= self.simconfig.domain_size
                x += self.simconfig.domain_start
                
                grid_info_nd[i].setup_manualgrid(x, boundaries=boundary_conditions[i])
            
            else:
                raise Exception("Grid1D setup '"+self.simconfig.grid_setup+"' not supported")
    
    
    def setup(
            self
    ):
        if self.grid_layout == "a":
            staggered_dim_rho = -1
            staggered_dim_u = -1
            staggered_dim_v = -1
            staggered_dim_p = -1
            staggered_dim_t = -1
            
        elif self.grid_layout == "c":
            """
            TODO: Try out different grid layouts
            """
            staggered_dim_rho = -1
            staggered_dim_u = 0
            staggered_dim_v = 1
            staggered_dim_p = 666
            staggered_dim_t = 666
            
        else:
            raise Exception("Unknown grid staggering")
        
        u_grid_info_ = [libpdefd.GridInfo1D("u_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(u_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_u, staggered_dim=staggered_dim_u, variable_id="u")
        self.u_grid = libpdefd.GridInfoND(u_grid_info_, name="u")
        
        w_grid_info_ = [libpdefd.GridInfo1D("w_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(w_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_w, staggered_dim=staggered_dim_v, variable_id="w")
        self.w_grid = libpdefd.GridInfoND(w_grid_info_, name="w")
        
        p_grid_info_ = [libpdefd.GridInfo1D("p_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(p_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_p, staggered_dim=staggered_dim_p, variable_id="p")
        self.p_grid = libpdefd.GridInfoND(p_grid_info_, name="p")
        
        rho_grid_info_ = [libpdefd.GridInfo1D("rho_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(rho_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_rho, staggered_dim=staggered_dim_rho, variable_id="rho")
        self.rho_grid = libpdefd.GridInfoND(rho_grid_info_, name="rho")
        
        t_grid_info_ = [libpdefd.GridInfo1D("t_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(t_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_t, staggered_dim=staggered_dim_t, variable_id="t")
        self.t_grid = libpdefd.GridInfoND(t_grid_info_, name="t")
        
        
        """
        Helper grids for intermediate variables
        """
        rho_u_grid_info_ = [libpdefd.GridInfo1D("rho_u_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(rho_u_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_rho_u, staggered_dim=staggered_dim_rho, variable_id="rho_u")
        self.rho_u_grid = libpdefd.GridInfoND(rho_u_grid_info_, name="rho_u")
        
        rho_w_grid_info_ = [libpdefd.GridInfo1D("rho_w_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(rho_w_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_rho_w, staggered_dim=staggered_dim_rho, variable_id="rho_v")
        self.rho_w_grid = libpdefd.GridInfoND(rho_w_grid_info_, name="rho_w")
        
        rho_t_grid_info_ = [libpdefd.GridInfo1D("rho_t_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(rho_t_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_rho_t, staggered_dim=staggered_dim_rho, variable_id="rho_t")
        self.rho_t_grid = libpdefd.GridInfoND(rho_t_grid_info_, name="rho_t")
        
        """
        Potential temperature
        """
        pot_t_grid_info_ = [libpdefd.GridInfo1D("pot_t_d"+str(i), dim=i) for i in range(2)]
        self._setup_grid_info_info(pot_t_grid_info_, boundary_conditions=self.simconfig.boundary_conditions_t, staggered_dim=staggered_dim_t, variable_id="pot_t")
        self.pot_t_grid = libpdefd.GridInfoND(pot_t_grid_info_, name="pot_t")
        
        self._grid_info_info_nd_list = [self.u_grid, self.w_grid, self.p_grid, self.rho_grid, self.t_grid, self.pot_t_grid]
        self.grid_info_nd_set = libpdefd.GridInfoNDSet(self._grid_info_info_nd_list)
        
        
        """
        Setup Viscosity operators which are the same across all discretizations
        """
        self.op_u__laplace_u_to_u = libpdefd.OperatorDiffND(
                diff_dim = 0,
                diff_order = self.simconfig.const_viscosity_order,
                min_approx_order = self.simconfig.min_spatial_approx_order,
                src_grid = self.u_grid,
                dst_grid = self.u_grid,
            )
        self.op_u__laplace_u_to_u += libpdefd.OperatorDiffND(
                diff_dim = 1,
                diff_order = self.simconfig.const_viscosity_order,
                min_approx_order = self.simconfig.min_spatial_approx_order,
                src_grid = self.u_grid,
                dst_grid = self.u_grid,
            )
        
        
        
        self.op_w__laplace_w_to_w = libpdefd.OperatorDiffND(
                diff_dim = 0,
                diff_order = self.simconfig.const_viscosity_order,
                min_approx_order = self.simconfig.min_spatial_approx_order,
                src_grid = self.w_grid,
                dst_grid = self.w_grid,
            )
        self.op_w__laplace_w_to_w += libpdefd.OperatorDiffND(
                diff_dim = 1,
                diff_order = self.simconfig.const_viscosity_order,
                min_approx_order = self.simconfig.min_spatial_approx_order,
                src_grid = self.w_grid,
                dst_grid = self.w_grid,
            )
                
        
        self.alpha = 1.0/(self.simconfig.kappa - 1.0)
        self.beta = self.simconfig.kappa/(self.simconfig.kappa - 1.0)
        self.R = self.simconfig.const_R
        
        
    """
    Compute Exner pressure
    """
    def get_exner_from_p(self, data_p):
        a = self.simconfig.const_R / self.simconfig.const_c_p
        return (data_p / self.simconfig.const_p0) ** a

    
    def getGridInfoNDSet(self):
        return self.grid_info_nd_set
    
    
    def getMeshNDSet(self):
        """
        Meshes
        """
        _mesh_nd_list = [libpdefd.MeshND(i) for i in self._grid_info_info_nd_list]
        mesh_nd_set = libpdefd.MeshNDSet(_mesh_nd_list)
        
        return mesh_nd_set
    
    
    def get_prog_variable_index_by_name(self, name):
        try:
            return (['u', 'w']+self.thermo_vars).index(name)
        except:
            return None
    
    
    def get_variable_set(self):
        varset = [
                libpdefd.VariableND(self.u_grid, "u"),
                libpdefd.VariableND(self.w_grid, "w")
            ]
    
        if 'p' in self.thermo_vars:
            varset += [libpdefd.VariableND(self.p_grid, "p")]
    
        if 'rho' in self.thermo_vars:
            varset += [libpdefd.VariableND(self.rho_grid, "rho")]
    
        if 't' in self.thermo_vars:
            varset += [libpdefd.VariableND(self.t_grid, "t")]
    
        return libpdefd.VariableNDSet(varset)
    
    
    def get_variable_set_all(self):
        varset = [
                libpdefd.VariableND(self.u_grid, "u"),
                libpdefd.VariableND(self.w_grid, "w"),
                libpdefd.VariableND(self.p_grid, "p"),
                libpdefd.VariableND(self.rho_grid, "rho"),
                libpdefd.VariableND(self.t_grid, "t")
            ]
    
        return libpdefd.VariableNDSet(varset)

    
    def print(self):
                
        print("")
        print(self.rho_grid)
        print(self.rho_grid[0])
        print(self.rho_grid[1])
        
        print("")
        print(self.rho_u_grid)
        print(self.rho_u_grid[0])
        print(self.rho_u_grid[1])
        
        print("")
        print(self.rho_w_grid)
        print(self.rho_w_grid[0])
        print(self.rho_w_grid[1])
        
        print("")
        print(self.u_grid)
        print(self.u_grid[0])
        print(self.u_grid[1])
        
        print("")
        print(self.w_grid)
        print(self.w_grid[0])
        print(self.w_grid[1])
        
        print("")
        print(self.p_grid)
        print(self.p_grid[0])
        print(self.p_grid[1])



class SimPDE_NSNonlinearA__p_rho(SimPDE_Base):
    def __init__(self, simconfig):
        SimPDE_Base.__init__(self, simconfig)
        
        self.grid_layout = "a"
        
        self.thermo_vars = ["p", "rho"]
        
        self.var_names_prognostic = ["u", "w"] + self.thermo_vars
        
        self.setup()
    
    
    
    def setup(
            self
    ):
        SimPDE_Base.setup(self)
        
        """
        See docs/navier_stokes_compressible.pdf
        """
        
        """
        Momentum equations
        
        \frac{\partial u}{\partial t}u&=-\boldsymbol{v}\cdot\nabla u-\frac{1}{\rho}\nabla_{u}p
        """
        self.op_u__grad_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__grad_du_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__grad_dp_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__rho_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__w_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.u_grid,
        )
        
        
        """
        \frac{\partial v}{\partial t}v&=-\boldsymbol{v}\cdot\nabla v-\frac{1}{\rho}\nabla_{v}p
        """
        self.op_w__grad_dw_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.w_grid,
        )
        self.op_w__grad_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__grad_dp_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__rho_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__u_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.w_grid,
        )
        
        
        """
        Continuity
        
        \frac{\partial\rho}{\partial t}=-\nabla\cdot\left(\rho\vec{v}\right)
        """
        self.op_rho__div_drho_u_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_u_grid,
            dst_grid = self.p_grid,
        )
        self.op_rho__div_drho_w_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_w_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_rho__u_to_rho_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.rho_u_grid,
        )
        
        self.op_rho__rho_to_rho_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.rho_u_grid,
        )
        
        self.op_rho__w_to_rho_w = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.rho_w_grid,
        )
        
        self.op_rho__rho_to_rho_w = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.rho_w_grid,
        )
        
        
        
        """
        Pressure equation
        
        \frac{\partial p}{\partial t}&=-\boldsymbol{v}\cdot\nabla p+\alpha p\nabla\cdot\boldsymbol{v}
        """
        self.op_p__grad_dp_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.p_grid,
        )
        self.op_p__grad_dp_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__div_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.p_grid,
        )
        self.op_p__div_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__u_to_p = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__w_to_p = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.p_grid,
        )
    
    
    
    def dU_dt(self, Uset):
        u = Uset[0]
        w = Uset[1]
        p = Uset[2]
        rho = Uset[3]
        
        """
        du/dt
        """
        rho_reciprocal_ = self.op_u__rho_to_u(rho.reciprocal())
        w_ = self.op_u__w_to_u(w)
        self.du_dt =    -self.op_u__grad_du_dx(u) * u         \
                        -self.op_u__grad_du_dz(u) * w_        \
                        -rho_reciprocal_*self.op_u__grad_dp_dx(p)
        
        """
        dw/dt
        """
        rho_ = self.op_w__rho_to_w(rho.reciprocal())
        u_ = self.op_w__u_to_w(u)
        self.dw_dt =    -self.op_w__grad_dw_dx(w) * u_         \
                        -self.op_w__grad_dw_dz(w) * w         \
                        -rho_*self.op_w__grad_dp_dz(p)
        
        if self.simconfig.gravity_field is not None:
            self.dw_dt -= self.simconfig.gravity_field
        elif self.simconfig.const_g != 0:
            self.dw_dt -= self.simconfig.const_g

        """
        drho/dt
        """
        u_ = self.op_rho__u_to_rho_u(u)
        rho1_ = self.op_rho__rho_to_rho_u(rho)
        w_ = self.op_rho__w_to_rho_w(w)
        rho2_ = self.op_rho__rho_to_rho_w(rho)
        
        self.drho_dt =  - self.op_rho__div_drho_u_dx(u_*rho1_)     \
                        - self.op_rho__div_drho_w_dz(w_*rho2_)
        

        """
        dp/dt
        """
        u_ = self.op_p__u_to_p(u)
        w_ = self.op_p__w_to_p(w)                
        self.dp_dt =    -u_ * self.op_p__grad_dp_dx(p)         \
                        -w_ * self.op_p__grad_dp_dz(p)         \
                        + self.alpha * p * ( self.op_p__div_du_dx(u) + self.op_p__div_dw_dz(w))
        
        if self.simconfig.const_viscosity != 0:
            self.du_dt += self.simconfig.const_viscosity*self.op_u__laplace_u_to_u(u)
            self.dw_dt += self.simconfig.const_viscosity*self.op_w__laplace_w_to_w(w)
        
        retval = libpdefd.VariableNDSet_Empty_Like(Uset)
        
        retval[0] = self.du_dt
        retval[1] = self.dw_dt
        retval[2] = self.dp_dt
        retval[3] = self.drho_dt
        return retval


    def get_var(self, Uset, varname):
        """
        Return either prognostic variable from Uset or compute the variable if it doesn't exist
        """
        
        if varname in self.var_names_prognostic:
            idx = self.get_prog_variable_index_by_name(varname)
            return Uset[idx]
    
        p = Uset[2]
        rho = Uset[3]
        
        if varname == "p":
            return p
        
        elif varname == "rho":
            return rho
        
        elif varname == "t":
            t = p / (rho*self.R)
            return t
        
        elif varname == "pot_t":
            t = p / (rho*self.R)
            return t/self.get_exner_from_p(p)
        
        raise Exception("Unkown variable '"+str(varname))


class SimPDE_NSNonlinearA__p_t(SimPDE_Base):
    def __init__(self, simconfig):
        SimPDE_Base.__init__(self, simconfig)
        
        self.grid_layout = "a"
        self.thermo_vars = ["p", "t"]
        self.var_names_prognostic = ["u", "w"] + self.thermo_vars
        
        self.setup()
    
    
    def setup(
        self
    ):
        SimPDE_Base.setup(self)
        
        """
        See docs/navier_stokes_compressible.pdf
        """
        
        """
        Momentum equation
        
        \frac{\partial}{\partial t}u&=-\boldsymbol{v}\cdot\nabla u-\frac{1}{\rho}\nabla_{u}p
        """
        self.op_u__grad_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.u_grid,
        )
        self.op_u__grad_du_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__grad_dp_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__w_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__p_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__t_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.t_grid,
            dst_grid = self.u_grid,
        )
        
        
        """
        \frac{\partial}{\partial t}v&=-\boldsymbol{v}\cdot\nabla v-\frac{1}{\rho}\nabla_{v}p
        """
        self.op_w__grad_dw_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.w_grid,
        )
        self.op_w__grad_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__grad_dp_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__u_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__p_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__t_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.t_grid,
            dst_grid = self.w_grid,
        )
        
        
        """
        Pressure equations
        
        \frac{\partial p}{\partial t}&=-\boldsymbol{v}\cdot\nabla p+\alpha p\nabla\cdot\boldsymbol{v}
        """
        self.op_p__grad_dp_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__grad_dp_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.p_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__div_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.p_grid,
        )
        self.op_p__div_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__u_to_p = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_p__w_to_p = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.p_grid,
        )
    
        
        """
        \frac{\partial T}{\partial t}&=-\boldsymbol{v}\cdot\nabla T+\frac{RT}{c_{p}}\alpha\nabla\cdot\boldsymbol{v}.
        """
        self.op_t__grad_dt_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.t_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__grad_dt_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.t_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__div_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.t_grid,
        )
        self.op_t__div_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__u_to_t = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__w_to_t = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.t_grid,
        )
    
    
    def dU_dt(self, Uset):
        u = Uset[0]
        w = Uset[1]
        p = Uset[2]
        t = Uset[3]
        
        """
        du/dt
        """
        w_ = self.op_u__w_to_u(w)
        p_ = self.op_u__p_to_u(p)
        t_ = self.op_u__t_to_u(t)
        self.du_dt =    - self.op_u__grad_du_dx(u) * u         \
                        - self.op_u__grad_du_dz(u) * w_        \
                        - self.R*t_/p_*self.op_u__grad_dp_dx(p)
        
        """
        dw/dt
        """
        u_ = self.op_w__u_to_w(u)
        p_ = self.op_w__p_to_w(p)
        t_ = self.op_w__t_to_w(t)
        self.dw_dt =    - self.op_w__grad_dw_dx(w) * u_         \
                        - self.op_w__grad_dw_dz(w) * w         \
                        - self.R*t_/p_*self.op_w__grad_dp_dz(p)
        
        if self.simconfig.gravity_field is not None:
            self.dw_dt -= self.simconfig.gravity_field
        elif self.simconfig.const_g != 0:
            self.dw_dt -= self.simconfig.const_g
        
        """
        dp/dt
        """
        u_ = self.op_p__u_to_p(u)
        w_ = self.op_p__w_to_p(w)    
        self.dp_dt =    - u_ * self.op_p__grad_dp_dx(p)         \
                        - w_ * self.op_p__grad_dp_dz(p)         \
                        + self.alpha * p * (self.op_p__div_du_dx(u) + self.op_p__div_dw_dz(w))
        
        """
        dT/dt
        """
        u_ = self.op_t__u_to_t(u)
        w_ = self.op_t__w_to_t(w)    
        self.dt_dt =    - u_ * self.op_t__grad_dt_dx(t)         \
                        - w_ * self.op_t__grad_dt_dz(t)         \
                        + self.beta * t * (self.op_t__div_du_dx(u) + self.op_t__div_dw_dz(w))
        
        if self.simconfig.const_viscosity != 0:
            self.du_dt += self.simconfig.const_viscosity*self.op_u__laplace_u_to_u(u)
            self.dw_dt += self.simconfig.const_viscosity*self.op_w__laplace_w_to_w(w)
        
        retval = libpdefd.VariableNDSet_Empty_Like(Uset)
        
        retval[0] = self.du_dt
        retval[1] = self.dw_dt
        retval[2] = self.dp_dt
        retval[3] = self.dt_dt
        return retval


    def get_var(self, Uset, varname):
        """
        Return either prognostic variable from Uset or compute the variable if it doesn't exist
        """
        
        if varname in self.var_names_prognostic:
            idx = self.get_prog_variable_index_by_name(varname)
            return Uset[idx]
        
        p = Uset[2]
        t = Uset[3]
        
        if varname == "rho":
            rho = p/(t*self.R)
            return rho
        
        elif varname == "pot_t":
            return t/self.get_exner_from_p(p)
        
        raise Exception("Unkown variable '"+str(varname))



class SimPDE_NSNonlinearA__rho_t(SimPDE_Base):
    def __init__(self, simconfig):
        SimPDE_Base.__init__(self, simconfig)
        
        self.grid_layout = "a"
        
        self.thermo_vars = ["rho", "t"]
        
        self.var_names_prognostic = ["u", "w"] + self.thermo_vars
        
        self.setup()
    
    
    def setup(
        self
    ):
        SimPDE_Base.setup(self)
        
        """
        See docs/navier_stokes_compressible.pdf
        """
        
        """
        Momentum equation
        
        \frac{\partial u}{\partial t}&=-\boldsymbol{v}\cdot\nabla u-\frac{R}{\rho}\nabla_{u}\left(\rho T\right)
        """
        self.op_u__grad_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.u_grid,
        )
        self.op_u__grad_du_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__grad_drho_t_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_t_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__rho_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.u_grid,
        )
        
        self.op_u__w_to_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.u_grid,
        )
        
        
        
        """
        \frac{\partial}{\partial t}w=-\boldsymbol{v}\cdot\nabla w-\frac{R}{\rho}\nabla_{w}\left(\rho T\right)
        """
        self.op_w__grad_dw_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.w_grid,
        )
        self.op_w__grad_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__grad_drho_t_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_t_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__rho_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.w_grid,
        )
        
        self.op_w__u_to_w = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.w_grid,
        )
        
        
        
        """
        Continuity
        
        \frac{\partial\rho}{\partial t}=-\nabla\cdot\left(\rho\vec{v}\right)
        """
        self.op_rho__div_drho_u_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_u_grid,
            dst_grid = self.p_grid,
        )
        self.op_rho__div_drho_w_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_w_grid,
            dst_grid = self.p_grid,
        )
        
        self.op_rho__u_to_rho_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.rho_u_grid,
        )
        
        self.op_rho__rho_to_rho_u = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.rho_u_grid,
        )
        
        self.op_rho__w_to_rho_w = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.rho_w_grid,
        )
        
        self.op_rho__rho_to_rho_w = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.rho_grid,
            dst_grid = self.rho_w_grid,
        )
        
        
        
        """
        \frac{\partial T}{\partial t}&=-\boldsymbol{v}\cdot\nabla T+\frac{RT}{c_{p}}\alpha\nabla\cdot\boldsymbol{v}.
        """
        self.op_t__grad_dt_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.t_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__grad_dt_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.t_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__div_du_dx = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.t_grid,
        )
        self.op_t__div_dw_dz = libpdefd.OperatorDiffND(
            diff_dim = 1,
            diff_order = 1,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.t_grid,
        )
    
        self.op_t__u_to_t = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.u_grid,
            dst_grid = self.t_grid,
        )
        
        self.op_t__w_to_t = libpdefd.OperatorDiffND(
            diff_dim = 0,
            diff_order = 0,
            min_approx_order = self.simconfig.min_spatial_approx_order,
            src_grid = self.w_grid,
            dst_grid = self.t_grid,
        )
    
    
    
    def dU_dt(self, Uset):
        u = Uset[0]
        w = Uset[1]
        rho = Uset[2]
        t = Uset[3]
        
        """
        du/dt
        """
        rho_mul_t = rho*t
        
        rho_reciprocal_ = self.op_u__rho_to_u(rho.reciprocal())
        w_ = self.op_u__w_to_u(w)
        self.du_dt =    - self.op_u__grad_du_dx(u) * u          \
                        - self.op_u__grad_du_dz(u) * w_         \
                        - self.R*rho_reciprocal_*self.op_u__grad_drho_t_dx(rho_mul_t)
        
        """
        dw/dt
        """
        rho_reciprocal_ = self.op_w__rho_to_w(rho.reciprocal())
        u_ = self.op_w__u_to_w(u)
        self.dw_dt =    - self.op_w__grad_dw_dx(w) * u_         \
                        - self.op_w__grad_dw_dz(w) * w          \
                        - self.R*rho_reciprocal_*self.op_w__grad_drho_t_dz(rho_mul_t)
        
        if self.simconfig.gravity_field is not None:
            self.dw_dt -= self.simconfig.gravity_field
        elif self.simconfig.const_g != 0:
            self.dw_dt -= self.simconfig.const_g
        
        
        """
        drho/dt
        """
        u_ = self.op_rho__u_to_rho_u(u)
        rho1_ = self.op_rho__rho_to_rho_u(rho)
        w_ = self.op_rho__w_to_rho_w(w)
        rho2_ = self.op_rho__rho_to_rho_w(rho)
        self.drho_dt =  - self.op_rho__div_drho_u_dx(u_*rho1_)     \
                        - self.op_rho__div_drho_w_dz(w_*rho2_)
        
        """
        dT/dt
        """
        u_ = self.op_t__u_to_t(u)
        w_ = self.op_t__w_to_t(w)
        self.dt_dt =    - u_ * self.op_t__grad_dt_dx(t)         \
                        - w_ * self.op_t__grad_dt_dz(t)         \
                        + self.beta * t * (self.op_t__div_du_dx(u) + self.op_t__div_dw_dz(w))
        
        if self.simconfig.const_viscosity != 0:
            self.du_dt += self.simconfig.const_viscosity*self.op_u__laplace_u_to_u(u)
            self.dw_dt += self.simconfig.const_viscosity*self.op_w__laplace_w_to_w(w)
        
        retval = libpdefd.VariableNDSet_Empty_Like(Uset)
        
        retval[0] = self.du_dt
        retval[1] = self.dw_dt
        retval[2] = self.drho_dt
        retval[3] = self.dt_dt
        return retval


    def get_var(self, Uset, varname):
        assert isinstance(Uset, libpdefd.VariableNDSet)
        
        """
        Return either prognostic variable from Uset or compute the variable if it doesn't exist
        """
        if varname in self.var_names_prognostic:
            idx = self.get_prog_variable_index_by_name(varname)
            retval = Uset[idx]
            assert isinstance(retval, libpdefd.VariableND)
            assert isinstance(retval.data, libpdefd.array._array_base)
            return retval
        
        rho = Uset[2]
        t = Uset[3]
        
        if varname == "p":
            p = rho*t*self.R
            assert isinstance(p, libpdefd.VariableND)
            assert isinstance(p.data, libpdefd.array._array_base)
            return p
    
        elif varname == "pot_t":
            """
            Compute potential temperature
            """
            p = rho*t*self.R
            pot_t = t/self.get_exner_from_p(p)
            assert isinstance(pot_t, libpdefd.VariableND)
            assert isinstance(pot_t.data, libpdefd.array._array_base)
            return pot_t
        
        raise Exception("Unkown variable '"+str(varname))
