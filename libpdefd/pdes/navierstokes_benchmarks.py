import libtide.libfd.libfd as libfd
import libtide.libfd.libfd_tools as libfd_tools
import libtide.libfd.libnsec as libsimpde
import libtide.libfd.atmos_consts as atmos_consts
import numpy as np
from scipy.interpolate import interp1d
import sys



class Benchmarks:
    
    D = 2

    def g_fun(self, mesh_data):
        y_coords = mesh_data[:,:,1]
        f = interp1d(atmos_consts.altitude, atmos_consts.gravity, kind='cubic')
        retval = f(y_coords)
        return retval
    
    def setup_simconfig_bc_horizontal_dirichlet(
        self,
        simconfig
    ):
        simconfig.boundary_conditions_u = [[libfd.BoundaryDirichlet(0) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_w = [[libfd.BoundaryDirichlet(0) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho = [[libfd.BoundaryDirichlet(simconfig.sim_rho_avg) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_p = [[libfd.BoundaryDirichlet(simconfig.sim_p_avg) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_t = [[libfd.BoundaryDirichlet(simconfig.sim_t_avg) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho_u = [[libfd.BoundaryDirichlet(simconfig.sim_rho_avg*0) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho_w = [[libfd.BoundaryDirichlet(simconfig.sim_rho_avg*0) for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho_t = [[libfd.BoundaryDirichlet(simconfig.sim_rho_avg*simconfig.sim_t_avg) for _ in range(2)] for _D in range(self.D)]
    
    
    def setup_simconfig_bc_horizontal_periodic(
        self,
        simconfig
    ):
        simconfig.boundary_conditions_u = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_w = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_p = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_t = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho_u = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho_w = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
        simconfig.boundary_conditions_rho_t = [[libfd.BoundaryPeriodic() for _ in range(2)] for _D in range(self.D)]
    
    def setup_simconfig_bc_vertical_periodic_dirichlet(
        self,
        simconfig
    ):
        """
        Zero velocities at top and bottom boundaries
        """
        simconfig.boundary_conditions_u = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                            [libfd.BoundarySymmetric(), libfd.BoundarySymmetric()],
                                        ]
        simconfig.boundary_conditions_w = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                            [libfd.BoundaryDirichlet(0), libfd.BoundaryDirichlet(0)],
                                        ]
        simconfig.boundary_conditions_rho_u = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                            [libfd.BoundarySymmetric(), libfd.BoundarySymmetric()],
                                        ]
        simconfig.boundary_conditions_rho_w = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                            [libfd.BoundaryDirichlet(0), libfd.BoundaryDirichlet(0)],
                                        ]
        
        if 0:
            """
            Interpolate boundary values 
            """
            f = interp1d(atmos_consts.altitude, atmos_consts.density, kind='cubic')
            rho_bottom = f(simconfig.domain_start[1])
            rho_top = f(simconfig.domain_end[1])
            simconfig.boundary_conditions_rho = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryDirichlet(rho_bottom), libfd.BoundaryDirichlet(rho_top)],
                                            ]
            
            f = interp1d(atmos_consts.altitude, atmos_consts.pressure, kind='cubic')
            p_bottom = f(simconfig.domain_start[1])
            p_top = f(simconfig.domain_end[1])
            simconfig.boundary_conditions_p = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryDirichlet(p_bottom), libfd.BoundaryDirichlet(p_top)],
                                            ]
            
            f = interp1d(atmos_consts.altitude, atmos_consts.pressure, kind='cubic')
            t_bottom = f(simconfig.domain_start[1])
            t_top = f(simconfig.domain_end[1])
            simconfig.boundary_conditions_t = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryDirichlet(t_bottom), libfd.BoundaryDirichlet(t_top)],
                                            ]
    
            simconfig.boundary_conditions_rho_t = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryDirichlet(rho_bottom*t_bottom), libfd.BoundaryDirichlet(rho_top*t_top)],
                                            ]
            
        elif 0:
            """
            4th order Neumann BC seem to work extremely well
            """
            order_neumann = 4
            simconfig.boundary_conditions_rho = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryNeumannExtrapolated(0, order_neumann), libfd.BoundaryNeumannExtrapolated(0, order_neumann)],
                                            ]
            
            simconfig.boundary_conditions_p = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryNeumannExtrapolated(0, order_neumann), libfd.BoundaryNeumannExtrapolated(0, order_neumann)],
                                            ]
            
            simconfig.boundary_conditions_t = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryNeumannExtrapolated(0, order_neumann), libfd.BoundaryNeumannExtrapolated(0, order_neumann)],
                                            ]
    
            simconfig.boundary_conditions_rho_t = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundaryNeumannExtrapolated(0, order_neumann), libfd.BoundaryNeumannExtrapolated(0, order_neumann)],
                                            ]

        else:
            """
            Use symmetric boundary conditions
            """
            simconfig.boundary_conditions_rho = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundarySymmetric(), libfd.BoundarySymmetric()],
                                            ]
            
            simconfig.boundary_conditions_p = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundarySymmetric(), libfd.BoundarySymmetric()],
                                            ]
            
            simconfig.boundary_conditions_t = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundarySymmetric(), libfd.BoundarySymmetric()],
                                            ]
    
            simconfig.boundary_conditions_rho_t = [ [libfd.BoundaryPeriodic() for _ in range(2)],
                                                [libfd.BoundarySymmetric(), libfd.BoundarySymmetric()],
                                            ]
            
            
    
    def setup_simconfig(
        self,
        simconfig
    ):
        if simconfig.benchmark_name in ["horizontal_bump"]:
            simconfig.const_g = 0
            simconfig.domain_end = np.array([10e3, 10e3])
            self.setup_simconfig_bc_horizontal_periodic(simconfig)
            
            simconfig.compute_t0_ideal_gas()
        
        elif simconfig.benchmark_name in ["horizontal_steady_state"]:
            simconfig.const_g = 0
            simconfig.domain_end = np.array([10e3, 10e3])
            self.setup_simconfig_bc_horizontal_periodic(simconfig)
            
            simconfig.compute_t0_ideal_gas()
        
        elif simconfig.benchmark_name in ["vertical_straka", "vertical_straka_symmetric", "vertical_straka_symmetric_nobump"]:
            
            """
            gas constant for dry air
            """
            simconfig.const_R = 287.0
            
            """
            specific heat at constant pressure
            """
            simconfig.const_c_p = 1004.0
            
            """
            specific heat at constant volume
            """
            #self.const_c_v = 717.0
            
            """
            reference pressure
            """
            simconfig.const_p0 = 100e3

            """
            graviational parameter
            """
            simconfig.const_g = 9.81
            
            """
            Viscosity parameter
            """
            simconfig.const_viscosity_order = 2
            simconfig.const_viscosity = 75
            
            """
            Surface temperature in Kelvin
            """
            simconfig.const_t0 = 300
            
            """
            Automatically compute the density
            """
            simconfig.compute_rho0_ideal_gas()
            
            """
            Domain size: 80 km x 80 km
            """
            if simconfig.benchmark_name in ["vertical_straka"]:
                simconfig.domain_start = np.array([0., 0.])
                simconfig.domain_end = np.array([25.6*1e3, 6.4*1e3])
            
            elif simconfig.benchmark_name in ["vertical_straka_symmetric", "vertical_straka_symmetric_nobump"]:
                simconfig.domain_start = np.array([-25.6*1e3, 0.])
                simconfig.domain_end = np.array([25.6*1e3, 6.4*1e3])
        
            
            """
            Boundary conditions
            """
            dirichlet0 = libfd.BoundaryDirichlet(0)
            symmetric = libfd.BoundarySymmetric()
    
            """
            Zero velocities at top and bottom boundaries
            """
            simconfig.boundary_conditions_u =       [ [dirichlet0 for _ in range(2)], [symmetric for _ in range(2)]]
            simconfig.boundary_conditions_rho_u =   [ [dirichlet0 for _ in range(2)], [symmetric for _ in range(2)]]

            simconfig.boundary_conditions_w =       [ [symmetric for _ in range(2)], [dirichlet0 for _ in range(2)]]
            simconfig.boundary_conditions_rho_w =   [ [symmetric for _ in range(2)], [dirichlet0 for _ in range(2)]]
            
            simconfig.boundary_conditions_rho =     [ [symmetric for _ in range(2)], [symmetric for _ in range(2)]]
            simconfig.boundary_conditions_p =       [ [symmetric for _ in range(2)], [symmetric for _ in range(2)]]
            simconfig.boundary_conditions_t =       [ [symmetric for _ in range(2)], [symmetric for _ in range(2)]]
            simconfig.boundary_conditions_rho_t =   [ [symmetric for _ in range(2)], [symmetric for _ in range(2)]]
        
        
        elif simconfig.benchmark_name in ["vertical_steady_state", "vertical_bump", "vertical_steady_state_gravity_field"]:
            
            simconfig.const_g = 9.81            
            simconfig.initial_condition_default_center = [0.5, 0.3]
            
            """
            Domain size: 80 km x 80 km
            """
            simconfig.domain_start = np.array([0., 0.])
            simconfig.domain_end = np.array([10e3, 10e3])
            
            self.setup_simconfig_bc_vertical_periodic_dirichlet(simconfig)
           
        
        elif simconfig.benchmark_name in ["vertical_bubble_robert"]:
            """
            Robert, A. (2002). Bubble Convection Experiments with a Semi-implicit Formulation of the Euler Equations.
            In Journal of the Atmospheric Sciences (Vol. 50, Issue 13, pp. 1865–1873).
            https://doi.org/10.1175/1520-0469(1993)050<1865:bcewas>2.0.co;2
            """
            simconfig.const_g = 9.81
            simconfig.gravity_field = None
            
            """
            Domain size: 1km x 1km
            """
            simconfig.domain_start = np.array([0., 0.])
            simconfig.domain_end = np.array([1e3, 1e3])
            
            self.setup_simconfig_bc_vertical_periodic_dirichlet(simconfig)

        
        else:
            raise Exception("Benchmark '"+simconfig.benchmark_name+"' not supported")
        
        simconfig.update()
    
    
    def setup_variables(
        self,
        simpde,
        simconfig : libsimpde.SimConfig,
        simmeshes,
        variable_set_prognostic,
        variable_set_prognostic_background = None
    ):
        variable_set_all = simpde.get_variable_set_all()
        
        sim_rho_scaling = 0.01    # Scaling of density IC
        sim_p_scaling = 0.01      # Scaling of pressure IC
        

        """
        d0 information:
        
        The smaller d0, the sharper the gradient, the sooner the vortex structures are generated.
        
        Choosing 'd0 = 1/9' leads to a behavior similar to the Galwesky et al. test case
        
        Choosing 'd0 = 1/20' leads to a significantly faster generation of the vortices.
        The smaller d0, the faster the vortices are generated.
        However, if the vortices are generated faster, they are also less smooth.
        """
        
        d0 = 1/16
        
        """
        Reset to default values
        """
        variable_set_all['u'].set_all(0)
        variable_set_all['w'].set_all(0)
        variable_set_all['p'].set_all(-1)
        variable_set_all['rho'].set_all(-1)
        variable_set_all['t'].set_all(-1)
        
        u_mesh_data = simmeshes['u'].data
        w_mesh_data = simmeshes['w'].data
        p_mesh_data = simmeshes['p'].data
        rho_mesh_data = simmeshes['rho'].data
        t_mesh_data = simmeshes['t'].data
        
        
        def fun_gaussian_bump(mesh_data, center_abs = None):
            if center_abs == None:
                center_abs = simconfig.initial_condition_default_center*simconfig.domain_size
            
            return libfd_tools.gaussian_bump(
                        p_mesh_data,
                        ic_center = center_abs,
                        domain_size = simconfig.domain_size,
                        boundary_condition = simconfig.boundary_conditions_rho[0],
                        exp_parameter = 120.0
                    )
        
        
        def fun_t_from_p_rho(mesh_data, p_var, rho_var):
            """
            Apply ideal gas las    p = R*T*rho
            t = p/(R*rho)
            """
            return p_var.data/(simconfig.const_R*rho_var.data)
        
        
        def fun_rho_from_p_t(mesh_data, p_var, t_var):
            """
            Apply ideal gas las    p = R*T*rho
            rho = p/(R*T)
            """
            return p_var.data/(simconfig.const_R*t_var.data)
        
        
        def fun_p_from_rho_t(mesh_data, rho_var, t_var):
            """
            Apply ideal gas las    p = R*T*rho
            """
            return simconfig.const_R*t_var.data*rho_var.data
    
    
        def setup_dofs_vertical_steady_state(
            variable_set_all
        ):
        
            def _setup_dofs_interpolated(
                varname,
                interpolation_x_values,
                interpolation_y_values
            ):
                """
                varname: Name of variable
                interpolation_values: interpolation values
                """
                grid_info_nd = simmeshes[varname].grid_info_nd
                
                """
                Load vertical DOF coordinates
                """
                y_dofs = grid_info_nd.grids1d_list[1].x_dofs
                shape = grid_info_nd.shape
                
                f = interp1d(interpolation_x_values, interpolation_y_values, kind='cubic')
                val_y = f(y_dofs)
                val_data = np.repeat(np.expand_dims(val_y, 0), shape[0], axis=0)
                return val_data
            
            variable_set_all['rho'].set(_setup_dofs_interpolated('rho', atmos_consts.altitude, atmos_consts.density))
            variable_set_all['p'].set(_setup_dofs_interpolated('p', atmos_consts.altitude, atmos_consts.pressure))
            variable_set_all['t'].set(fun_t_from_p_rho(t_mesh_data, variable_set_all['p'], variable_set_all['rho']))
            
        
        
        if simconfig.benchmark_name in ["horizontal_steady_state", "horizontal_bump"]:
            
            variable_set_all['p'].set(np.ones_like(p_mesh_data[:,:,0])*simconfig.sim_p_avg)
            variable_set_all['rho'].set(np.ones_like(rho_mesh_data[:,:,0])*simconfig.sim_rho_avg)
            variable_set_all['t'].set(fun_t_from_p_rho(t_mesh_data, variable_set_all['p'], variable_set_all['rho']))
            
            if variable_set_prognostic_background != None:
                for i in simpde.var_names_prognostic:
                    variable_set_prognostic_background[i] = variable_set_all[i]
            
            if simconfig.benchmark_name in ["horizontal_bump"]:
                            
                variable_set_all['p'] += fun_gaussian_bump(p_mesh_data)*simconfig.sim_p_avg*sim_p_scaling
                variable_set_all['rho'] += fun_gaussian_bump(rho_mesh_data)*simconfig.sim_rho_avg*sim_rho_scaling

                variable_set_all['t'].set(fun_t_from_p_rho(t_mesh_data, variable_set_all['p'], variable_set_all['rho']))
                

        elif simconfig.benchmark_name in ["vertical_straka", "vertical_straka_symmetric", "vertical_straka_symmetric_nobump"]:
            
            def fun_straka_t(mesh_data):
                z_coords = mesh_data[:,:,1]
                return simconfig.const_t0 - z_coords*simconfig.const_g/simconfig.const_c_p
            
            def fun_straka_p(mesh_data):
                return simconfig.const_p0*np.power(fun_straka_t(mesh_data)/simconfig.const_t0, simconfig.const_c_p/simconfig.const_R)
            
            variable_set_all['p'].set(fun_straka_p(t_mesh_data))
            
            if variable_set_prognostic_background != None:
                variable_set_all['t'].set(fun_straka_t(t_mesh_data))
                variable_set_all['rho'].set(fun_rho_from_p_t(t_mesh_data, variable_set_all['p'], variable_set_all['t']))
                
                for i in simpde.var_names_prognostic:
                    variable_set_prognostic_background[i] = variable_set_all[i].copy()
            
            if "nobump" not in simconfig.benchmark_name:
                x = t_mesh_data[:,:,0]
                z = t_mesh_data[:,:,1]
                
                xc = 0.0*1e3
                xr = 4.0*1e3
                zc = 3.0*1e3
                zr = 2.0*1e3
                
                L = np.sqrt( ((x-xc)/xr)**2 + ((z-zc)/zr)**2 )
                
                dt = -15.0 * (np.cos(np.pi*L) + 1.0)/2
                dt *= np.less_equal(L, 1).astype(float)
                
                variable_set_all['t'].set(fun_straka_t(t_mesh_data) + dt)

            else:
                variable_set_all['t'].set(fun_straka_t(t_mesh_data))
            
            variable_set_all['rho'].set(fun_rho_from_p_t(t_mesh_data, variable_set_all['p'], variable_set_all['t']))
            
        
        elif simconfig.benchmark_name in ["vertical_steady_state", "vertical_bump", "vertical_steady_state_gravity_field"]:
            
            if simconfig.benchmark_name in ["vertical_steady_state_gravity_field"]:
                simconfig.gravity_field = self.g_fun(w_mesh_data)
            
            setup_dofs_vertical_steady_state(variable_set_all)
            
            if variable_set_prognostic_background != None:
                for i in simpde.var_names_prognostic:
                    variable_set_prognostic_background[i] = variable_set_all[i]

            """
            Compute steady state pressure distribution
            """
            #p_static = simpde.op_p__grad_dp_dz.solve(variable_set_all['rho'].data*simconfig.const_g, solver_tol=1e-1)
            #variable_set_all['p'].set(p_static)
            
            #variable_set_all['t'].set(setup_dofs('t', atmos_consts.altitude, atmos_consts.temperature))
            
            if simconfig.benchmark_name == "vertical_bump":
                variable_set_all['rho'] -= fun_gaussian_bump(rho_mesh_data)*simconfig.const_rho0*sim_rho_scaling*10
                #variable_set_all['p'] += p_fun_bump(p_mesh_data, base=0.0)
            
            variable_set_all['t'].set(fun_t_from_p_rho(t_mesh_data, variable_set_all['p'], variable_set_all['rho']))
            
        elif simconfig.benchmark_name in ["vertical_bubble_robert"]:
            """
            Robert, A. (2002). Bubble Convection Experiments with a Semi-implicit Formulation of the Euler Equations.
            In Journal of the Atmospheric Sciences (Vol. 50, Issue 13, pp. 1865–1873).
            https://doi.org/10.1175/1520-0469(1993)050<1865:bcewas>2.0.co;2
            """
            
            setup_dofs_vertical_steady_state(variable_set_all)
            
            if variable_set_prognostic_background != None:
                for i in simpde.var_names_prognostic:
                    variable_set_prognostic_background[i] = variable_set_all[i]

            """
            Compute steady state pressure distribution
            """
            center_abs = [500, 510]
            variable_set_all['t'] += fun_gaussian_bump(rho_mesh_data, center_abs=center_abs)*0.5
            variable_set_all['rho'].set(fun_rho_from_p_t(t_mesh_data, variable_set_all['p'], variable_set_all['t']))
            
        else:
            raise Exception("Benchmark '"+simconfig.benchmark_name+"' not implemented")


        """
        Setup prognostic variables if they are requested as well
        """
        for i in simpde.var_names_prognostic:
            variable_set_prognostic[i] = variable_set_all[i]

