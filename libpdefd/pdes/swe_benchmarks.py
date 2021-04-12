
import libpdefd
import libpdefd.pdes.swe as pde_swe
import numpy as np
import sys



class Benchmarks:
    def __init__(self, benchmark_name):
        self.benchmark_name = benchmark_name

    
    def setup_simconfig(self, simconfig):
        """
        Benchmark:
            - geostrophic_balance
            - geostrophic_balance_with_bump
            - geostrophic_balance_symmetric
            - geostrophic_balance_symmetric_with_bump
            - gaussian_bump
        """

        if self.benchmark_name in ["geostrophic_balance", "geostrophic_balance_with_bump", "geostrophic_balance_symmetric", "geostrophic_balance_symmetric_with_bump"]:
            
            simconfig.sim_h0 = 1e4
            simconfig.sim_g = 9.80616
            omega = 7.292e-5
            simconfig.sim_f = omega*2.0*np.cos(np.pi*0.25)*0.25
            simconfig.sim_hpert_scaling = 1000
            
            simconfig.cell_res = np.array([int(simconfig.base_res*simconfig.sim_domain_aspect), simconfig.base_res])
            
            simconfig.domain_end = np.array([20e6*simconfig.sim_domain_aspect, 20e6])
    
            if self.benchmark_name in ["geostrophic_balance", "geostrophic_balance_with_bump"]:
                simconfig.boundary_conditions_hpert = ["periodic", "neumann"]
                simconfig.boundary_conditions_u = ["periodic", "neumann"]
                simconfig.boundary_conditions_v = ["periodic", "neumann"]
                simconfig.boundary_conditions_q = ["periodic", "neumann"]
            
            elif self.benchmark_name in ["geostrophic_balance_symmetric", "geostrophic_balance_symmetric_with_bump"]:
                simconfig.boundary_conditions_hpert = ["periodic", "periodic"]
                simconfig.boundary_conditions_u = ["periodic", "periodic"]
                simconfig.boundary_conditions_v = ["periodic", "periodic"]
                simconfig.boundary_conditions_q = ["periodic", "periodic"]
                    
            else:
                raise Exception("")
            
        elif self.benchmark_name in ["gaussian_bump"]:
            pass
        
        else:
            raise Exception("Benchmark '"+self.benchmark_name+"' not supported")

        simconfig.update()


    def setup_variables(
            self,
            simpde,
            simconfig : pde_swe.SimConfig,
            simmesh,
            variable_set_prognostic,
    ):
        variable_set_all = simpde.get_variable_set_all() 
        
        """
        Setup initial conditions
        """
    
        """
        d0 information:
        
        The smaller d0, the sharper the gradient, the sooner the vortex structures are generated.
        
        Choosing 'd0 = 1/9' leads to a behavior similar to the Galwesky et al. test case
        
        Choosing 'd0 = 1/20' leads to a significantly faster generation of the vortices.
        The smaller d0, the faster the vortices are generated.
        However, if the vortices are generated faster, they are also less smooth.
        """
    
        d0 = 1/16
        
        if self.benchmark_name in ["geostrophic_balance", "geostrophic_balance_with_bump", "geostrophic_balance_symmetric", "geostrophic_balance_symmetric_with_bump"]:
            
            if self.benchmark_name in ["geostrophic_balance", "geostrophic_balance_with_bump"]:
                y_offset = 0
            else:
                y_offset = -0.25
            
            
            def tanh_curve(mesh_data):
                y = mesh_data[:,:,1]/simconfig.domain_size[1] + y_offset
                y = -(y*2-1)/d0
                s = np.tanh(y)
                return s
            
            def dtanh_dy_curve(mesh_data):
                y = mesh_data[:,:,1]/simconfig.domain_size[1] + y_offset
                y = -(y*2-1)/d0
                
                s = 1.0/(np.cosh(y)**2)
                
                # post differentiation
                s *= -(2.0 / (simconfig.domain_size[1]*d0))
                
                return s
            
            if 1:
                """
                Surface height
                """
                
                h_mesh_array = simmesh['hpert'].data
                
                s = tanh_curve(h_mesh_array)*simconfig.sim_hpert_scaling
                variable_set_all['hpert'] += s
                
                if self.benchmark_name in ["geostrophic_balance_with_bump", "geostrophic_balance_symmetric_with_bump"]:
                    
                    pert = libpdefd.tools.gaussian_bump(
                                h_mesh_array,
                                ic_center = np.array([0.5, 0.5 - y_offset])*simconfig.domain_size,
                                domain_size = [simconfig.domain_size[0], simconfig.domain_size[1]],
                                boundary_condition = simconfig.boundary_conditions_hpert[0],
                                exp_parameter = 120.0,
                                x_scale_d3 = 1.0
                            )
                    
                    variable_set_all['hpert'] += pert*0.1
                
            if 1:
                """
                Velocity in x direction
                """
                u_mesh_array = simmesh['u'].data
                
                s = dtanh_dy_curve(u_mesh_array)
                s *= simconfig.sim_hpert_scaling
                s *= -simconfig.sim_g/simconfig.sim_f
                
                variable_set_all['u'] += s
        
        
            if self.benchmark_name in ["geostrophic_balance_symmetric", "geostrophic_balance_symmetric_with_bump"]:
                """
                Hpert
                """
                k = variable_set_all['hpert'].shape[1]
                variable_set_all['hpert'].data[:,:k//2] = np.flip(variable_set_all['hpert'][:,k-k//2:k], axis=1)
                
                """
                Velocity in x direction
                """
                k = variable_set_all['u'].shape[1]
                variable_set_all['u'].data[:,:k//2] = -np.flip(variable_set_all['u'][:,k-k//2:k], axis=1)
                
        
        elif self.benchmark_name == "gaussian_bump":
            
            h_mesh_array = simmesh['hpert'].data
    
            variable_set_all['hpert'] += libpdefd.tools.gaussian_bump(
                            h_mesh_array,
                            ic_center = simconfig.initial_condition_default_center*simconfig.domain_size,
                            domain_size = simconfig.domain_size,
                            boundary_condition = simconfig.boundary_conditions_hpert[0],
                            exp_parameter = 120.0
                        )
        
        else:
            raise Exception("Benchmark '"+self.benchmark_name+"' not implemented")



        """
        Setup prognostic variables if they are requested as well
        """
        for i in ['hpert', 'u', 'v']:
            variable_set_prognostic[i] = variable_set_all[i]
