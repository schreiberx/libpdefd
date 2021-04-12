#! /usr/bin/env python3

import sys
import numpy as np
import libtide.libfd.libfd_tools
import libtide.libfd.libfd as libfd



for num_dims in [2,3,4]:
    for boundary_condition in ["dirichlet", "neumann"]:
        
        base_res = 512 // 2**num_dims
        
        print("*"*80)
        print("* DIMS: "+str(num_dims))
        print("* BC: "+boundary_condition)
        print("* base_res: "+str(base_res))
        print("*"*80)
        
        sim_var_avg = 1.234
        cell_res = np.array([int(base_res + 2*(i+1)) for i in range(num_dims)])
        
        domain_start = np.array([-20*(i+1) for i in range(num_dims)])
        domain_end = np.array([20*(i+1) for i in range(num_dims)])
        domain_size = domain_end - domain_start
        
        min_spatial_approx_order = 2
        
        def setup_grid(
            grid_nd,
            staggered_dim
        ):
            for i in range(num_dims):    
                if boundary_condition == "dirichlet":
                    boundary_left = libfd.BoundaryDirichlet(sim_var_avg)
                    boundary_right = libfd.BoundaryDirichlet(sim_var_avg)
                elif boundary_condition == "neumann":
                    boundary_left = libfd.BoundaryNeumann()
                    boundary_right = libfd.BoundaryNeumann()
                
                boundaries = [boundary_left, boundary_right]
            
                staggered = (i==staggered_dim)
                grid_nd[i].setup_autogrid(domain_start[i], domain_end[i], cell_res[i]+1, boundaries=boundaries, staggered=staggered)
            
        
        """
        Grid for rho, v0, v1, p, T
        """
        grid_nd_ = [libfd.GridInfo1D("p_d"+str(i), dim=i) for i in range(num_dims)]
        staggered_dim = -1
        
        for i in range(num_dims):    
            boundary_left = libfd.BoundaryDirichlet(sim_var_avg)
            boundary_right = libfd.BoundaryDirichlet(sim_var_avg)
            boundaries = [boundary_left, boundary_right]
        
            staggered = (i==staggered_dim)
            grid_nd_[i].setup_autogrid(domain_start[i], domain_end[i], cell_res[i]+1, boundaries=boundaries, staggered=staggered)
        
        rho_grid = libfd.GridInfoND(grid_nd_, name="rho")
        
        grad = [ libfd.OperatorDiffND(
                        diff_dim = i,
                        diff_order = 1,
                        min_approx_order = min_spatial_approx_order,
                        src_grid = rho_grid,
                        dst_grid = rho_grid,
                    ) for i in range(num_dims)]
        
        var = libfd.VariableND(rho_grid)
        var += sim_var_avg
        
        for dim in range(num_dims):
            print(" + testing diff in dim: "+str(dim))
    
            err = np.max(np.abs(var.data)) - sim_var_avg
            assert err < 1e-10, "Error too high"
            
            g = grad[dim]
            
            grad_var = g(var)
            
            err = np.max(np.abs(grad_var.data))
            assert err < 1e-10, "Error too high"
    
