#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd


import simconfig_a
simconfig = simconfig_a.SimConfig(2)



"""
This is an ugly hack which shouldn't used operationally.

If we work with staggered grids, some alignment checks would indicate that
we should first align our variables properly. Since this is just a demo, we
can override it. 
"""
op_gen_assert_aligned = False if simconfig.use_staggering else True



"""
Setup boundaries
"""
def get_boundaries(boundary_condition):
    
    # Use periodic boundaries
    if boundary_condition == "periodic":
        boundary_left = libpdefd.BoundaryPeriodic()
        boundary_right = libpdefd.BoundaryPeriodic()
        
    elif boundary_condition == "dirichlet0":
        boundary_left = libpdefd.BoundaryDirichlet(0)
        boundary_right = libpdefd.BoundaryDirichlet(0)
        
    elif boundary_condition == "neumann0":
        boundary_left = libpdefd.BoundaryNeumannExtrapolated(0)
        boundary_right = libpdefd.BoundaryNeumannExtrapolated(0)
        
    elif boundary_condition == "symmetric":
        boundary_left = libpdefd.BoundarySymmetric()
        boundary_right = libpdefd.BoundarySymmetric()
        
    else:
        raise Exception("Boundary condition '"+boundary_condition+"' is not supported")

    boundaries = [boundary_left, boundary_right]
    return boundaries



"""
Setup grid
"""
def setup_grid(
        grid_nd,
        staggered,
        boundary_conditions
):
    for i in range(simconfig.num_dims):
        boundaries = get_boundaries(boundary_conditions[i])
    
        """
        Setup grids for each variable
        """
        if simconfig.grid_setup == "auto":
            grid_nd[i].setup_autogrid(simconfig.domain_start[i], simconfig.domain_end[i], simconfig.cell_res[i]+1, boundaries=boundaries, staggered=staggered)
        
        elif simconfig.grid_setup == "manual":
            if use_staggering:
                raise Exception("TODO: Implement this here, but it's supported in libpdefd")
        
            x = np.linspace(0, 1, cell_res[1]+1, endpoint=True)
            
            x = np.tanh(x*2.0-1.0)
            x /= np.abs(x[0])
        
            x = x*0.5+0.5
            
            x *= domain_size
            x += domain_start
            
            grid_nd[i].setup_manualgrid(x, boundaries=boundaries)
        
        else:
            raise Exception("GridInfo1D setup '"+grid_setup+"' not supported")



"""
Grid for \rho variable
"""
rho_grid_ = [libpdefd.GridInfo1D("rho_d"+str(i), dim=i) for i in range(simconfig.num_dims)]
setup_grid(rho_grid_, staggered=False, boundary_conditions=simconfig.boundary_conditions_rho)
rho_grid = libpdefd.GridInfoND(rho_grid_)



"""
Grid for velocity variables
"""
vels_grids_ = [[libpdefd.GridInfo1D("v"+str(j)+"_d"+str(i), dim=i) for i in range(simconfig.num_dims)] for j in range(simconfig.num_dims)]
for j in range(simconfig.num_dims):
    setup_grid(vels_grids_[j], staggered=simconfig.use_staggering, boundary_conditions=simconfig.boundary_conditions_rho)
vels_grids = [libpdefd.GridInfoND(g) for g in vels_grids_]

for j in range(simconfig.num_dims):
    print("Velocity "+str(j)+" grid information:")
    print(vels_grids[j])



"""
Setup differential operators
"""

"""
First, we set up the one for the conserved quantity (e.g. density, SWE height)
"""
rho_diff_ops = [None for i in range(simconfig.num_dims+1)]
vels_diff_ops = [None for i in range(simconfig.num_dims+1)]

for i in range(simconfig.num_dims):
    """
    For rho we need to compute
    
    \rho_t = dv0_dx0 + dv1_dx1 + dv2_dx2
    """

    """
    Create 1D differential operator
    """
    rho_diff_ops[i] = libpdefd.OperatorDiffND(
        diff_dim = i,
        diff_order = 1,
        min_approx_order = simconfig.min_spatial_approx_order,
        dst_grid = rho_grid,
        src_grid = vels_grids[i],
        assert_aligned = op_gen_assert_aligned
    ).bake()
    
    
    """
    For the velocities, we need to compute
    
    \vel0_t = drho_dx0
    \vel1_t = drho_dx1
    \vel2_t = drho_dx2
    """
    vels_diff_ops[i] = libpdefd.OperatorDiffND(
        diff_dim = i,
        diff_order = 1,
        min_approx_order = simconfig.min_spatial_approx_order,
        dst_grid = vels_grids[i],
        src_grid = rho_grid,
        assert_aligned = op_gen_assert_aligned
    ).bake()


"""
Time tendencies
"""
def dU_dt(U):
    """
    Compute
        u_t = v_x
        v_t = u_x
    """
    
    retval = libpdefd.VariableNDSet_Empty_Like(U)
    
    rho = U[0]
    v = U[1:]
    
    retval[0] = -rho_diff_ops[0](v[0])
    for i in range(1, simconfig.num_dims):
        retval[0] += -rho_diff_ops[i](v[i])
    
    for i in range(0, simconfig.num_dims):
        retval[1+i] = -vels_diff_ops[i](rho)
    
    return retval



"""
Generate mesh
"""
rho_mesh = libpdefd.MeshND(rho_grid)
vels_mesh = [libpdefd.MeshND(i) for i in vels_grids]

#rho_x_dofs = rho_grid.get_x_grid()
#vels_x_dofs = [vel.get_x_grid() for vel in vels_grids]

# For plotting
#rho_mesh = rho_grid.get_plotting_mesh()


"""
Setup variables
"""
rho_var = libpdefd.VariableND(rho_grid, "rho")
vels_vars = [libpdefd.VariableND(vels_grids[i], "v_d"+str(i)) for i in range(simconfig.num_dims)]


"""
Setup initial conditions
"""
rho_var += libpdefd.tools.gaussian_bump(
                rho_mesh.data,
                ic_center = simconfig.initial_condition_center*simconfig.domain_size,
                domain_size = simconfig.domain_size,
                boundary_condition = simconfig.boundary_conditions_rho[0],
                exp_parameter = 120.0
            )

U = libpdefd.VariableNDSet([rho_var]+vels_vars)


if simconfig.output_freq != None:
    
    if simconfig.num_dims == 1:
        vis = libpdefd.visualization.Visualization1D(
            use_symlog=use_symlog
        )
        
    else:
        vis = libpdefd.visualization.Visualization2DMesh(
            vis_dim_x = simconfig.vis_dim_x,
            vis_dim_y = simconfig.vis_dim_y,
            vis_slice = simconfig.vis_slice
        )
    
    if simconfig.num_dims == 1:
        vis.update_plots([rho_grid, vels_grids[0]], [U[0], U[1]])
    else:
        vis.update_plots(rho_grid, rho_var)

    title = "timestamp: "+str(0)
    vis.set_title(title)
    
    vis.show(block=False)


import time
time_start = time.time()

for i in range(simconfig.num_timesteps):
    
    # RK4
    U = libpdefd.tools.RK4(dU_dt, U, simconfig.dt)

    if simconfig.output_freq != None:
        if (i+1) % simconfig.output_freq == 0:

            if simconfig.num_dims == 1:
                vis.update_plots([rho_grid, vels_grids[0]], [U[0], U[1]])
            else:
                vis.update_plots(rho_grid, U[0])
                
            title = "timestamp: "+str(round(i*simconfig.dt, 5))
            vis.set_title(title)

            vis.show(block=False)


time_end = time.time()
print("Time: "+str(time_end-time_start))

if simconfig.output_freq != None and simconfig.test_run == False:
    vis.show()
