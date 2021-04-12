#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd


"""
************************** OPTIONS START **************************
"""

"""
Number of dimensions
"""
num_dims = 2


"""
Resolution of simulation in number of cells
"""
base_res = 256//(2**num_dims)
#base_res = 16//(2**num_dims)
cell_res = np.array([int(base_res*(1+0.2*(i+1))) for i in range(num_dims)])


"""
Center of initial condition (relative to domain)
"""
initial_condition_center = np.array([0.25 for i in range(num_dims)])


"""
Visualization dimensions for 2D plots
"""
vis_dim_x = 0
vis_dim_y = 1


"""
Slice to extract if the dimension is not visualized
"""
vis_slice  = [cell_res[i]//2 for i in range(num_dims)]


"""
Use grid staggering
"""
use_staggering = True


"""
Domain start/end coordinate
"""
domain_start = np.array([0 for _ in range(num_dims)])
domain_end = np.array([320//(1+0.5*(i+1)) for i in range(num_dims)])
domain_size = domain_end - domain_start


"""
Boundary condition: 'periodic', '' or 'neumann'
"""
boundary_conditions_rho = ["dirichlet0" for _ in range(num_dims)]
boundary_conditions_v = ["dirichlet0" for _ in range(num_dims)]


"""
GridInfo1D layout: 'auto' or 'manual'
"""
grid_setup = "auto"


"""
Minimum order of spatial approximation
"""
min_spatial_approx_order = 2


"""
Use symlog for plot
"""
use_symlog = False
if "neumann0" in boundary_conditions_rho or "neumann0" in boundary_conditions_v:
    use_symlog = True


"""
This is an ugly hack which shouldn't used operationally.

If we work with staggered grids, some alignment checks would indicate that
we should first align our variables properly. Since this is just a demo, we
can override it. 
"""
op_gen_assert_aligned = False if use_staggering else True

"""
************************** OPTIONS END   **************************
"""

"""
Guess time step size
"""
dt = np.min(domain_size/(cell_res+1))
dt *= 0.25


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
    for i in range(num_dims):    
        boundaries = get_boundaries(boundary_conditions[i])
    
        """
        Setup grids for each variable
        """
        if grid_setup == "auto":
            grid_nd[i].setup_autogrid(domain_start[i], domain_end[i], cell_res[i]+1, boundaries=boundaries, staggered=staggered)
        
        elif grid_setup == "manual":
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
rho_grid_ = [libpdefd.GridInfo1D("rho_d"+str(i), dim=i) for i in range(num_dims)]
setup_grid(rho_grid_, staggered=False, boundary_conditions=boundary_conditions_rho)
rho_grid = libpdefd.GridInfoND(rho_grid_)


"""
Grid for velocity variables
"""
vels_grids_ = [[libpdefd.GridInfo1D("v"+str(j)+"_d"+str(i), dim=i) for i in range(num_dims)] for j in range(num_dims)]
for j in range(num_dims):
    setup_grid(vels_grids_[j], staggered=use_staggering, boundary_conditions=boundary_conditions_rho)
vels_grids = [libpdefd.GridInfoND(g) for g in vels_grids_]

for j in range(num_dims):
    print("Velocity "+str(j)+" grid information:")
    print(vels_grids[j])



"""
Setup differential operators
"""

"""
First, we set up the one for the conserved quantity (e.g. density, SWE height)
"""
rho_diff_ops = [None for i in range(num_dims+1)]
vels_diff_ops = [None for i in range(num_dims+1)]

for i in range(num_dims):
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
        min_approx_order = min_spatial_approx_order,
        dst_grid = rho_grid,
        src_grid = vels_grids[i],
        assert_aligned = op_gen_assert_aligned
    )
    
    
    """
    For the velocities, we need to compute
    
    \vel0_t = drho_dx0
    \vel1_t = drho_dx1
    \vel2_t = drho_dx2
    """
    vels_diff_ops[i] = libpdefd.OperatorDiffND(
        diff_dim = i,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        dst_grid = vels_grids[i],
        src_grid = rho_grid,
        assert_aligned = op_gen_assert_aligned
    )


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
    for i in range(1, num_dims):
        retval[0] += -rho_diff_ops[i](v[i])
    
    for i in range(0, num_dims):
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
vels_vars = [libpdefd.VariableND(vels_grids[i], "v_d"+str(i)) for i in range(num_dims)]


"""
Setup initial conditions
"""
rho_var += libpdefd.tools.gaussian_bump(
                rho_mesh.data,
                ic_center = initial_condition_center*domain_size,
                domain_size = domain_size,
                boundary_condition = boundary_conditions_rho[0],
                exp_parameter = 120.0
            )

U = libpdefd.VariableNDSet([rho_var]+vels_vars)


output_freq = 10
if len(sys.argv) >= 2:
    output_freq = int(sys.argv[1])
    if output_freq < 0:
        output_freq = None


num_timesteps = 10000
if len(sys.argv) >= 3:
    num_timesteps = int(sys.argv[2])
    if num_timesteps < 0:
        num_timesteps = None



if output_freq != None:
    
    if num_dims == 1:
        vis = libpdefd.vis.Visualization1D(
            use_symlog=use_symlog
        )
        
    else:
        vis = libpdefd.vis.Visualization2DMesh(
            vis_dim_x = vis_dim_x,
            vis_dim_y = vis_dim_y,
            vis_slice = vis_slice
        )
    
    if num_dims == 1:
        vis.update_plots([rho_grid, vels_grids[0]], [U[0], U[1]])
    else:
        vis.update_plots(rho_grid, rho_var)

    title = "timestamp: "+str(0)
    vis.set_title(title)
    
    vis.show(block=False)


import time
time_start = time.time()

for i in range(num_timesteps):
    
    # RK4
    U = libpdefd.tools.RK4(dU_dt, U, dt)

    if output_freq != None:
        if (i+1) % output_freq == 0:

            if num_dims == 1:
                vis.update_plots([rho_grid, vels_grids[0]], [U[0], U[1]])
            else:
                vis.update_plots(rho_grid, U[0])
                
            title = "timestamp: "+str(round(i*dt, 5))
            vis.set_title(title)

            vis.show(block=False)


time_end = time.time()
print("Time: "+str(time_end-time_start))

if output_freq != None:
    vis.show()
