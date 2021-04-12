#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd


"""
************************** OPTIONS START **************************
"""

"""
Sim settings
"""

dt_scaling = 1.0

"""
Simulation scenario:
    "rho_bump"
"""

"""
Default values for rescaling
"""
sim_rho_avg = 0
sim_rho_init_pert_rescale = 1e0

sim_v0_avg = 1
sim_v1_avg = 1.5


ic_gaussian_bump_exp_parameter = 120.0



"""
Resolution of simulation in number of cells
"""
base_res = 256



"""
Visualization dimensions for 2D plots
"""
vis_dim_x = 0
vis_dim_y = 1



"""
Domain start/end coordinate
"""
domain_start = np.array([0 for _ in range(2)])
domain_end = np.array([20e3 for i in range(2)])


"""
Visualization variable "rho", "v0", "v1", "op_v0__grad_drho_dx", "op_v1__grad_drho_dy", "op_rho__div_dv0_dx", "op_rho__div_dv1_dy"
"""
vis_variable = "rho"


"""
Initial condition
"""
initial_condition_center = np.array([0.5, 0.25])


"""
Boundary condition: 'periodic', 'dirichlet' or 'neumann'
"""
boundary_conditions_rho = ["periodic" for _ in range(2)]
boundary_conditions_v0 = ["periodic" for _ in range(2)]
boundary_conditions_v1 = ["periodic" for _ in range(2)]



cell_res = np.array([base_res for i in range(2)])
domain_size = domain_end - domain_start


"""
Slice to extract if the dimension is not visualized
"""
vis_slice  = [cell_res[i]//2 for i in range(2)]


"""
Grid setup: 'auto' or 'manual'
"""
grid_setup = "auto"


"""
Use grid staggering: "a", "c"
"""
grid_alignment = "a"


"""
Minimum order of spatial approximation
"""
min_spatial_approx_order = 2




"""
************************** OPTIONS END   **************************
"""


"""
Setup boundaries
"""
def get_boundaries(boundary_condition, variable_id, dim):
    
    # Use periodic boundaries
    if boundary_condition == "periodic":
        boundary_left = libpdefd.BoundaryPeriodic()
        boundary_right = libpdefd.BoundaryPeriodic()
        
    elif boundary_condition == "dirichlet0":
        if variable_id == "rho":
            boundary_left = libpdefd.BoundaryDirichlet(sim_rho_avg)
            boundary_right = libpdefd.BoundaryDirichlet(sim_rho_avg)
 
        elif variable_id == "v0":
            boundary_left = libpdefd.BoundaryDirichlet(sim_v0_avg)
            boundary_right = libpdefd.BoundaryDirichlet(sim_v0_avg)
            
        elif variable_id == "v1":
            boundary_left = libpdefd.BoundaryDirichlet(sim_v1_avg)
            boundary_right = libpdefd.BoundaryDirichlet(sim_v1_avg)
            
        else:
            raise Exception("TODO " + str(variable_id))
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


if grid_alignment not in ['a', 'c']:
    raise Exception("Grid alignment '"+str(grid_alignment)+"' not supported")

"""
Setup grid
"""
def setup_grid(
        grid_nd,
        staggered_dim,
        boundary_conditions,
        variable_id
):
    for i in range(2):
        boundaries = get_boundaries(boundary_conditions[i], variable_id, i)
        
        """
        Setup grids for each variable
        """
        if grid_setup == "auto":
            staggered = (i==staggered_dim)
            grid_nd[i].setup_autogrid(domain_start[i], domain_end[i], cell_res[i]+1, boundaries=boundaries, staggered=staggered)
        
        elif grid_setup == "manual":
            if grid_alignment == "c":
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
Grid for rho, v0, v1, p, T
"""
rho_grid_ = [libpdefd.GridInfo1D("rho_d"+str(i), dim=i) for i in range(2)]
staggered_dim = -1
setup_grid(rho_grid_, boundary_conditions=boundary_conditions_rho, staggered_dim=-1, variable_id="rho")
rho_grid = libpdefd.GridInfoND(rho_grid_, name="rho")

v0_grid_ = [libpdefd.GridInfo1D("v0_d"+str(i), dim=i) for i in range(2)]
if grid_alignment == "a":
    staggered_dim = -1
elif grid_alignment == "c":
    staggered_dim = 0
setup_grid(v0_grid_, boundary_conditions=boundary_conditions_v0, staggered_dim=staggered_dim, variable_id="v0")
v0_grid = libpdefd.GridInfoND(v0_grid_, name="v0")

v1_grid_ = [libpdefd.GridInfo1D("v1_d"+str(i), dim=i) for i in range(2)]
if grid_alignment == "a":
    staggered_dim = -1
elif grid_alignment == "c":
    staggered_dim = 1
setup_grid(v1_grid_, boundary_conditions=boundary_conditions_v1, staggered_dim=staggered_dim, variable_id="v1")
v1_grid = libpdefd.GridInfoND(v1_grid_, name="v1")



"""
Differential operators
"""
if grid_alignment == "a":
    """
    drho/dt = - \nabla \cdot (rho vel)
    
    drho_dt = -op_rho__div_dv0_dx(rho*v0) - op_rho__div_dv1_dy(rho*v1)
    """
    op_rho__div_dv0_dx = libpdefd.OperatorDiffND(
        diff_dim = 0,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        src_grid = v0_grid,
        dst_grid = rho_grid,
    )
    
    op_rho__div_dv1_dy = libpdefd.OperatorDiffND(
        diff_dim = 1,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        src_grid = v1_grid,
        dst_grid = rho_grid,
    )
    
elif grid_alignment == "c":
    """
    drho/dt = - \nabla \cdot (rho vel)
    
    drho_dt = -op_rho__div_dv0_dx(rho*v0) - op_rho__div_dv1_dy(rho*v1)
    
    drho_dt =   - op_rho__div_dv0_dx(v0)*rho - op_v0__grad_drho_dx(rho)*v0
                - op_rho__div_dv1_dy(v1)*rho - op_v1__grad_drho_dy(rho)*v1
    """
    op_rho__div_dv0_dx = libpdefd.OperatorDiffND(
        diff_dim = 0,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        src_grid = v0_grid,
        dst_grid = rho_grid,
    )
    
    op_rho__div_dv1_dy = libpdefd.OperatorDiffND(
        diff_dim = 1,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        src_grid = v1_grid,
        dst_grid = rho_grid,
    )
    
    op_v0__interpolate_from_rho = libpdefd.OperatorDiffND(
        diff_dim = 0,
        diff_order = 0,
        min_approx_order = min_spatial_approx_order,
        src_grid = rho_grid,
        dst_grid = v0_grid,
    )
    
    op_v1__interpolate_from_rho = libpdefd.OperatorDiffND(
        diff_dim = 1,
        diff_order = 0,
        min_approx_order = min_spatial_approx_order,
        src_grid = rho_grid,
        dst_grid = v1_grid,
    )
    
    op_v0__grad_drho_dx = libpdefd.OperatorDiffND(
        diff_dim = 0,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        src_grid = rho_grid,
        dst_grid = v0_grid,
    )
    
    op_v1__grad_drho_dy = libpdefd.OperatorDiffND(
        diff_dim = 1,
        diff_order = 1,
        min_approx_order = min_spatial_approx_order,
        src_grid = rho_grid,
        dst_grid = v1_grid,
    )
    
    op_rho__interpolate_from_v0 = libpdefd.OperatorDiffND(
        diff_dim = 0,
        diff_order = 0,
        min_approx_order = min_spatial_approx_order,
        src_grid = v0_grid,
        dst_grid = rho_grid,
    )
    
    op_rho__interpolate_from_v1 = libpdefd.OperatorDiffND(
        diff_dim = 1,
        diff_order = 0,
        min_approx_order = min_spatial_approx_order,
        src_grid = v1_grid,
        dst_grid = rho_grid,
    )



"""
Time tendencies
"""
def dU_dt(U):
    rho = U[0]
    
    if grid_alignment == "a":
        drho_dt = - op_rho__div_dv0_dx(v0_var*rho) - op_rho__div_dv1_dy(v1_var*rho)
        
    elif grid_alignment == "c":
        drho_dt =   - op_rho__div_dv0_dx(v0_var)*rho    \
                    - op_rho__interpolate_from_v0(op_v0__grad_drho_dx(rho)*v0_var)    \
                    - op_rho__div_dv1_dy(v1_var)*rho \
                    - op_rho__interpolate_from_v0(op_v1__grad_drho_dy(rho)*v1_var)
        
    retval = libpdefd.VariableNDSet_Empty_Like(U)
    
    retval[0] = drho_dt
    
    if 0:
        print(" + drho_dt: "+str(np.min(drho_dt.data))+", "+str(np.max(drho_dt.data)))

    return retval



"""
Setup variables
"""
rho_var = libpdefd.VariableND(rho_grid, "rho")
v0_var = libpdefd.VariableND(v0_grid, "v0")
v1_var = libpdefd.VariableND(v1_grid, "v1")

rho_var += sim_rho_avg
v0_var += sim_v0_avg
v1_var += sim_v1_avg



"""
Setup initial conditions
"""
rho_mesh = libpdefd.MeshND(rho_grid)
rho_var += libpdefd.tools.gaussian_bump(
                rho_mesh.data,
                ic_center = initial_condition_center*domain_size,
                domain_size = domain_size,
                boundary_condition = boundary_conditions_rho[0],
                exp_parameter = ic_gaussian_bump_exp_parameter
            )*sim_rho_init_pert_rescale

if 1:
    print(" + rho: "+str(np.min(rho_var.data))+", "+str(np.max(rho_var.data)))
    print(" + v0: "+str(np.min(v0_var.data))+", "+str(np.max(v0_var.data)))
    print(" + v1: "+str(np.min(v1_var.data))+", "+str(np.max(v1_var.data)))



U = libpdefd.VariableNDSet([rho_var])



"""
Guess time step size
"""
dt = np.min(domain_size/(cell_res+1))
dt *= 1.0/np.sqrt(np.max([np.abs(v0_var.data), np.abs(v1_var.data)]))
#dt *= 0.25
dt *= dt_scaling

print("dt: "+str(dt))


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
    
    vis = libpdefd.vis.Visualization2DMesh(
        vis_dim_x = vis_dim_x,
        vis_dim_y = vis_dim_y,
        vis_slice = vis_slice
    )

    def plot_update_data():
        if vis_variable == "rho":
            #vis.update_plots(rho_grid, U[0] - sim_rho_avg)
            vis.update_plots(rho_grid, U[0])
            
        elif vis_variable == "v0":
            vis.update_plots(v0_grid, v0_var)
            
        elif vis_variable == "v1":
            vis.update_plots(v1_grid, v1_var)
            
        elif vis_variable == "op_v0__grad_drho_dx":
            vis.update_plots(v0_grid, op_v0__grad_drho_dx(U[0]))
            
        elif vis_variable == "op_v1__grad_drho_dy":
            vis.update_plots(v1_grid, op_v1__grad_drho_dy(U[0]))
            
        elif vis_variable == "op_rho__div_dv0_dx":
            vis.update_plots(rho_grid, op_rho__div_dv0_dx(v0_var))
            
        elif vis_variable == "op_rho__div_dv1_dy":
            vis.update_plots(rho_grid, op_rho__div_dv1_dy(v1_var))
            
        else:
            raise Exception("No valid vis_varialbe")


    def plot_update_title(i):
        title = ""
        title += vis_variable
        title += ", t="+str(round(i*dt, 5))
        vis.set_title(title)

    plot_update_data()
    plot_update_title(0)

    vis.show(block=False)

import time
time_start = time.time()

for i in range(num_timesteps):
    
    U = libpdefd.tools.RK4(dU_dt, U, dt)
    #U = libpdefd.tools.RK1(dU_dt, U, dt)

    if output_freq != None:
        if (i+1) % output_freq == 0:
            plot_update_data()
            plot_update_title(i)
            
            #vis.show()
            vis.show(block=False)


time_end = time.time()
print("Time: "+str(time_end-time_start))

if output_freq != None:
    vis.show()
