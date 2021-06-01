#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd
import argparse


import simconfig_a
simconfig = simconfig_a.SimConfig()



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
        boundary_left = libpdefd.BoundaryNeumannExtrapolated(neumann_value=0)
        boundary_right = libpdefd.BoundaryNeumannExtrapolated(neumann_value=0)
        
    elif boundary_condition == "symmetric":
        boundary_left = libpdefd.BoundarySymmetric()
        boundary_right = libpdefd.BoundarySymmetric()
        
    else:
        raise Exception("Boundary condition '"+boundary_condition+"' is not supported")

    boundaries = [boundary_left, boundary_right]
    return boundaries



boundaries_rho = get_boundaries(simconfig.boundary_conditions_rho[0])
boundaries_vel = get_boundaries(simconfig.boundary_conditions_vel[0])


"""
Setup variables
"""

rho_grid = libpdefd.GridInfo1D("rho")
vel_grid = libpdefd.GridInfo1D("vel")

"""
Setup grids for each variable
"""
if simconfig.grid_setup == "auto":
    rho_grid.setup_autogrid(simconfig.domain_start[0], simconfig.domain_end[0], simconfig.cell_res[0]+1, boundaries=boundaries_rho, staggered=False)
    vel_grid.setup_autogrid(simconfig.domain_start[0], simconfig.domain_end[0], simconfig.cell_res[0]+1, boundaries=boundaries_vel, staggered=simconfig.use_staggering)

elif simconfig.grid_setup == "manual":
    if simconfig.use_staggering:
        raise Exception("TODO: Implement this here, but it's supported in libpdefd")

    x = np.linspace(0, 1, cell_res+1, endpoint=True)
    
    x = np.tanh(x*2.0-1.0)
    x /= np.abs(x[0])

    x = x*0.5+0.5
    
    x *= domain_size
    x += domain_start
    
    rho_grid.setup_manualgrid(x, boundaries=boundaries_rho)
    vel_grid.setup_manualgrid(x, boundaries=boundaries_vel)

else:
    raise Exception("GridInfo1D setup '"+grid_setup+"' not supported")

"""
Show what we are doing
"""
print(rho_grid)
print(vel_grid)

"""
Setup differential operator
"""
rho_diff = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = simconfig.min_spatial_approx_order,
    dst_grid = vel_grid,
    src_grid = rho_grid,
).bake()

vel_diff = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = simconfig.min_spatial_approx_order,
    dst_grid = rho_grid,
    src_grid = vel_grid,
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
    
    retval[0] = -vel_diff(U[1])
    retval[1] = -rho_diff(U[0])

    return retval



"""
Setup initial conditions
"""

u = libpdefd.VariableND(rho_grid, "u")
v = libpdefd.VariableND(vel_grid, "v")

ic_center = 0.75*(simconfig.domain_start + simconfig.domain_end)

def initial_condition(x):
    t = (x - ic_center)/simconfig.domain_size
    t = t**2
    return np.exp(-t*120)

if simconfig.boundary_conditions_rho[0] == "periodic" or simconfig.boundary_conditions_vel[0] == "periodic":
    range_ = range(-10, 10+1)
else:
    range_ = [0]

for i in range_:
    u += initial_condition(rho_grid.x_dofs + simconfig.domain_size*i)


U = libpdefd.VariableNDSet([u,v])


if simconfig.output_freq != None:
    """
    Prepare plotting
    """
    
    import matplotlib.pyplot as plt
    import libpdefd.plot_config as pc
    fig, ax = pc.setup()
    
    ps = pc.PlotStyles()
    
    plotstyle = ps.getNextStyle(len(rho_grid.x_dofs), 15)
    line_rho, = ax.plot(rho_grid.x_dofs, U[0].to_numpy_array(), **plotstyle, label="u(x)")
    
    plotstyle = ps.getNextStyle(len(vel_grid.x_dofs), 15)
    line_vel, = ax.plot(vel_grid.x_dofs, U[1].to_numpy_array(), **plotstyle, label="v(x)")
    
    ax.legend()
    maxy = U[0].reduce_maxabs()
    ax.set_ylim(-maxy, maxy)
    if simconfig.use_symlog:
        ax.set_yscale("symlog", linthresh=1e-4)
    
    plt.show(block=False)


import time
time_start = time.time()


for i in range(simconfig.num_timesteps):
    
    U = libpdefd.tools.RK4(dU_dt, U, simconfig.dt)

    if simconfig.output_freq != None:
        if i % simconfig.output_freq == 0:
            line_rho.set_ydata(U[0].to_numpy_array())
            line_vel.set_ydata(U[1].to_numpy_array())
            
            ax.set_title("timestep "+str(i))
            
            fig.canvas.draw()
            fig.canvas.flush_events()


time_end = time.time()

print("Time: "+str(time_end-time_start))

if simconfig.output_freq != None and simconfig.test_run == False:
    plt.show()
