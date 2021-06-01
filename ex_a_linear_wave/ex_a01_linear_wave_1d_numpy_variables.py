#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd

from libpdefd.array_matrix.libpdefd_array import *


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

elif grid_setup == "manual":
    if use_staggering:
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
    src_grid = rho_grid,
    dst_grid = vel_grid,
).bake()

vel_diff = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = simconfig.min_spatial_approx_order,
    src_grid = vel_grid,
    dst_grid = rho_grid,
).bake()



"""
Runge-Kutta 4 time integrator
"""
def RK4(f, u, dt):
    N = len(u)

    k1 = f(u)
    k2 = f([u[i] + k1[i]*(dt*0.5) for i in range(N)])
    k3 = f([u[i] + k2[i]*(dt*0.5) for i in range(N)])
    k4 = f([u[i] + k3[i]*dt for i in range(N)])
    
    # u + 1/6*dt * (k1 + 2*k2 + 2*k3 + k4)
    return [u[i] + (k1[i] + k2[i]*2 + k3[i]*2 + k4[i])*(1./6.*dt) for i in range(N)]




"""
Time tendencies
"""
def dU_dt(U):
    """
    Compute
        u_t = v_x
        v_t = u_x
    """
    
    rho, vel = U[0], U[1]
    
    rho_t = -vel_diff(vel)
    vel_t = -rho_diff(rho)

    return [rho_t, vel_t]



"""
Setup initial conditions
"""
rho = array_zeros_like(rho_grid.x_dofs)
vel = array_zeros_like(vel_grid.x_dofs)

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
    rho[:] += initial_condition(rho_grid.x_dofs + simconfig.domain_size*i)

U = [rho, vel]



def U_scalar_mul(a, b):
    return [b[i]*a for i in range(len(b))]

def U_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]



if simconfig.output_freq != None:
    """
    Prepare plotting
    """
    
    import matplotlib.pyplot as plt
    import libpdefd.plot_config as pc
    
    fig, ax = pc.setup()
    ps = pc.PlotStyles()
    
    plotstyle = ps.getNextStyle(len(rho_grid.x_dofs), 15)
    line_rho, = ax.plot(rho_grid.x_dofs, U[0].to_numpy_array(), **plotstyle, label="rho(x)")
    
    plotstyle = ps.getNextStyle(len(vel_grid.x_dofs), 15)
    line_vel, = ax.plot(vel_grid.x_dofs, U[1].to_numpy_array(), **plotstyle, label="vel(x)")
    
    ax.legend()
    maxy = U[0].reduce_maxabs()
    ax.set_ylim(-maxy, maxy)
    if simconfig.use_symlog:
        ax.set_yscale("symlog", linthresh=1e-4)
    
    plt.show(block=False)



import time
time_start = time.time()


for i in range(simconfig.num_timesteps):
    
    U = RK4(dU_dt, U, simconfig.dt)

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
