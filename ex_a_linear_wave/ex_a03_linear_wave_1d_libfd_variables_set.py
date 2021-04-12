#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd


"""
************************** OPTIONS START **************************
"""

"""
Resolution of simulation in number of cells
"""
cell_res = 512

"""
Use grid staggering
"""
use_staggering = True

"""
Domain start/end coordinate
"""
domain_start = 0
domain_end = 320
domain_size = domain_end - domain_start

"""
Boundary condition: 'periodic', 'dirichlet' or 'neumann'
"""
boundary_condition_u = "dirichlet0"
boundary_condition_v = "dirichlet0"


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
if boundary_condition_u == "neumann0" or boundary_condition_v == "neumann0":
    use_symlog = True


"""
************************** OPTIONS END   **************************
"""


"""
Guess time step size
"""

dt = domain_size/(cell_res+1)
dt *= 0.5


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

boundaries_u = get_boundaries(boundary_condition_u)
boundaries_v = get_boundaries(boundary_condition_v)


"""
Setup variables
"""

u_grid = libpdefd.GridInfo1D("u")
v_grid = libpdefd.GridInfo1D("v")

"""
Setup grids for each variable
"""
if grid_setup == "auto":
    u_grid.setup_autogrid(domain_start, domain_end, cell_res+1, boundaries=boundaries_u, staggered=False)
    v_grid.setup_autogrid(domain_start, domain_end, cell_res+1, boundaries=boundaries_v, staggered=use_staggering)

elif grid_setup == "manual":
    if use_staggering:
        raise Exception("TODO: Implement this here, but it's supported in libpdefd")

    x = np.linspace(0, 1, cell_res+1, endpoint=True)
    
    x = np.tanh(x*2.0-1.0)
    x /= np.abs(x[0])

    x = x*0.5+0.5
    
    x *= domain_size
    x += domain_start
    
    u_grid.setup_manualgrid(x, boundaries=boundaries_u)
    v_grid.setup_manualgrid(x, boundaries=boundaries_v)

else:
    raise Exception("GridInfo1D setup '"+grid_setup+"' not supported")

"""
Show what we are doing
"""
print(u_grid)
print(v_grid)

"""
Setup differential operator
"""
rho_diff = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    dst_grid = v_grid,
    src_grid = u_grid,
)

vel_diff = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    dst_grid = u_grid,
    src_grid = v_grid,
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
    
    retval[0] = -vel_diff(U[1])
    retval[1] = -rho_diff(U[0])

    return retval



"""
Setup initial conditions
"""

u = libpdefd.Variable1D(u_grid, "u")
v = libpdefd.Variable1D(v_grid, "v")

ic_center = 0.75*(domain_start + domain_end)

def initial_condition(x):
    t = (x - ic_center)/domain_size
    t = t**2
    return np.exp(-t*120)

if boundary_condition_u == "periodic" or boundary_condition_v == "periodic":
    range_ = range(-10, 10+1)
else:
    range_ = [0]

for i in range_:
    u += initial_condition(u_grid.x_dofs + domain_size*i)
"""
u += libpdefd_tools.gaussian_bump(
                u_grid.x_dofs,
                ic_center = ic_center*domain_size,
                domain_size = domain_size,
                boundary_condition = boundary_condition_u,
                exp_parameter = 120.0
            )
"""

U = libpdefd.VariableNDSet([u,v])


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
    """
    Prepare plotting
    """
    
    import matplotlib.pyplot as plt
    
    import libtide.plot_config as pc
    fig, ax = pc.setup()

    ps = pc.PlotStyles()
    
    plotstyle = ps.getNextStyle(len(u_grid.x_dofs), 15)
    line_u, = ax.plot(U[0].grid.x_dofs, U[0].data, **plotstyle, label="u(x)")
    
    plotstyle = ps.getNextStyle(len(v_grid.x_dofs), 15)
    line_v, = ax.plot(U[1].grid.x_dofs, U[1].data, **plotstyle, label="v(x)")
    
    ax.legend()
    maxy = np.max(np.abs(U[0].data))
    ax.set_ylim(-maxy, maxy)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=1e-4)
    
    plt.show(block=False)


import time
time_start = time.time()

for i in range(num_timesteps):
    
    U = libpdefd.tools.RK4(dU_dt, U, dt)

    if output_freq != None:
        if i % output_freq == 0:
            line_u.set_ydata(U[0].data)
            line_v.set_ydata(U[1].data)
            
            ax.set_title("timestep "+str(i))
            
            fig.canvas.draw()
            fig.canvas.flush_events()


time_end = time.time()

print("Time: "+str(time_end-time_start))

if output_freq != None:
    plt.show()
