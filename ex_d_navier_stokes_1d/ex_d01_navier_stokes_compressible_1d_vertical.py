#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd
#import libtide.libpdefd.libpdefd as libpdefd
#import libtide.libpdefd.libpdefd_tools as libpdefd_tools


"""
************************** OPTIONS START **************************
"""

"""
Resolution of simulation in number of cells
"""
cell_res = 128

"""
Use grid staggering
"""
use_staggering = False

"""
Domain start/end coordinate
"""
domain_start = 0
domain_end = 6.4*1e3
domain_size = domain_end - domain_start



"""
GridInfo1D layout: 'auto' or 'manual'
"""
grid_setup = "auto"


"""
Minimum order of spatial approximation
"""
min_spatial_approx_order = 2


"""
Straka benchmark parameters
"""
const_R = 287
const_c_p = 1004
#const_K = 75
const_p0 = 100*1e3  # Surface pressure
const_t0 = 300      # Surface temperature
const_g = 9.81

kappa = const_R / const_c_p
alpha = 1.0/(kappa - 1.0)
beta = kappa*alpha


"""
Use symlog for plot
"""
use_symlog = False


"""
************************** OPTIONS END   **************************
"""


p0 = const_p0
t0 = const_t0
R = const_R
C = const_c_p
k = C/R
g = const_g

kappa = C/R
#kappa = R/C


def fun_t(z):
    return const_t0 - z*const_g/const_c_p


def fun_p(z):
    return const_p0 * np.power(fun_t(z)/const_t0, kappa)

def fun_rho(z):
    return fun_p(z)/(const_R*fun_t(z))



"""
Guess time step size
"""

dt = domain_size/(cell_res+1)
dt *= 0.001


"""
Setup boundaries
"""
def get_boundaries(boundary_condition, value_bottom, value_top):
    
    # Use periodic boundaries
    if boundary_condition == "dirichlet0":
        boundary_bottom = libpdefd.BoundaryDirichlet(value_bottom)
        boundary_top = libpdefd.BoundaryDirichlet(value_top)
        
    elif boundary_condition == "neumann0":
        boundary_bottom = libpdefd.BoundaryNeumannExtrapolated(value_bottom, diff_order=1)
        boundary_top = libpdefd.BoundaryNeumannExtrapolated(value_top, diff_order=1)
        
    elif boundary_condition == "symmetric":
        boundary_bottom = libpdefd.BoundarySymmetric()
        boundary_top = libpdefd.BoundarySymmetric()
        
    else:
        raise Exception("Boundary condition '"+boundary_condition+"' is not supported")
    
    boundaries = [boundary_bottom, boundary_top]
    return boundaries


"""
Boundary condition: 'dirichlet0', 'neumann0', 'symmetric'
"""

if 0:
    boundaries_u = get_boundaries("dirichlet0", 0, 0)
    boundaries_p = get_boundaries("dirichlet0", fun_p(domain_start), fun_p(domain_end))
    boundaries_t = get_boundaries("dirichlet0", fun_t(domain_start), fun_t(domain_end))

else:
    boundaries_u = get_boundaries("dirichlet0", 0, 0)
    boundaries_p = get_boundaries("symmetric", 0, 0)
    boundaries_t = get_boundaries("symmetric", 0, 0)


"""
Setup variables
"""

w_grid = libpdefd.GridInfo1D("u")
p_grid = libpdefd.GridInfo1D("p")
t_grid = libpdefd.GridInfo1D("t")

"""
Setup grids for each variable
"""
if grid_setup == "auto":
    w_grid.setup_autogrid(domain_start, domain_end, cell_res+1, boundaries=boundaries_u, staggered=False)
    p_grid.setup_autogrid(domain_start, domain_end, cell_res+1, boundaries=boundaries_p, staggered=False)
    t_grid.setup_autogrid(domain_start, domain_end, cell_res+1, boundaries=boundaries_t, staggered=False)

elif grid_setup == "manual":
    if use_staggering:
        raise Exception("TODO: Implement this here, but it's supported in libpdefd")

    x = np.linspace(0, 1, cell_res+1, endpoint=True)
    
    x = np.tanh(x*2.0-1.0)
    x /= np.abs(x[0])

    x = x*0.5+0.5
    
    x *= domain_size
    x += domain_start
    
    w_grid.setup_manualgrid(x, boundaries=boundaries_u)
    p_grid.setup_manualgrid(x, boundaries=boundaries_p)
    t_grid.setup_manualgrid(x, boundaries=boundaries_t)

else:
    raise Exception("GridInfo1D setup '"+grid_setup+"' not supported")

"""
Show what we are doing
"""
#print(w_grid)
#print(p_grid)
#print(t_grid)

"""
Setup differential operator
"""
op_w__grad_dp_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    src_grid = p_grid,
    dst_grid = w_grid,
)

op_w__grad_dw_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    src_grid = w_grid,
    dst_grid = w_grid,
)

op_w__t_to_w = libpdefd.OperatorDiff1D(
    diff_order = 0,
    min_approx_order = min_spatial_approx_order,
    src_grid = t_grid,
    dst_grid = w_grid,
)

op_w__p_to_w = libpdefd.OperatorDiff1D(
    diff_order = 0,
    min_approx_order = min_spatial_approx_order,
    src_grid = p_grid,
    dst_grid = w_grid,
)

op_p__grad_dp_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    src_grid = p_grid,
    dst_grid = p_grid,
)

op_p__div_dw_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    src_grid = w_grid,
    dst_grid = p_grid,
)

op_p__w_to_p = libpdefd.OperatorDiff1D(
    diff_order = 0,
    min_approx_order = min_spatial_approx_order,
    src_grid = w_grid,
    dst_grid = p_grid,
)


op_t__grad_dt_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    src_grid = t_grid,
    dst_grid = t_grid,
)

op_t__div_dw_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    min_approx_order = min_spatial_approx_order,
    src_grid = w_grid,
    dst_grid = t_grid,
)

op_t__w_to_t = libpdefd.OperatorDiff1D(
    diff_order = 0,
    min_approx_order = min_spatial_approx_order,
    src_grid = w_grid,
    dst_grid = t_grid,
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
    w, p, t = U[:]
        
    dw_dt = -w*op_w__grad_dw_dz(w) - const_R*op_w__t_to_w(t)/op_w__p_to_w(p) * op_w__grad_dp_dz(p) - const_g
    dp_dt = -op_p__w_to_p(w)*op_p__grad_dp_dz(p) + alpha*p*op_p__div_dw_dz(w)
    dt_dt = -op_t__w_to_t(w)*op_t__grad_dt_dz(t) + beta*t*op_t__div_dw_dz(w)
    
    retval[0] = dw_dt
    retval[1] = dp_dt
    retval[2] = dt_dt
    
    return retval



"""
Setup initial conditions
"""

w = libpdefd.Variable1D(w_grid, "w")
p = libpdefd.Variable1D(p_grid, "p")
t = libpdefd.Variable1D(t_grid, "t")


z_w = w_grid.x_dofs
z_p = p_grid.x_dofs
z_t = t_grid.x_dofs


w.set(z_w*0)
p.set(fun_p(z_p))
t.set(fun_t(z_t))


if 1:
    """
    Bump as initial condition
    """
    zc = 3.0*1e3
    zr = 2.0*1e3
    
    L = np.sqrt( ((z_t-zc)/zr)**2 )
    
    delta_t = -15.0 * (np.cos(np.pi*L) + 1.0)/2
    delta_t *= np.less_equal(L, 1).astype(float)
    
    t += delta_t


U = libpdefd.VariableNDSet([w, p, t])

U_t0 = U.copy()


output_freq = 1

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
    
    plotstyle = ps.getNextStyle(len(w_grid.x_dofs), 15)
    line_w, = ax.plot(w_grid.x_dofs, np.zeros_like(w_grid.x_dofs), **plotstyle, label="w(x)")
    
    plotstyle = ps.getNextStyle(len(p_grid.x_dofs), 15)
    line_p, = ax.plot(p_grid.x_dofs, np.zeros_like(p_grid.x_dofs), **plotstyle, label="p(x)")
    
    plotstyle = ps.getNextStyle(len(t_grid.x_dofs), 15)
    line_t, = ax.plot(t_grid.x_dofs, np.zeros_like(t_grid.x_dofs), **plotstyle, label="t(x)")
    
    ax.legend()
    #maxy = np.max(np.abs(U[0].data))
    #ax.set_ylim(-maxy, maxy)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=1e-4)
    
    plt.show(block=False)


def update_plot():
    if 0:
        line_w.set_ydata(U[0].data)
        line_p.set_ydata(U[1].data/const_p0)
        line_t.set_ydata(U[2].data/const_t0)

    else:
        line_w.set_ydata((U[0].data - U_t0[0].data))
        line_p.set_ydata((U[1].data - U_t0[1].data)*0.005)
        line_t.set_ydata((U[2].data - U_t0[2].data)*5)

    ax.set_ylim(-10, 10)

update_plot()


import time
time_start = time.time()

prev_timestep = None
for i in range(num_timesteps):
    
    import time
    #time.sleep(0.1)

    if 0:
        U = libpdefd.tools.time_integrator_RK4(dU_dt, U, dt)
        
    else:

        prev_timestep_next = U.copy()
        U = libpdefd.tools.time_integrator_leapfrog(dU_dt, U, dt, prev_timestep)
        prev_timestep = prev_timestep_next

    if output_freq != None:
        if i % output_freq == 0:
            
            ax.set_title("timestep "+str(i))
            update_plot()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            

time_end = time.time()

print("Time: "+str(time_end-time_start))

if output_freq != None:
    plt.show()
