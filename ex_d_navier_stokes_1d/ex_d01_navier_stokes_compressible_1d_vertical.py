#! /usr/bin/env python3

import sys
import numpy as np
import libpdefd
#import libpdefd.libpdefd.libpdefd as libpdefd
#import libpdefd.libpdefd.libpdefd_tools as libpdefd_tools

import libpdefd.time.time_integrators as time_integrators
import libpdefd.pdes.navierstokes as pde_navierstokes

simconfig = pde_navierstokes.SimConfig(1)

simconfig.domain_start[0] = 0
simconfig.domain_end[0] = 6.4*1e3
#simconfig.use_staggering = False


"""
Straka benchmark parameters
"""
simconfig.const_R = 287
simconfig.const_c_p = 1004
simconfig.const_p0 = 100*1e3  # Surface pressure
simconfig.const_t0 = 300      # Surface temperature
simconfig.const_g = 9.81

grid_setup = "auto"

simconfig.update()

"""
Use symlog for plot
"""
use_symlog = False



"""
Use symlog for plot
"""
use_symlog = False


alpha = 1.0/(simconfig.kappa - 1.0)
beta = simconfig.kappa*alpha



def fun_t(z):
    return simconfig.const_t0 - z*simconfig.const_g/simconfig.const_c_p


def fun_p(z):
    return simconfig.const_p0 * np.power(fun_t(z)/simconfig.const_t0, 1.0/simconfig.kappa)

def fun_rho(z):
    return fun_p(z)/(simconfig.const_R*fun_t(z))



"""
Guess time step size
"""

dt = simconfig.domain_size[0]/(simconfig.cell_res[0]+1)
dt *= 0.001


"""
Number of time steps
"""
if simconfig.num_timesteps == None:
    if simconfig.sim_time != None:
        simconfig.num_timesteps = int(round(simconfig.sim_time/dt))
    else:
        simconfig.num_timesteps = None



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
    w_grid.setup_autogrid(simconfig.domain_start[0], simconfig.domain_end[0], simconfig.cell_res[0]+1, boundaries=boundaries_u, staggered=False)
    p_grid.setup_autogrid(simconfig.domain_start[0], simconfig.domain_end[0], simconfig.cell_res[0]+1, boundaries=boundaries_p, staggered=False)
    t_grid.setup_autogrid(simconfig.domain_start[0], simconfig.domain_end[0], simconfig.cell_res[0]+1, boundaries=boundaries_t, staggered=False)

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
    src_grid = p_grid,
    dst_grid = w_grid,
).bake()

op_w__grad_dw_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    src_grid = w_grid,
    dst_grid = w_grid,
).bake()

op_w__t_to_w = libpdefd.OperatorDiff1D(
    diff_order = 0,
    src_grid = t_grid,
    dst_grid = w_grid,
).bake()

op_w__p_to_w = libpdefd.OperatorDiff1D(
    diff_order = 0,
    src_grid = p_grid,
    dst_grid = w_grid,
).bake()

op_p__grad_dp_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    src_grid = p_grid,
    dst_grid = p_grid,
).bake()

op_p__div_dw_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    src_grid = w_grid,
    dst_grid = p_grid,
).bake()

op_p__w_to_p = libpdefd.OperatorDiff1D(
    diff_order = 0,
    src_grid = w_grid,
    dst_grid = p_grid,
).bake()


op_t__grad_dt_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    src_grid = t_grid,
    dst_grid = t_grid,
).bake()

op_t__div_dw_dz = libpdefd.OperatorDiff1D(
    diff_order = 1,
    src_grid = w_grid,
    dst_grid = t_grid,
).bake()

op_t__w_to_t = libpdefd.OperatorDiff1D(
    diff_order = 0,
    src_grid = w_grid,
    dst_grid = t_grid,
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
    w, p, t = U[:]
        
    dw_dt = -w*op_w__grad_dw_dz(w) - simconfig.const_R*op_w__t_to_w(t)/op_w__p_to_w(p) * op_w__grad_dp_dz(p) - simconfig.const_g
    dp_dt = -op_p__w_to_p(w)*op_p__grad_dp_dz(p) + alpha*p*op_p__div_dw_dz(w)
    dt_dt = -op_t__w_to_t(w)*op_t__grad_dt_dz(t) + beta*t*op_t__div_dw_dz(w)
    
    retval[0] = dw_dt
    retval[1] = dp_dt
    retval[2] = dt_dt
    
    return retval



"""
Setup initial conditions
"""

w = libpdefd.VariableND(w_grid, "w")
p = libpdefd.VariableND(p_grid, "p")
t = libpdefd.VariableND(t_grid, "t")


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


if simconfig.output_freq != None:
    """
    Prepare plotting
    """
    
    import matplotlib.pyplot as plt
    
    import libpdefd.plot_config as pc
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
        line_w.set_ydata((U[0] - U_t0[0]).to_numpy_array())
        line_p.set_ydata((U[1] - U_t0[1]).to_numpy_array()*0.005)
        line_t.set_ydata((U[2] - U_t0[2]).to_numpy_array()*5)

    ax.set_ylim(-10, 10)


update_plot()


"""
Setup time integrator
"""
class time_deriv:
    def comp_du_dt(self, i_U, i_timestamp, i_dt):
        return dU_dt(i_U)


time_integrator = time_integrators.TimeIntegrators(
        time_integration_method = simconfig.time_integration_method,
        diff_eq_methods = time_deriv(),
        time_integration_order = simconfig.time_integration_order,
        leapfrog_ra_filter_value = simconfig.time_leapfrog_ra_filter_value,
    )


import time
time_start = time.time()

prev_timestep = None
for i in range(simconfig.num_timesteps):
    
    U = time_integrator.time_integration_method.comp_time_integration(U, dt, dt*i)

    if simconfig.output_freq != None:
        if i % simconfig.output_freq == 0:
            
            ax.set_title("timestep "+str(i))
            update_plot()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            

time_end = time.time()

print("Time: "+str(time_end-time_start))

if simconfig.output_freq != None and simconfig.test_run == False:
    plt.show()
