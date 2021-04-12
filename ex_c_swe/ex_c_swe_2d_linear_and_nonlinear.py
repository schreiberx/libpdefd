#! /usr/bin/env python3

import sys
import numpy as np

import libpdefd
import libpdefd.pdes.swe as pde_swe
import libpdefd.pdes.swe_benchmarks as pde_swe_benchmarks



"""
Configuration of simulation    
"""
simconfig = pde_swe.SimConfig()


"""
Visualization Variable: hpert, hpert_diff_hpert0, u, v, vort
"""
simconfig.vis_variable = "vort"


"""
SWE type: linear_a_grid, linear_c_grid, nonlinear_a_grid, nonlinear_c_grid    
"""
simconfig.swe_type = "nonlinear_c_grid"


"""
Order of approximation
"""
simconfig.min_spatial_approx_order = 2

simconfig.base_res = 128
simconfig.sim_domain_aspect = 2

simconfig.sim_visc = 0


"""
Update other variables to make everything else match the currently setup variables
"""
simconfig.base_res = 256
simconfig.sim_domain_aspect = 1


simconfig.dt_scaling *= 0.5e-0

if simconfig.swe_type == "linear_a_grid":
    simpde = pde_swe.SimPDE_SWELinearA(simconfig)
elif simconfig.swe_type == "linear_c_grid":
    simpde = pde_swe.SimPDE_SWELinearC(simconfig)
elif simconfig.swe_type == "nonlinear_a_grid":
    simpde = pde_swe.SimPDE_SWENonlinearA(simconfig)
elif simconfig.swe_type == "nonlinear_c_grid":
    simpde = pde_swe.SimPDE_SWENonlinearC(simconfig)
else:
    raise Exception("SWE of type '"+simconfig.swe_type+"' not supported")


"""
Benchmark:
    - geostrophic_balance
    - geostrophic_balance_with_bump
    - geostrophic_balance_symmetric
    - geostrophic_balance_symmetric_with_bump
    - gaussian_bump
"""

benchmark = pde_swe_benchmarks.Benchmarks("geostrophic_balance_with_bump")
benchmark.setup_simconfig(simconfig)


"""
Setup PDE itself (operators, grid info)
"""
simpde.setup()


"""
Generate grid infos
"""
simgridinfos = simpde.getGridInfoNDSet()


"""
Generate mesh
"""
simmeshes = simpde.getMeshNDSet()


"""
Get variable set
"""
variable_set = simpde.get_variable_set()

 
"""
Setup variable set
"""
benchmark.setup_variables(simpde, simconfig, simmeshes, variable_set)


U = variable_set

hpert0_var = variable_set['hpert'].copy()


"""
Guess time step size
"""
dt = np.min(simconfig.domain_size/(simconfig.cell_res+1))
dt *= 1.0/np.sqrt(simconfig.sim_h0*simconfig.sim_g)
dt *= simconfig.dt_scaling

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
        vis_dim_x = simconfig.vis_dim_x,
        vis_dim_y = simconfig.vis_dim_y,
        vis_slice = simconfig.vis_slice,
        rescale = 1.0
    )


    def plot_update_data():
        
        hpert_idx = simpde.get_prog_variable_index_by_name('hpert')
        u_idx = simpde.get_prog_variable_index_by_name('u')
        v_idx = simpde.get_prog_variable_index_by_name('v')

        prog_var_idx = simpde.get_prog_variable_index_by_name(simconfig.vis_variable)  
        if prog_var_idx is not None:
            vis.update_plots(simgridinfos[simconfig.vis_variable], U[prog_var_idx])

        else:
            if simconfig.vis_variable == "hpert_diff_hpert0":
                vis.update_plots(simgridinfos['hpert'], U['hpert']-hpert0_var)
            
            elif simconfig.vis_variable == "vort":
                vort = simpde.get_vort_from_uv_to_q(U[u_idx], U[v_idx])
                vis.update_plots(simpde.q_grid, vort)
            
            else:
                raise Exception("No valid vis_variable")
        
        for i in simpde.var_names_prognostic:
            idx = simpde.get_prog_variable_index_by_name(i)
            print(i+": min/max "+str(np.min(U[idx].data))+", "+str(np.max(U[idx].data)))



    def plot_update_title(i):
        title = ""
        title += simconfig.vis_variable
        title += ", t="+str(round(i*dt/(60*60), 3))+" h"
        print(title)
        vis.set_title(title)

    plot_update_data()
    plot_update_title(0)

    #vis.show()
    vis.show(block=False)

import time
time_start = time.time()

num_timesteps = int(6*60*60*24/dt)
num_timesteps = int(60*60*45/dt)
print("num_timesteps: "+str(num_timesteps))

for i in range(num_timesteps):
    
    U = libpdefd.tools.RK4(simpde.dU_dt, U, dt)

    if output_freq != None:
        if (i+1) % output_freq == 0:

            plot_update_data()
            plot_update_title(i)
            
            #vis.show()
            vis.show(block=False)


time_end = time.time()
print("*"*80)
print("Total simulation time: "+str(time_end-time_start))
print("*"*80)

if output_freq != None:
    plot_update_data()
    plot_update_title(i)
    
    vis.show(block=False)
    vis.show()
