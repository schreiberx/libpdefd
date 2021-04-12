#! /usr/bin/env python3

import sys
import time
import numpy as np
import libpdefd
import libpdefd.pdes.navierstokes as pde_navierstokes
import libpdefd.pdes.navierstokes_benchmarks as pde_navierstokes_benchmarks


"""
Configuration of simulation    
"""
simconfig = pde_navierstokes.SimConfig()


"""
Setup benchmark constants:

    - geostrophic_balance
    - geostrophic_balance_with_bump
    - geostrophic_balance_symmetric
    - geostrophic_balance_symmetric_with_bump
    - gaussian_bump
"""
benchmark = pde_navierstokes_benchmarks.Benchmarks()
benchmark.setup_simconfig(simconfig)


simconfig.print_config()


"""
Setup PDE itself
"""
if simconfig.ns_type == "nonlinear_a_grid__p_rho":
    simpde = pde_navierstokes.SimPDE_NSNonlinearA__p_rho(simconfig)
    
elif simconfig.ns_type == "nonlinear_a_grid__rho_t":
    simpde = pde_navierstokes.SimPDE_NSNonlinearA__rho_t(simconfig)
    
elif simconfig.ns_type == "nonlinear_a_grid__p_t":
    simpde = pde_navierstokes.SimPDE_NSNonlinearA__p_t(simconfig)

else:
    raise Exception("SWE of type '"+simconfig.ns_type+"' not supported")


"""
Generate grid information
"""
print("Setup GridInfoNDSet")
simgridinfondset = simpde.getGridInfoNDSet()

"""
Generate mesh
"""
print("Setup MeshNDSet")
simmeshes = simpde.getMeshNDSet()

"""
Get variable set
"""
variable_set = simpde.get_variable_set()    # Prognostic ones
variable_set_background = simpde.get_variable_set()    # Prognostic ones

"""
Setup variable set
"""
print("Setup benchmark variables")
benchmark.setup_variables(simpde, simconfig, simmeshes, variable_set, variable_set_background)

U = variable_set

if 0:
    p_t0 = simpde.get_var(U, "p")
    rho_t0 = simpde.get_var(U, "rho")
    t_t0 = simpde.get_var(U, "t")
else:
    p_t0 = simpde.get_var(variable_set_background, "p")
    rho_t0 = simpde.get_var(variable_set_background, "rho")
    t_t0 = simpde.get_var(variable_set_background, "t")


"""
Guess time step size
"""

dt = np.min(simconfig.domain_size/(simconfig.cell_res+1))
dt *= 1.0/np.sqrt(simconfig.const_rho0)
dt *= simconfig.dt_scaling

if simconfig.dt is not None:
    dt = simconfig.dt


print("dt: "+str(dt))



plot_generate = simconfig.output_plot_filename != "" or simconfig.output_pickle_filename != "" or simconfig.gui


if plot_generate:
    vis = libpdefd.vis.Visualization2DMesh(
        vis_dim_x = simconfig.vis_dim_x,
        vis_dim_y = simconfig.vis_dim_y,
        vis_slice = simconfig.vis_slice,
        rescale = 1.0
    )
    
    
    def plot_get_data(variable_name):
        
        if variable_name in ['u', 'w', 'rho', 't', 'p']:
            var = simpde.get_var(U, variable_name)
            vargridinfo = simgridinfondset[variable_name]
        
        elif variable_name == "p_diff":
            var = simpde.get_var(U, "p") - p_t0
            vargridinfo = simgridinfondset["p"]
        
        elif variable_name == "rho_diff":
            var = simpde.get_var(U, "rho") - rho_t0
            vargridinfo = simgridinfondset["rho"]
        
        elif variable_name == "t_diff":
            var = simpde.get_var(U, "t") - t_t0
            vargridinfo = simgridinfondset["t"]
            
        else:
            raise Exception("variable_name "+str(variable_name)+" not supported")
        
        return var.data, vargridinfo

    
    def plot_update_title(i, vis_variable = None, title_prefix=""):
        if vis_variable == None:
            vis_variable = simconfig.vis_variable
        
        title = title_prefix
        title += vis_variable
        #title += ", t="+str(round(i*dt/(60*60), 3))+" h"
        title += ", t="+str(round(i*dt, 3))+" sec"
        vis.set_title(title)
    
    
    def do_gui_plots(num_timestep, gui_block = True):
        
        var, simgridinfo = plot_get_data(simconfig.vis_variable)
        vis.update_plots(simgridinfo, var)
        
        plot_update_title(num_timestep)
        vis.show(block=gui_block)
    
    def get_simtime_str(simtime):
        return "{:012.4f}".format(simtime)

    
    def do_file_plots(num_timestep, simtime):
        for varname in ['u', 'w', 'p', 'rho', 't', 'p_diff', 'rho_diff', 't_diff']:
            
            simtime_str = get_simtime_str(simtime)
            
            var_data, simgridinfo = plot_get_data(varname)
            vis.update_plots(simgridinfo, var_data)
            
            plot_update_title(num_timestep, varname, title_prefix=simconfig.ns_type+"\n")
            
            filename = simconfig.output_plot_filename
            filename = filename.replace("VARNAME", varname)
            filename = filename.replace("TIMESTEP", str(num_timestep).zfill(10))
            filename = filename.replace("SIMTIME", simtime_str)
            print("Plotting to '"+filename+"'")
            vis.savefig(filename)
    
    
    def do_file_pickle(num_timestep, simtime):
        for var_name in ['u', 'w', 'p', 'rho', 't', 'p_diff', 'rho_diff', 't_diff']:
            
            simtime_str = get_simtime_str(simtime)
            
            var_data, var_gridinfo = plot_get_data(var_name)
            vis.update_plots(var_gridinfo, var)
            
            pickle_data = {
                'var_name': var_name,
                'var_data': var_data,
                'var_gridinfo': var_gridinfo,
                'state_dt': dt,
                'state_num_timestep': num_timestep,
                'state_simtime': simtime,
                'simconfig': simconfig,
            }
            
            filename = simconfig.output_pickle_filename
            filename = filename.replace("VARNAME", var_name)
            filename = filename.replace("TIMESTEP", str(num_timestep))
            filename = filename.replace("SIMTIME", simtime_str)
            print("Saving data to '"+filename+"'")
            
            import pickle
            
            with open(filename, 'wb') as file:
                pickle.dump(pickle_data, file)
    
    
    if simconfig.gui:
        var, simgridinfo = plot_get_data(simconfig.vis_variable)
        vis.update_plots(simgridinfo, var)
        plot_update_title(0)
    
        vis.show(block=False)
        vis.show(block=False)



import time
time_start = time.time()

if simconfig.sim_time != None:
    num_total_timesteps = int(simconfig.sim_time/dt)
else:
    num_total_timesteps = None




print("Run simulation with dt="+str(dt))

output_plot_simtime_interval_next_output = 0
num_timestep = 0
simtime = 0

U_leapfrog_prev = None

while True:
    if simconfig.output_text_freq > 0:
        if num_timestep % simconfig.output_text_freq == 0:
    
            if simconfig.verbosity >= 1:
                print("timestep: "+str(num_timestep))
                print(" + simtime: "+str(simtime))
     
            if simconfig.verbosity >= 2:
                for varname in ['u', 'w', 'p', 'rho', 't']:
                    var = simpde.get_var(U, varname)
                    min_ = np.min(np.abs(var.data))
                    max_ = np.max(np.abs(var.data))
                    print(" + "+varname+": "+str(min_)+" / "+str(max_))
    
    
    do_output = False
    if simconfig.output_plot_timesteps_interval > 0:
        if num_timestep % simconfig.output_plot_timesteps_interval == 0:
            do_output = True
            
    if simconfig.output_plot_simtime_interval > 0:
        if simtime >= output_plot_simtime_interval_next_output:
            do_output = True
            
            while output_plot_simtime_interval_next_output <= simtime:
                output_plot_simtime_interval_next_output += simconfig.output_plot_simtime_interval
        
    if do_output:
        if simconfig.gui:
            do_gui_plots(num_timestep, False)
        
        if simconfig.output_plot_filename != "":
            do_file_plots(num_timestep, simtime)

        if simconfig.output_pickle_filename != "":
            do_file_pickle(num_timestep, simtime)

    
    if num_total_timesteps != None:
        if num_timestep >= num_total_timesteps:
            break
    
    if simconfig.time_integrator == "leapfrog":
        U_backup = U

        U = libpdefd.tools.time_integrator_leapfrog(simpde.dU_dt, U, dt, U_leapfrog_prev)
        
        U_leapfrog_prev = U_backup
    else:
        U = libpdefd.tools.time_integrator(simconfig.time_integrator, simpde.dU_dt, U, dt)
    
    num_timestep += 1
    simtime += dt
    
    if simconfig.timestep_sleep > 0:
        time.sleep(simconfig.timestep_sleep)
    
    if np.isnan(U[0][0,0]):
        raise Exception("NaN detected")
    


time_end = time.time()
print("Time: "+str(time_end-time_start))

if simconfig.gui:
    do_gui_plots(num_timestep, True)

if simconfig.output_plot_filename != "":
    do_file_plots(num_timestep, simtime)
