#! /usr/bin/env python3

import sys
import time
import numpy as np
import libpdefd
import libpdefd.time.time_integrators as time_integrators
import libpdefd.pdes.navierstokes as pde_navierstokes
import libpdefd.pdes.navierstokes_benchmarks as pde_navierstokes_benchmarks


"""
Configuration of simulation    
"""
simconfig = pde_navierstokes.SimConfig(num_dims=2)
simconfig.update()


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
    raise Exception("Navier-Stokes equation of type '"+simconfig.ns_type+"' not supported")


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

p_t0 = simpde.get_var(variable_set_background, "p")
rho_t0 = simpde.get_var(variable_set_background, "rho")
t_t0 = simpde.get_var(variable_set_background, "t")
pot_t_t0 = simpde.get_var(variable_set_background, "pot_t")


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
    vis = libpdefd.visualization.Visualization2DMesh(
        vis_dim_x = simconfig.vis_dim_x,
        vis_dim_y = simconfig.vis_dim_y,
        vis_slice = simconfig.vis_slice,
        rescale = 1.0
    )
    
    vis_contours = None
    
    
    if simconfig.plot_contour_info is not None:
        
        vis_contours = np.arange(simconfig.plot_contour_info[0], simconfig.plot_contour_info[1], simconfig.plot_contour_info[2])
        
        # Remove contour around 0
        vis_contours = np.delete(vis_contours, np.where(np.isclose(vis_contours, 0)))
        
        print("Visualization contours: "+str(vis_contours))
    
    
    def plot_get_data(variable_name):
        
        if variable_name in ['u', 'w', 'rho', 't', 'p', 'pot_t']:
            variable = simpde.get_var(U, variable_name)
            vargridinfo = simgridinfondset[variable_name]
        
        elif variable_name == "p_diff":
            variable = simpde.get_var(U, "p") - p_t0
            vargridinfo = simgridinfondset["p"]
        
        elif variable_name == "rho_diff":
            variable = simpde.get_var(U, "rho") - rho_t0
            vargridinfo = simgridinfondset["rho"]
        
        elif variable_name == "t_diff":
            variable = simpde.get_var(U, "t") - t_t0
            vargridinfo = simgridinfondset["t"]
        
        elif variable_name == "pot_t_diff":
            variable = simpde.get_var(U, "pot_t") - pot_t_t0
            vargridinfo = simgridinfondset["pot_t"]
            
        else:
            raise Exception("variable_name "+str(variable_name)+" not supported")
        
        return variable, vargridinfo

    
    def plot_update_title(i, vis_variable = None, title_prefix=""):
        if vis_variable == None:
            vis_variable = simconfig.vis_variable
        
        title = title_prefix
        title += vis_variable
        #title += ", t="+str(round(i*dt/(60*60), 3))+" h"
        title += ", t="+str(round(i*dt, 3))+" sec"
        vis.set_title(title)
    
    
    def do_gui_plots(num_timestep, gui_block = True):
        
        var_data, simgridinfo = plot_get_data(simconfig.vis_variable)
        vis.update_plots(simgridinfo, var_data.to_numpy_array(), vis_contours)
        
        plot_update_title(num_timestep)
        vis.show(block=gui_block)
    
    def get_simtime_str(simtime):
        return "{:013.4f}".format(simtime)

    
    def do_file_plots(num_timestep, simtime):
        for varname in ['u', 'w', 'p', 'rho', 't', 'p_diff', 'rho_diff', 't_diff']:
            
            simtime_str = get_simtime_str(simtime)
            
            var_data, simgridinfo = plot_get_data(varname)
            vis.update_plots(simgridinfo, var_data.to_numpy_array(), vis_contours)
            
            plot_update_title(num_timestep, varname, title_prefix=simconfig.ns_type+"\n")
            
            filename = simconfig.output_plot_filename
            filename = filename.replace("VARNAME", varname)
            filename = filename.replace("TIMESTEP", str(num_timestep).zfill(10))
            filename = filename.replace("SIMTIME", simtime_str)
            print("Plotting to '"+filename+"'")
            vis.savefig(filename)
    
    
    def do_file_pickle(num_timestep, simtime):
        for var_name in ['u', 'w', 'p', 'rho', 't', 'p_diff', 'rho_diff', 't_diff', 'pot_t', 'pot_t_diff']:
            
            simtime_str = get_simtime_str(simtime)
            
            var_data, var_gridinfo = plot_get_data(var_name)
            
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
        
        
        var_data, simgridinfo = plot_get_data(simconfig.vis_variable)
        vis.update_plots(simgridinfo, var_data.to_numpy_array(), vis_contours)
        plot_update_title(0)
    
        vis.show(block=False)
        vis.show(block=False)


if simconfig.num_timesteps == None:
    if simconfig.sim_time != None:
        simconfig.num_timesteps = int(round(simconfig.sim_time/dt))
    else:
        simconfig.num_timesteps = None




print("Run simulation with dt="+str(dt))

output_plot_simtime_interval_next_output = 0
num_timestep = 0
simtime = 0


"""
Setup time integrator
"""
class time_deriv:
    def comp_du_dt(self, i_U, i_timestamp, i_dt):
        return simpde.dU_dt(i_U)

time_integrator = time_integrators.TimeIntegrators(
        time_integration_method = simconfig.time_integration_method,
        diff_eq_methods = time_deriv(),
        time_integration_order = simconfig.time_integration_order,
        leapfrog_ra_filter_value = simconfig.time_leapfrog_ra_filter_value,
    )


import time
time_start = time.time()


while True:
    
    if simconfig.output_text_freq > 0:
        if num_timestep % simconfig.output_text_freq == 0:
        
            print("timestep: "+str(num_timestep))
            print(" + simtime: "+str(simtime))
            
            time_current = time.time()
            
            if num_timestep != 0:
                seconds_per_timestep = (time_current - time_start)/num_timestep
                print(" + seconds_per_timestep: "+str(seconds_per_timestep))
            
            if simconfig.verbosity >= 2:
                for varname in ['u', 'w', 'p', 'rho', 't']:
                    var = simpde.get_var(U, varname)
                    min_ = var.reduce_minabs()
                    max_ = var.reduce_maxabs()
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

    
    if simconfig.num_timesteps != None:
        if num_timestep >= simconfig.num_timesteps:
            break
        
    U = time_integrator.time_integration_method.comp_time_integration(U, dt, dt*num_timestep)
    #U = libpdefd.tools.time_integrator("rk4", simpde.dU_dt, U, dt)
    
    num_timestep += 1
    simtime += dt
    
    if simconfig.timestep_sleep > 0:
        time.sleep(simconfig.timestep_sleep)
    
    if simconfig.stop_nan_simulation:
        if np.isnan(U[0][0,0]):
            raise Exception("NaN detected")
    


print("*")
print("Simulation finished")
print("*")
time_end = time.time()
total_time = time_end-time_start
print("Time: "+str(total_time))

total_seconds_per_timestep = (time_end - time_start)/float(num_timestep)
print(" + total_time: "+str(total_time))
print(" + number_of_timesteps: "+str(num_timestep))
print(" + total_seconds_per_timestep: "+str(total_seconds_per_timestep))

if simconfig.gui:
    do_gui_plots(num_timestep, True)

if simconfig.output_plot_filename != "":
    do_file_plots(num_timestep, simtime)
