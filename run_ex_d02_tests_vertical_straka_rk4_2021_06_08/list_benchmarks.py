#! /usr/bin/env python

import os
import sys


def gen_run(
        benchmark_name,
        ns_type,
        spatial_order,
        time_integration_method,
        time_integration_order,
        ra_filter = None,
        res_scaling_factor = 1,
        dt_scaling_factor = 1,
):
    args = []
    args += ["-v", "10"]
    args += ["--gui=0"]
    args += ["--vis-variable=t_diff"]
    args += ["--output-text-freq=10"]
    args += ["--output-plot-simtime-interval=5"]
    args += ["--sim-time=900"]
    
    # Higher viscosity
    args += ["--const-hyperviscosity-all-order=2"]
    args += ["--const-hyperviscosity-all=0"]

    args += ["--min-spatial-approx-order="+str(spatial_order)]
    args += ["--ns-type="+ns_type]
    args += ["--benchmark-name="+benchmark_name]

    
    # Time step size from Straka paper for 25m resolution
    dt = 0.015625*dt_scaling_factor
    args += ["--dt="+str(dt)]

    # Convergence at 25m
    # Domain size (half of it!): 25.6km x 6.4km
    resx = 1024 * res_scaling_factor
    resx = resx*1
    resy = resx//4

    args += [" --cell-res", str(resx), str(resy)]
    #args+=" --cell-res 128 128"


    args += ["--time-integration-method="+time_integration_method]
    args += ["--time-integration-order="+str(time_integration_order)]
    if ra_filter != None:
        args += ["--time-leapfrog-ra-filter-value="+str(ra_filter)]


    prefix = "output_plot"
    prefix += "__"
    prefix += "bench_"+benchmark_name
    prefix += "__"
    prefix += "nstype_"+ns_type
    prefix += "__"
    prefix += "ti_"+time_integration_method
    prefix += "__"
    prefix += "tord_"+str(time_integration_order)

    if ra_filter != None:
        prefix += "__"
        prefix += "ra_"+("{:.02f}".format(ra_filter))

    prefix += "__"
    prefix += "sord_"+str(spatial_order)
    prefix += "__"
    prefix += "resx_"+("{:04g}".format(resx))
    prefix += "__"
    prefix += "resy_"+("{:04g}".format(resy))

    prefix += "__"
    prefix += "dt_"+("{:.08f}".format(dt))

    try:
        os.mkdir(prefix)
    except:
        pass

    #output_filename_plot="${prefix}/output_varname_simtime.pdf"
    #args+=" --output-plot-filename=$output_filename_plot"

    output_filename_pickle=prefix+"/output_varname_simtime.pickle"
    args += ["--output-pickle-filename="+output_filename_pickle]

    args = ['python', "../ex_e_navier_stokes_2d/ex_d01_navier_stokes_compressible_2d_horizontal_vertical.py"] + args
    return args

benchmark_name="vertical_straka"

ns_type_ = ['nonlinear_a_grid__p_rho', 'nonlinear_a_grid__rho_t', 'nonlinear_a_grid__p_t']
ns_type_ = ['nonlinear_a_grid__p_rho']

spatial_order_ = [2, 4, 6, 8]
spatial_order_ = [2, 4, 6]

for spatial_order in spatial_order_:
    for ns_type in ns_type_:
        for time_integration_method in ["erk", "leapfrog"]:

            for dt_scaling_factor in [1, 0.5]:
                for res_scaling_factor in [1, 2]:

                    if time_integration_method == "erk":

                        for time_integration_order in [1, 2, 4]:
                            args = gen_run(
                                benchmark_name,
                                ns_type,
                                spatial_order,
                                time_integration_method,
                                time_integration_order,

                                res_scaling_factor = res_scaling_factor,
                                dt_scaling_factor = dt_scaling_factor
                            )

                            print(" ".join(args))

                    elif time_integration_method == "leapfrog":

                        for ra_filter in [0.0, 0.01, 0.02, 0.04, 0.05]:
                            args = gen_run(
                                benchmark_name,
                                ns_type,
                                spatial_order,
                                time_integration_method,
                                time_integration_order = 2,
                                ra_filter = ra_filter,
                                
                                res_scaling_factor = res_scaling_factor,
                                dt_scaling_factor = dt_scaling_factor
                            )

                            print(" ".join(args))

