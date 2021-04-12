#! /bin/bash

BENCHMARK_NAME=vertical_bump
TIME_INTEGRATOR=rk4

for NS_TYPE in nonlinear_a_grid__p_rho nonlinear_a_grid__rho_t nonlinear_a_grid__p_t; do
#for ns_type in nonlinear_a_grid__p_rho; do

	ARGS=""
	ARGS+=" -v 10"
	ARGS+=" --sim-time=300"
	ARGS+=" --time-integrator $TIME_INTEGRATOR"
	ARGS+=" --dt-scaling=0.001"
	ARGS+=" --min-spatial-approx-order=4"
	ARGS+=" --ns-type=$NS_TYPE"
	ARGS+=" --benchmark-name=$BENCHMARK_NAME"
	#ARGS+=" --cell-res 1024 1024"
	ARGS+=" --cell-res 128 128"

	ARGS+=" --gui 1"
	ARGS+=" --vis-variable=t"
	ARGS+=" --output-text-freq=10"
	ARGS+=" --output-plot-simtime-interval=0.1"
	#ARGS+=" --output-plot-filename=output_plot__${BENCHMARK_NAME}__${NS_TYPE}__${TIME_INTEGRATOR}__VARNAME_TIMESTEP.pdf"

	EXEC="python ex_d01_navier_stokes_compressible_2d_horizontal_vertical.py"
	echo $EXEC $ARGS $@
	$EXEC $ARGS $@ || exit 1
done
