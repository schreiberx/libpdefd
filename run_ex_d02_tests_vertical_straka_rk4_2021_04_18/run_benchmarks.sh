#! /bin/bash

BENCHMARK_NAME=vertical_straka
TIME_INTEGRATOR=rk4

for ORDER in 2 4 6 8 10; do

	for NS_TYPE in nonlinear_a_grid__p_rho; do
	#for NS_TYPE in nonlinear_a_grid__p_rho nonlinear_a_grid__rho_t nonlinear_a_grid__p_t; do
		ARGS=""
		ARGS+=" -v 10"
		ARGS+=" --gui 0"
		ARGS+=" --vis-variable=t_diff"
		ARGS+=" --output-text-freq=10"
		ARGS+=" --output-plot-simtime-interval=5"
		ARGS+=" --sim-time=900"

		#ARGS+=" --dt-scaling=0.0003"
		# Time step size from Straka paper for 25m resolution
		#ARGS+=" --dt=0.015625"

		# Use a 2 times smaller one
		#ARGS+=" --dt=0.0078125"

		# Use a 4 times smaller one
		ARGS+=" --dt=0.00390625"

		ARGS+=" --min-spatial-approx-order=$ORDER"
		ARGS+=" --ns-type=$NS_TYPE"
		ARGS+=" --benchmark-name=$BENCHMARK_NAME"

		# Convergence at 25m
		# Domain size: 25.6km x 6.4km
		RX=1024
		RX=$((RX*2))
		RY=$((RX/4))
		ARGS+=" --cell-res $RX $RY"
		ARGS+=" --time-integrator $TIME_INTEGRATOR"
		#ARGS+=" --cell-res 128 128"

		PREFIX="output_plot__bench_${BENCHMARK_NAME}__nstype_${NS_TYPE}__ti_${TIME_INTEGRATOR}__order_${ORDER}"
		mkdir -p "$PREFIX"

		#OUTPUT_FILENAME_PLOT="${PREFIX}/output_VARNAME_SIMTIME.pdf"
		#ARGS+=" --output-plot-filename=$OUTPUT_FILENAME_PLOT"

		OUTPUT_FILENAME_PICKLE="${PREFIX}/output_VARNAME_SIMTIME.pickle"
		ARGS+=" --output-pickle-filename=$OUTPUT_FILENAME_PICKLE"

		PROG=""
		EXEC="python ../ex_e_navier_stokes_2d/ex_d01_navier_stokes_compressible_2d_horizontal_vertical.py"
		echo $EXEC $ARGS

		if [ "#$1" != "#print_parallel_jobs" ]; then
			$EXEC $ARGS
		fi
	done
done
