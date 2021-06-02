#! /bin/bash

BENCHMARK_NAME=vertical_straka

TIME_INTEGRATION_METHOD=leapfrog
TIME_INTEGRATION_ORDER=2


for SPACE_ORDER in 2 4 6 8; do

	for NS_TYPE in nonlinear_a_grid__p_rho; do
	#for NS_TYPE in nonlinear_a_grid__p_rho nonlinear_a_grid__rho_t nonlinear_a_grid__p_t; do
		for RA_FILTER in 0.0 0.01 0.02 0.04 0.05; do
			ARGS=""
			ARGS+=" -v 10"
			ARGS+=" --gui 0"
			ARGS+=" --vis-variable=t_diff"
			ARGS+=" --output-text-freq=10"
			ARGS+=" --output-plot-simtime-interval=5"
			ARGS+=" --sim-time=900"
			
			# Higher viscosity
			ARGS+=" --const-hyperviscosity-all-order=2"
			ARGS+=" --const-hyperviscosity-all=0"

			#ARGS+=" --dt-scaling=0.0003"
			# Time step size from Straka paper for 25m resolution
			ARGS+=" --dt=0.015625"

			ARGS+=" --time-integration-method=$TIME_INTEGRATION_METHOD"
			ARGS+=" --time-integration-order=$TIME_INTEGRATION_ORDER"
			ARGS+=" --time-leapfrog-ra-filter-value=$RA_FILTER"

			# Use a 2 times smaller one
			#ARGS+=" --dt=0.0078125"

			# Use a 4 times smaller one
			#ARGS+=" --dt=0.00390625"

			ARGS+=" --min-spatial-approx-order=$SPACE_ORDER"
			ARGS+=" --ns-type=$NS_TYPE"
			ARGS+=" --benchmark-name=$BENCHMARK_NAME"

			# Convergence at 25m
			# Domain size: 25.6km x 6.4km
			RX=1024
			RX=$((RX*1))
			RY=$((RX/2))
			ARGS+=" --cell-res $RX $RY"
			#ARGS+=" --cell-res 128 128"

			PREFIX="output_plot__bench_${BENCHMARK_NAME}__nstype_${NS_TYPE}__ti_${TIME_INTEGRATION_METHOD}__order_${SPACE_ORDER}"
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
done

