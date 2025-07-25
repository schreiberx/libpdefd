#! /bin/bash


#for ns_type in nonlinear_a_grid__p_rho nonlinear_a_grid__rho_t nonlinear_a_grid__p_t; do
for ns_type in nonlinear_a_grid__p_rho; do
	OUTPUT_FILENAME="output_VARNAME_TIMESTEP.pdf"

	ARGS=""

	ARGS+=" --sim-time=20"
	#ARGS+=" --dt=0.1"
        ARGS+=" --dt-scaling=0.001"
	ARGS+=" --min-spatial-approx-order=2"
	ARGS+=" --ns-type=$ns_type"
	ARGS+=" --benchmark-name=horizontal_bump"

	ARGS+=" -v 10"
	ARGS+=" --gui 1"
	ARGS+=" --vis-variable=t"
	ARGS+=" --output-text-freq=10"
	ARGS+=" --output-plot-simtime-interval=0.01"
	#ARGS+=" --output-plot-freq=0"
	#ARGS+=" --output-plot-filename=output_plot__""$ns_type""__VARNAME_TIMESTEP.pdf"

	EXEC="python ex_d01_navier_stokes_compressible_2d_horizontal_vertical.py"
	echo $EXEC $ARGS $@
	$EXEC $ARGS || exit 1
done
