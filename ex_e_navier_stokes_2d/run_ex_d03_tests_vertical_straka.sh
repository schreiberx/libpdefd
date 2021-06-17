#! /bin/bash
BENCHMARK_NAME=vertical_straka


#if true; then
if false; then
	TIME_INTEGRATOR=erk
	TIME_INTEGRATION_ORDER=4
	LEAPFROG_RA_FILTER_VALUE=0
	DT_SCALING=0.001
fi

if true; then
#if false; then
	TIME_INTEGRATOR=leapfrog
	TIME_INTEGRATION_ORDER=2
	LEAPFROG_RA_FILTER_VALUE=0.01
	#LEAPFROG_RA_FILTER_VALUE=0.00
	DT_SCALING=0.001
fi

NS_TYPE=nonlinear_a_grid__p_rho
#NS_TYPE=nonlinear_a_grid__p_t
#NS_TYPE=nonlinear_a_grid__rho_t



ARGS=""
ARGS+=" -v 10"
ARGS+=" --sim-time=900"
ARGS+=" --time-integration-method=$TIME_INTEGRATOR"
ARGS+=" --time-integration-order=$TIME_INTEGRATION_ORDER"
ARGS+=" --time-leapfrog-ra-filter-value=$LEAPFROG_RA_FILTER_VALUE"
ARGS+=" --dt-scaling=$DT_SCALING"
ARGS+=" --min-spatial-approx-order=4"
ARGS+=" --ns-type=$NS_TYPE"
ARGS+=" --benchmark-name=$BENCHMARK_NAME"
#ARGS+=" --cell-res 1024 1024"
ARGS+=" --cell-res 256 64"
#ARGS+=" --cell-res 14 12"

ARGS+=" --output-text-freq=100"
ARGS+=" --output-plot-simtime-interval=5.0"


if true; then
#if false; then
	ARGS+=" --gui 1"
	ARGS+=" --vis-variable=pot_t_diff"
fi

#if true; then
if false; then
	ARGS+=" --output-plot-filename=output_plot__${BENCHMARK_NAME}__${NS_TYPE}__${TIME_INTEGRATOR}__VARNAME_TIMESTEP.pdf"
fi

EXEC="python ./ex_d01_navier_stokes_compressible_2d_horizontal_vertical.py"
echo $EXEC $ARGS $@
$EXEC $ARGS $@ || exit 1
