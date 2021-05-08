#! /bin/bash
BENCHMARK_NAME=vertical_straka

TIME_INTEGRATOR=rk4

NS_TYPE=nonlinear_a_grid__p_rho
#NS_TYPE=nonlinear_a_grid__p_t
#NS_TYPE=nonlinear_a_grid__rho_t



ARGS=""
ARGS+=" -v 10"
ARGS+=" --sim-time=900"
ARGS+=" --time-integrator $TIME_INTEGRATOR"
ARGS+=" --dt-scaling=0.001"
ARGS+=" --min-spatial-approx-order=4"
ARGS+=" --ns-type=$NS_TYPE"
ARGS+=" --benchmark-name=$BENCHMARK_NAME"
#ARGS+=" --cell-res 1024 1024"
ARGS+=" --cell-res 256 64"

ARGS+=" --output-text-freq=10"
ARGS+=" --output-plot-simtime-interval=0.1"


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
