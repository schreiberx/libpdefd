#! /bin/bash

BENCHMARK_NAME=vertical_straka
TIME_INTEGRATOR=rk4

if [ "#$1" == "#" ]; then
	ARGS=""
	#ARGS+=" --halt now,fail=1"
	ARGS+=" --halt never"
	ARGS+=" --eta"
	ARGS+=" --bar"
	ARGS+=" --progress"
	#ARGS+=" ::: "

	# Deactivate multithreading
	export MKL_NUM_THREADS=1
	export OMP_NUM_THREADS=1

	./$0 print_parallel_jobs || exit 1

	./$0 print_parallel_jobs | parallel $ARGS

	exit 0
fi


if [ "#$1" == "#print_parallel_jobs" ]; then
	for DIR in output_plot__bench_*__nstype_*__ti_*__order_*/; do
		for VARNAME in u w p rho t p_diff rho_diff t_diff pot_t pot_t_diff; do
			OUTPUT_PREFIX=""

			for INFILE in ${DIR}/output_${VARNAME}_[0-9]*.pickle; do

				OUTFILE="${INFILE/.pickle/.png}"
				EXEC="python ../bin/navierstokes_postprocessing_pickle_output.py"

				ARGS=""
				ARGS+=" --filename ${INFILE}"
				ARGS+=" --output ${OUTFILE}"
				ARGS+=" --dpi 600"
				ARGS+=" --figscale 1.5"

				echo "$EXEC $ARGS"
			done
		done
	done
	exit 0
fi

echo "ERROR"
exit 1

