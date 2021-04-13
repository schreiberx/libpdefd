#! /bin/bash

BENCHMARK_NAME=vertical_straka
TIME_INTEGRATOR=rk4

if [ "#$1" == "#" ]; then
	ARGS=""
	ARGS+=" --halt now,fail=1"
	#ARGS+=" --halt never"
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
		for VARNAME in t t_diff; do
			OUTPUT_PREFIX=""

			for INFILE_T in ${DIR}/output_${VARNAME}_[0-9]*.pickle; do

				OUTFILE_T="${INFILE_T/.pickle/.png}"
				EXEC="python ../bin/postprocessing_pickle_pot_temperature.py"

				ARGS=""
				ARGS+=" --input-temperature ${INFILE_T}"
				ARGS+=" --input-pressure ${INFILE_T/_t_/_p_}"
				ARGS+=" --output ${OUTFILE_T/_t_/_pot_t_}"
				ARGS+=" --dpi 300"

				echo "$EXEC $ARGS"
			done
		done
	done
	exit 0
fi

echo "ERROR"
exit 1

