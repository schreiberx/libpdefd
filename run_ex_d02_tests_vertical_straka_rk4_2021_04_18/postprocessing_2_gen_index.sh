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

	#./$0 print_parallel_jobs || exit 1

	./$0 print_parallel_jobs | parallel $ARGS

	exit 0
fi


if [ "#$1" == "#print_parallel_jobs" ]; then
	for DIR in output_plot__bench_*__nstype_*__ti_*__order_*/; do
		for VARNAME in u w p rho t p_diff rho_diff t_diff pot_t pot_t_diff; do
		#for VARNAME in t_diff; do
			OUTPUT_PREFIX=""

			C=0
			for FILE in ${DIR}/output_${VARNAME}_[0-9]*.png; do
				CSTR=$(printf "%08d" $C)
				F=$(basename "$FILE")
				echo ln -sf "$F" "${DIR}/output_idx_${VARNAME}_${CSTR}.png"
				C=$((C+1))
			done
		done
	done
	exit 0
fi

echo "ERROR"
exit 1

