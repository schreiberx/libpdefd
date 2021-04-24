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
		for VARNAME in u w p rho t p_diff rho_diff t_diff pot_t pot_t_diff; do
		#for VARNAME in t_diff; do
			OUTPUT_PREFIX=""

			ARGS=""
			#ARGS+=" -vcodec libx264"
			ARGS+=" -i ${DIR}/output_idx_${VARNAME}_%08d.png"
			ARGS+=" -framerate 25"
			ARGS+=" -c:v libvpx-vp9"
			ARGS+=" -y"
			ARGS+=" ${DIR/\/}_${VARNAME}_movie.mp4"
			#ARGS+=" ${DIR/\/}_movie.avi"
			echo ffmpeg $ARGS
		done
	done
	exit 0
fi

echo "ERROR"
exit 1

