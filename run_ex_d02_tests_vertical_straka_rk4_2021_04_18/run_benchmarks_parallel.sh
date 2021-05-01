#! /bin/bash


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

./run_benchmarks.sh print_parallel_jobs || exit 1

./run_benchmarks.sh print_parallel_jobs | parallel $ARGS

