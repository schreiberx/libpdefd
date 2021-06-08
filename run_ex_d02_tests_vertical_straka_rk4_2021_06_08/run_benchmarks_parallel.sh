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

./list_benchmarks.py || exit 1

./list_benchmarks.py | parallel $ARGS

