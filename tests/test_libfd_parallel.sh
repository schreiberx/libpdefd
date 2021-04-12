#! /bin/bash


ARGS=""
ARGS+=" --halt now,fail=1"
ARGS+=" --eta"
ARGS+=" --bar"
ARGS+=" --progress"
#ARGS+=" ::: "

python test_libfd.py print_parallel_jobs

python test_libfd.py print_parallel_jobs | parallel $@ $ARGS

