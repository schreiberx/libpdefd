#! /bin/bash

for i in ex_?_*/ex_*.py; do

	echo "************************************"
	echo "$i"
	echo "************************************"
	python ./$i --test-run=1 || break
done
