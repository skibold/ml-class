#! /bin/bash

T=$1
arg3="../proj1/x_test.csv"
arg4="../proj1/y_test.csv"
log="proj1boosted_${T}.log"

rm -f $log 2>/dev/null
for ((i=0; i<100; i++))
do
	arg1="../proj1/x_test_${i}_10.csv"
	arg2="../proj1/y_test_${i}_10.csv"
	python Proj1Boosted.py $arg1 $arg2 $arg3 $arg4 $T >> $log
done

acc=($(grep accuracy $log | cut -d '=' -f 2 | bc))
avg=0
for a in ${acc[@]}
do
	avg=$(echo "$avg + $a" | bc)
done
avg=$(echo "scale=8; $avg / ${#acc[@]}" | bc)
echo "${#acc[@]} examples, T = $T, avg accuracy = $avg"
