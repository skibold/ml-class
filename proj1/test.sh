#!/bin/bash
for ((i=0; i<100; i++))
do
	python fraction_xy.py x_test.csv y_test.csv .1
	python myproj.py x_test__10.csv y_test__10.csv myproj.log
	python gnb.py x_test__10.csv y_test__10.csv gnb.log
done

accuracy=$(cat myproj.log | sed 's/.*,\(.*\)]/\1/')
len=$(echo $accuracy | wc -w)
avg=0
for a in $accuracy
do
	avg=$(echo "scale=8; $a+$avg" | bc)
done
echo "scale=8; $avg/$len" | bc