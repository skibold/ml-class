#! /bin/bash

function stats() { 
	accuracy=$(cat $1 | sed 's/.*,\(.*\)]/\1/') 
	len=$(echo $accuracy | wc -w) 
	avg=0 
	for a in $accuracy 
	do 
		avg=$(echo "scale=8; $a+$avg" | bc) 
	done 
	echo "scale=8; $avg/$len" | bc 
} 

for ((i=0; i<100; i++)) 
do 
	python fraction_xy.py x_test.csv y_test.csv .1 
	python myproj.py x_test__10.csv y_test__10.csv myproj7.log python 
	gnb.py x_test__10.csv y_test__10.csv gnb7.log 
done 

stats myproj7.log 
stats gnb7.log 
