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
#	python fraction_xy.py x_test.csv y_test.csv .1 $i
	python threeclass.py ../proj1/x_test_${i}_10.csv ../proj1/y_test_${i}_10.csv ../proj1/x_test.csv ../proj1/y_test.csv >>threeclass.log 
#	python gnb.py x_test_${i}_10.csv y_test_${i}_10.csv gnb.log 
done 

#stats threeclass.log 
#stats gnb.log 
