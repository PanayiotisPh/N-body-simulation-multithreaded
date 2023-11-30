#!/bin/bash
rm a.out
rm ./outputs/*
# Number of Steps
 for ((i=100; i<=100; i+=100 ))
	do
	#Number of Bodies
	for ((j=5; j<=200; j+=50 ))
		do
			gcc -Werror -Wall -O3 -lm n-body_std.c
				./a.out $i $j
			gcc -DDEBUG -Werror -Wall -O3 -lm n-body_std.c
				./a.out $i $j 2>> ./outputs/collisions.txt
			gnuplot -e "numberOfBodies  = $j; outputFile = './outputs/output-$i-$j.gif'" plot3D.gpl
		done
	done
	