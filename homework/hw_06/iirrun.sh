#!/bin/bash


source /home/tuo73004/.bashrc


for N in $(seq 100 500 14000); do
    for J in $(seq 29 -4 1); do
	export OMP_NUM_THREADS=$J
	./iir.exe $N $N 50
    done
done
