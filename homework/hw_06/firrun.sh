#!/bin/bash


source /home/tuo73004/.bashrc


for N in $(seq 1000 1000 15000); do
    for J in $(seq 29 -4 1); do
	export OMP_NUM_THREADS=$J
	./fir.exe $N 300
    done
done
