#!/bin/bash


source /home/tuo73004/.bashrc


for N in $(seq 1000 1000 100000); do

	./iir.exe $N $N 500

done
