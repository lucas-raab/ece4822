#!/bin/bash
. ~/.bashrc

for N in $(seq 50 50 2000); do
./p01.exe $N $N 1000
	 
done
