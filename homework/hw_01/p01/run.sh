#!/bin/bash
. ~/.bashrc

for N in $(seq 100 100 3000); do
./p01.exe $N $N 1000
	 
done
