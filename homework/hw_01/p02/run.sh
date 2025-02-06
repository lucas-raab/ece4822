#!/bin/bash


. ~/.bashrc



for N in $(seq 1000 200 100000)
do

    ./p02.exe $N 1000 100
done
