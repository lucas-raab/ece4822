#!/bin/bash

for N in $(seq 100 100 2100)
do

    ./hw_03.exe $N $N 500
done
