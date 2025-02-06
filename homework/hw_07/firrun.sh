#!/bin/bash


source /home/tuo73004/.bashrc


for N in $(seq 1000 1000 100000); do

    /home/tuo73004/ece_4822/homework/hw_07/cuda/fir.exe $N 500

done
