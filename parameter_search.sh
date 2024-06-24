#!/bin/bash
#for gnnlr in 1e-4 1e-3 1e-2 1e-5
##1e-4 1e-3 1e-2
#do
#    python run_GNN.py --gnnlr=$gnnlr
#done

for seed in 0 42 212330 2294892 990624
do
for nth_fold in 0 1 2  4
        do
        python run_GNN.py --seed=$seed --nth_fold=$nth_fold
        done
    done
#for nth_fold in 0 1 2 3 4
#do
#  python run_GNN.py --nth_fold=$nth_fold
#done