#!/bin/bash

REPO=/home/jcampos/repos/HTV-Learn
num_array=(40 60 85 125 180 260 380 550)
arg_str=""

for i in ${num_array[@]};
do
    arg_str+="output/nn/nn_6000_wd_1e-6/nn_cut_face_gaps_seed_10_"
    arg_str+="num_train_6000_hidden_${i}_wd_1e-6 "
done

python3 ${REPO}/scripts/plot_htv_vs_parameter.py num_hidden_neurons ${arg_str}
