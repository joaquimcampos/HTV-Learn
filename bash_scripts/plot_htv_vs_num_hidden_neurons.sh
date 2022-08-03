#!/bin/bash

REPO=/home/jcampos/repos/HTV-Learn

args="num_hidden_neurons "
args+=$(ls -d $REPO/output/saved_models/nn/* | xargs echo)
args+=" --cmp_log_dirs "
args+="$REPO/output/saved_models/rbf_cut_face_gaps_seed_10_num_train_"\
"6000_eps_13.0_lmbda_1e-3 "
args+="$REPO/output/saved_models/htv_cut_face_gaps_seed_10_num_train_"\
"6000_lmbda_1e-2"

python3 "$REPO"/scripts/plot_htv_vs_parameter.py $args
