#!/bin/bash

step=0
scotus_data_dir=/PATH/TO/SCOTUS_DATA_DIR
scotus_nosil_data=/PATH/TO/SCOTUS_NOSIL_DATA

if [ $step -le 0 ]; then
    #make the feats
    utils/fix_data_dir.sh $scotus_data_dir
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 20 \
             --cmd run.pl $scotus_data_dir
    
    utils/fix_data_dir.sh $scotus_data_dir
    sid/compute_vad_decision.sh --nj 20 --cmd run.pl $scotus_data_dir
    utils/fix_data_dir.sh $scotus_data_dir
fi

if [ $step -le 1 ]; then
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj 20 --cmd run.pl \
    $scotus_data_dir $scotus_nosil_data $scotus_data_dir/nosil_feats
    utils/fix_data_dir.sh $scotus_nosil_data
fi