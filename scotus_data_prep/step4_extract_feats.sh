#!/bin/bash

step=0
nj=20
base_outfolder=/PATH/TO/BASE_OUTFOLDER
scotus_veri_dir=$base_outfolder/veri_data
scotus_diar_dir=$base_outfolder/diar_data

if [ $step -le 0 ]; then
    #make the feats
    utils/fix_data_dir.sh $scotus_veri_dir
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj \
             --cmd run.pl $scotus_veri_dir
    
    utils/fix_data_dir.sh $scotus_veri_dir
    sid/compute_vad_decision.sh --nj $nj --cmd run.pl $scotus_veri_dir
    utils/fix_data_dir.sh $scotus_veri_dir
fi

if [ $step -le 1 ]; then
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd run.pl \
    $scotus_veri_dir ${scotus_veri_dir}_nosil $scotus_veri_dir/nosil_feats
    utils/fix_data_dir.sh ${scotus_veri_dir}_nosil
fi

if [ $step -le 2 ]; then
    #make the feats
    utils/fix_data_dir.sh $scotus_diar_dir
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj \
             --cmd run.pl $scotus_diar_dir
    
    utils/fix_data_dir.sh $scotus_diar_dir
    sid/compute_vad_decision.sh --nj $nj --cmd run.pl $scotus_diar_dir
    utils/fix_data_dir.sh $scotus_diar_dir
fi

if [ $step -le 3 ]; then
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd run.pl \
    $scotus_diar_dir ${scotus_diar_dir}_nosil $scotus_diar_dir/nosil_feats
    utils/fix_data_dir.sh ${scotus_diar_dir}_nosil
fi