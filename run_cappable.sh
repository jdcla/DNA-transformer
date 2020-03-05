#!/bin/bash

dataset="data/eco_capseq/eco_rm_capseq.npy"
d_head=32
d_model=32
n_head=6
n_layer=6
dropout=0.10
merge_size=1
tgt_len=512
conv_size=7
output="output/"
echo 'Run training...'

echo $dataset
python train.py --cuda \
    --eval-interval 3000 \
    --data_path $dataset \
    --output_path $output \
    --d_embed $d_model \
    --n_layer $n_layer \
    --d_model $d_model \
    --n_head $n_head \
    --d_head $d_head \
    --dropout $dropout \
    --dropatt 0 \
    --optim adam \
    --lr 0.0001 \
    --max_step 85000 \
    --warmup_step 3000 \
    --tgt_len $tgt_len \
    --mem_len 0 \
    --varlen \
    --eval_tgt_len $tgt_len \
    --batch_size 10 \
    --shift 20 \
    --ext_ds 0 \
    --gpu0_bsz 0 \
    --conv_size $conv_size \
    --not_tied \
    --work_dir_path $output
