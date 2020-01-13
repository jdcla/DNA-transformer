 #!/bin/bash

source activate DNA_trans_XL

data_path='data/eco_regulon/eco_regulon_70.npy'
#data_path='data/eco_TIS/eco_TIS.npy'
#data_path='data/eco_regulon/eco_regulon_70.npy'
d_embed=32
d_head=32
d_inner=128
d_model=32
n_head=6
n_layer=6
dropout=0.1
tgt_len=512
output="output/"

echo 'Run training...'
python train.py \
    --cuda \
    --output_path $output \
    --eval-interval 3000 \
    --data_path $data_path \
    --d_embed $d_embed \
    --n_layer $n_layer \
    --d_model $d_model \
    --n_head $n_head \
    --d_head $d_head \
    --d_inner $d_inner \
    --dropout $dropout \
    --dropatt 0 \
    --optim adam \
    --lr 0.0002 \
    --max_step 100000 \
    --warmup_step 4000 \
    --tgt_len $tgt_len \
    --mem_len 0 \
    --eval_tgt_len $tgt_len \
    --batch_size 10 \
    --shift 20 \
    --gpu0_bsz 0 \
    --not_tied \
    --varlen
