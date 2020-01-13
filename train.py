# coding: utf-8
import argparse
import time
import datetime
import math
import os
import itertools
import json
import numpy as np
import pandas as pd

from utils.dna_functions import Logger, FocalLoss

import torch
import torch.optim as optim
import torch.nn as nn

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data_path', type=str,
                    help='dataset path')
parser.add_argument('--output_path', type=str, default='output/',
                    help='output path')
parser.add_argument('--emb_path', type=str, default=None,
                    help='embedding path')
parser.add_argument('--n_layer', type=int, default=6,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=4,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=32,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=64,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=256,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--optim_type', type=int, default=0,
                    help='define optimizer type')
parser.add_argument('--shift', type=int, default=20,
                    help='shift labels')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--conv_size', type=int, default=7,
                    help='motif size of the conv layer')
parser.add_argument('--conv_emb', action='store_true',
                    help='convolutional embedding')
parser.add_argument('--pre_conv', action='store_true',
                    help='convolution before QKV')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=200000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--ext_ds', type=int, default=0,
                    help='number of tokens to include downstream of label')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=2222,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--methylation', action='store_true',
                    help='use methylation patterns -> 8 input classes')
args = parser.parse_known_args()[0]
print(args)
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.batch_size % args.batch_chunk == 0


args.work_dir = os.path.join('{}-{}'.format(args.output_path, args.data_path.split('/')[-1]), 
                             time.strftime('%Y%m%d-%H%M%S'))

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################
n_token_out = 2
n_token_in = 4 + 4 * args.methylation
eval_batch_size = args.batch_size

corpus = get_lm_corpus(args.data_path, labels=n_token_out, merge_size=1, shift=args.shift)
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, overlap=args.ext_ds, img=args.conv_emb,
                              device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, overlap=args.ext_ds, img=args.conv_emb,
                              device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, overlap=args.ext_ds, img=args.conv_emb,
                              device=device)
args.coords = [corpus.Gen_train.start, corpus.Gen_train.end, corpus.Gen_test.start, corpus.Gen_test.end,
               corpus.Gen_valid.start, corpus.Gen_valid.end]
print(f'Train set: {args.coords[0]}-{args.coords[1]}\t Test set: {args.coords[2]}-{args.coords[3]}\t Valid set: '
      f'{args.coords[4]}-{args.coords[5]}')

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
tie_projs += [False] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################


def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = MemTransformerLM(n_token_in, n_token_out, args.n_layer, args.n_head, args.d_model,
                             args.d_head, args.d_inner, args.dropout, args.dropatt, args.conv_size, args.conv_emb,
                             args.pre_conv, tie_weight=args.tied, d_embed=args.d_embed, tie_projs=tie_projs, 
                             tgt_len=args.tgt_len, mem_len=args.mem_len, ext_ds=args.ext_ds, 
                             cutoffs=cutoffs, same_length=args.same_length, clamp_len=args.clamp_len)
    model.apply(weights_init)
    model.word_emb.apply(weights_init)  # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk,
                                          model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, dim=1).to(device)
else:
    para_model = model.to(device)


###############################################################################
# Training code
###############################################################################

# optimizer
if args.optim.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

# scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)

elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                else step / (args.warmup_step ** 1.5)


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)

elif args.scheduler == 'constant':
    pass

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')


def evaluate(model, eval_iter, args, criterion=None, weights_optim=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.same_length = True
    model.reset_length(args.eval_tgt_len, args.eval_tgt_len, args.ext_ds)
    
    # Evaluation
    total_len, total_loss = 0, 0.
    probs, targets = [], []
    i_max = eval_iter.n_batch_same_length
    
    with torch.no_grad():
        mems = tuple()
        for i, (data, target_init, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            # Ensures all data is retained to reconstruct original genomic strand
            last = True if (i+1 == i_max) else False
            ret = model(data, target_init, *mems, criterion=criterion, last=last)   
            
            loss, prob, target, mems = ret[0], ret[1], ret[2], ret[3:]
            
            # transforming of data necessary to reconstruct original genomic sequence
            probs.append([p.reshape(-1,args.batch_size,
                                           p.shape[1]).transpose(1,0,2) for p in prob])
            targets.append([t.reshape(-1,args.batch_size).T for t in target])
            
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len
    
    # transforming of data necessary to reconstruct original genomic sequence
    probs = [np.concatenate([p[i] for p in probs], axis=1) for i in range(len(probs[0]))]
    targets = [np.concatenate([t[i] for t in targets], axis=1) for i in range(len(targets[0]))] 
    
    probs = [p.reshape(-1, p.shape[2]) for p in probs]
    targets = [t.reshape(-1) for t in targets]
    
    model.same_length = False
    model.reset_length(args.tgt_len, args.mem_len, args.ext_ds)
    model.train()

    return total_loss / total_len, probs, targets



def train(perf_logger=None, criterion=None, optimizer=optimizer):
    probs = []
    targets = []
    # Turn on training mode which enables dropout.
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    model.train()
    mems = tuple()
    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    for batch, (data, target_init, seq_len) in enumerate(train_iter):
        model.zero_grad()
        ret = para_model(data, target_init, *mems, criterion=criterion)
        loss, logit, target, mems = ret[0], ret[1], ret[2], ret[3:]
        probs.append(logit)
        targets.append(target)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        train_loss += loss.float().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            perf_logger.log_loss(cur_loss, smooth=False)
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.6f}'.format(epoch, train_step,
                                                                 batch + 1, optimizer.param_groups[0]['lr'],
                                                                 elapsed * 1000 / args.log_interval, cur_loss)
            log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            logging(log_str)
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss, val_probs, val_targets = evaluate(model, va_iter, args, criterion=criterion)
            mask = np.full(len(val_targets[0]), True)
            for v_prob, v_target in zip(val_probs, val_targets):
                perf_logger.log_metrics(v_target[mask], v_prob[mask], key='val')
                mask = v_target[mask] == v_target[mask].max()
            perf_logger.log_loss(val_loss, key='val', smooth=False)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.6f}'.format(train_step // args.eval_interval, train_step,
                                                    (time.time() - eval_start_time), val_loss)
            log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            logging(log_str)
            logging('-' * 100)
            print(perf_logger.metrics['val']['auc'][-1], perf_logger.metrics['val']['p-r'][-1])
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug:
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)

            eval_start_time = time.time()

        if train_step == args.max_step:
            break
    # print(targets[0].shape)
    probs = [np.concatenate([p[i] for p in probs]) for i in range(len(probs[0]))]
    targets = [np.concatenate([t[i] for t in targets]) for i in range(len(targets[0]))]
    mask = np.full(len(targets[0]), True)
    for prob, target in zip(probs, targets):
        perf_logger.log_metrics(target[mask], prob[mask], key='train')
        mask = target[mask] == target[mask].max()
    print(perf_logger.metrics['train']['auc'][-len(probs):], perf_logger.metrics['train']['p-r'][-len(probs):])


train_step = 0
train_loss = 0
best_val_loss = None

perf_logger = Logger(['AUC', 'P-R', 'acc'], True, ['val'])
logging = create_exp_dir(args.work_dir, scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)

targets = tr_iter.target.cpu().data.numpy().reshape(-1)
value_counts = pd.Series(targets).value_counts().sort_index()
alpha = list(((value_counts[0]/ value_counts)/(value_counts[0]/ value_counts).max()).values)
args.criterion_gamma = 0.8 # not a big effect
if args.optim_type == 0:
    gamma = 1
    criterion = FocalLoss(gamma=gamma, alpha=None)
    logging('FocalLoss(gamma=1, alpha=None)')
if args.optim_type == 1:
    gamma = 1
    criterion = FocalLoss(gamma=gamma, alpha=alpha)
    logging('FocalLoss(gamma=1, alpha=alpha)')
    

criterion = FocalLoss(gamma=args.criterion_gamma)
perf_logger.metrics['args'] = vars(args)

# At any point you can hit Ctrl + C to break out of training early.
log_start_time = time.time()
eval_start_time = time.time()
now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
try:
    for epoch in itertools.count(start=1):
        train(perf_logger, criterion=criterion, optimizer=optimizer)
        if train_step == args.max_step:
            val_loss, val_probs, val_targets = evaluate(model, va_iter, args, criterion=criterion)
            perf_logger.log_loss(val_loss, key='val')
            mask = np.full(len(val_targets[0]), True)
            for v_prob, v_target in zip(val_probs, val_targets):
                perf_logger.log_metrics(v_target[mask], v_prob[mask], key='val')
                mask = v_target[mask] == v_target[mask].max()
            with open(os.path.join(args.work_dir, f'{now}.json'), 'w') as f:
                json.dump(perf_logger.metrics, f)
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    val_loss, val_probs, val_targets = evaluate(model, va_iter, args, criterion=criterion)
    perf_logger.log_loss(val_loss, key='val')
    mask = np.full(len(val_targets[0]), True)
    for v_prob, v_target in zip(val_probs, val_targets):
        perf_logger.log_metrics(v_target[mask], v_prob[mask], key='val')
        mask = v_target[mask] == v_target[mask].max()
    with open(os.path.join(args.work_dir, f'{now}.json'), 'w') as f:
        json.dump(perf_logger.metrics, f)
    logging('-' * 100)
    logging('Exiting from training early')

with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

test_loss, test_probs, test_targets = evaluate(para_model, te_iter, args, criterion=criterion)
perf_logger.log_loss(test_loss, key='test', smooth=False)
mask = np.full(len(test_targets[0]), True)
for t_prob, t_target in zip(test_probs, test_targets):
    perf_logger.log_metrics(t_target[mask], t_prob[mask], key='test')
    mask = t_target[mask] == t_target[mask].max()
with open(os.path.join(args.work_dir, f'{now}.json'), 'w') as f:
    json.dump(perf_logger.metrics, f)
