import os
import json
import argparse

import numpy as np
import torch

from mem_transformer import MemTransformerLM
from utils.dna_functions import Logger, FocalLoss
import utils.proj_adaptive_softmax
from data_utils import get_lm_corpus


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--model_path', type=str,
                    help='model path')
input_args = parser.parse_known_args()[0]
print(input_args)
with open(input_args.model_path, 'r') as fh:
    args = argparse.Namespace(**json.load(fh)['args'])
    
args.restart_dir = '/'.join(input_args.model_path.split('/')[:-1])

device = torch.device('cuda' if args.cuda else 'cpu')

n_token_in = (4 + 4 * args.methylation) ** args.merge_size
n_token_out = 2

print(args.coords[1:4])

model = MemTransformerLM(n_token_in, n_token_out, args.n_layer, args.n_head, args.d_model,
                         args.d_head, args.d_inner, args.dropout, args.dropatt,
                         tie_weight=args.tied, d_embed=args.d_embed, custom_emb=None,
                         tie_projs=[False], pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
                         ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=[],
                         same_length=args.same_length, clamp_len=args.clamp_len, 
                         sample_softmax=args.sample_softmax)

model.load_state_dict(torch.load(os.path.join(args.restart_dir, 'model.pt')))
model = model.to(device)

def evaluate(model, eval_iter, args, criterion=None):
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
            last = True if (i+1 == i_max) else False
            ret = model(data, target_init, *mems, criterion=criterion, last=last)
            loss, prob, target, mems = ret[0], ret[1], ret[2], ret[3:]

            probs.append([p.reshape(-1, args.batch_size,
                                    p.shape[1]).transpose(1, 0, 2) for p in prob])

            targets.append([t.reshape(-1, args.batch_size).T for t in target])
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    probs = [np.concatenate([p[i] for p in probs], axis=1) for i in range(len(probs[0]))]
    targets = [np.concatenate([t[i] for t in targets], axis=1) for i in range(len(targets[0]))]

    probs = [p.reshape(-1, p.shape[2]) for p in probs]
    targets = [t.reshape(-1) for t in targets]
    
    model.same_length = False
    model.reset_length(args.tgt_len, args.mem_len, args.ext_ds)
    model.train()

    return total_loss / total_len, probs, targets


corpus = get_lm_corpus(args.data_path, labels=n_token_out, merge_size=args.merge_size, at_idx=args.coords[1:4])

eval_batch_size = 10
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
                              device=device, ext_len=args.ext_len)

perf_logger = Logger(['AUC', 'P-R', 'acc'], True, ['val'])
perf_logger.metrics['args'] = vars(args)
criterion = FocalLoss(gamma=args.criterion_gamma, alpha=None)
test_loss, test_probs, test_targets = evaluate(model, te_iter, args, criterion=criterion)
perf_logger.log_metrics(test_targets[0], test_probs[0])
