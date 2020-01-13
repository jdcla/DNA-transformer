from copy import copy
import scipy.stats as stats
import numpy as np
import torch

from utils.dna_functions import Genome, vec_to_img
from utils.vocabulary import Vocab


class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, target=None, overlap=0, img=False, device='cpu'):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.img = img
        self.bsz = bsz
        self.bptt = bptt  # equal tgt_len
        self.overlap = overlap # extend downstream (in case of ext_ds > 0)
        self.device = device
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        if self.img:
            self.data = torch.FloatTensor(vec_to_img(self.data.numpy())).view(bsz, -1, 4).permute(1,0,2).contiguous().to('cpu')
        else:
            self.data = self.data.view(bsz, -1).t().contiguous().to('cpu')

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt
        self.n_batch_same_length = (self.n_step + self.bptt - 1) // (self.bptt-self.overlap)

        if target is not None:
            target = target.narrow(0, 0, self.n_step * bsz)
            self.target = torch.LongTensor(target).view(bsz, -1).t().contiguous().to('cpu')
            
            #self.target = self.target_labs
        else:
            self.target = None

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - i)

        end_idx = i + seq_len
        beg_idx = max(0, i)
        
        data = self.data[beg_idx:end_idx]
        if self.target is not None:
            target = self.target[beg_idx:end_idx]
        else:
            target = self.data[i+1:i+1+seq_len]

        return data.to(self.device), target.to(self.device), seq_len

    def get_fixlen_iter(self, start=0, same_length=False):
        if same_length:
            for i in range(start, self.data.size(0) - 1, self.bptt-self.overlap):
                yield self.get_batch(i)
        else:
            for i in range(start, self.data.size(0) - 1, self.bptt):
                yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter(same_length=True)


class Corpus(object):
    def __init__(self, data_path, labels=2, merge_size=1, shift=20,
                 norm_dist=False, fracs=[0.2, 0.1], at_idx=None):
        self.data_path = data_path
        self.vocab = Vocab()
        self.labels = labels
        
        token_size = 4**merge_size
        self.Gen = Genome(data_path, shift=shift, merge_size=merge_size)
        if type(at_idx) is list and len(at_idx) > 1:
            parts = self.Gen.slice_genome(fractions=None, at_idx=at_idx)
        else:
            parts = self.Gen.slice_genome(fracs, at_idx)
        
        self.Gen_train = parts[0]
        print(f'Train set size: {len(self.Gen_train.labels)}')
        if len(parts) > 1:
            self.Gen_test = parts[1]
            print(f'Test set size: {len(self.Gen_test.labels)}')
            if len(parts) == 3:
                self.Gen_valid = parts[2]
            else:
                self.Gen_valid = copy(self.Gen_test)
        
        self.T_type = torch.LongTensor

        self.vocab.create_tokens(token_size, self.Gen_train.DNA.ravel())
        self.vocab.build_vocab()
        
        self.train_lab, self.valid_lab, self.test_lab = None, None, None

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            self.train = torch.LongTensor(self.Gen_train.dna_out)
            if self.labels is not None:
                self.train_lab = torch.Tensor(self.Gen_train.labels).type(self.T_type)
            data_iter = LMOrderedIterator(self.train, *args, target=self.train_lab, **kwargs)
        elif split == 'valid':
            assert self.Gen_valid, 'No valid data incorporated'
            self.valid = torch.LongTensor(self.Gen_valid.dna_out)
            if self.labels is not None:
                self.valid_lab = torch.Tensor(self.Gen_valid.labels).type(self.T_type)
            data_iter = LMOrderedIterator(self.valid, *args, target=self.valid_lab, **kwargs)
        elif split == 'test':
            assert self.Gen_test, 'No test data incorporated'
            self.test = torch.LongTensor(self.Gen_test.dna_out)
            if self.labels is not None:
                self.test_lab = torch.Tensor(self.Gen_test.labels).type(self.T_type)
            data_iter = LMOrderedIterator(self.test, *args, target=self.test_lab, **kwargs)
        else:
            return False
        return data_iter


def get_lm_corpus(data_path, labels=2, shift=20, merge_size=1, norm_dist=False, fracs=[0.2, 0.1], at_idx=None):
    kwargs = {}
    corpus = Corpus(data_path, labels=labels, merge_size=merge_size, shift=shift,
                    norm_dist=norm_dist, fracs=fracs, at_idx=at_idx, **kwargs)

    return corpus


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
