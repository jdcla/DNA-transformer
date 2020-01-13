import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, average_precision_score

# DNA LOADERS


class Genome:
    def __init__(self, data_path, shift=0, img=False, methylation=False, merge_size=1):
        """Creates Genome class object, holding DNA and label data for a given representation
        used to feed data into Transformer-XL data architecture
        __init__:
            data_path (str): path to np.array object holding array of DNA sequence and labels
            shift (int): shifts label downstream for positive int and upstream for negative int
            img (bool): one-hot (True) or vector (False) representation of output DNA
            methylation (bool): Use methylated nucleotide classes. Requires existence of
            {data_path}_meth.npy object holding methylation labels
            merge_size (int): Merges nucleotide classes and labels to k-mers
        """
        self.img = img

        # LOAD DATA
        self.data = np.load(data_path)
        if methylation:
            self.meth = True
            self.meth_labels = np.load(f'{data_path[:-4]}_meth.npy')
        else:
            self.meth = False
        # UNPACK
        is_img = np.shape(self.data)[1] >= 4 and self.data[:, :4].max() == 1
        dna_idx = 4 if is_img else 1
        if not is_complementary(self.data[:, :dna_idx], img=is_img):
            self.data = unpack_DNA(self.data, img=is_img)
        # STRIP
        self.DNA, self.labels = strip_labels(self.data, img=is_img)
        # TRANSFORM TO VECTOR
        if is_img:
            self.DNA = img_to_vec(self.DNA)
        # SHIFT LABELS
        self.shift = shift
        if self.shift != 0:
            # target labels are always rightmost
            target_labels = self.labels[:, -1]
            shifted_labels = np.concatenate((target_labels[-self.shift:],
                                            target_labels[:-self.shift]))
            if self.labels.shape[1] > 1:
                shifted_labels = np.concatenate((self.labels[:, :-1],
                                                 shifted_labels), axis=1)
            self.labels = shifted_labels.reshape(-1, 1)
        print("WARNING: Labels are shifted by {}".format(shift))
        if methylation:
            self.DNA = apply_meth(self.DNA, self.meth_labels)
        self.m_dict = merge_dict(merge_size, meth=self.meth)
        if merge_size > 1:
            if self.img:
                print('forcing vectorial representation')
                self.img = False
            trim_len = len(self.DNA) % (merge_size*2)
            self.DNA = self.DNA[:len(self.DNA)-trim_len]
            dna_str = unbinarize_DNA(self.DNA)
            dna_str_split = re.findall(f'[ACGT]{{{merge_size}}}', dna_str)
            dna_tokens = np.array([self.m_dict[2][t.lower()] for t in dna_str_split])
            self.labels = self.labels[:len(self.labels)-trim_len]
            self.DNA = dna_tokens.reshape(-1, 1)
            self.labels = np.array([np.max(lab.reshape(-1, merge_size), axis=1) for lab in self.labels.T]).T
        self.data = pack_DNA(self.DNA, self.labels, single=False)

    def __len__(self):
        return len(self.labels)//2

    def __getitem__(self, idx):
        assert (idx < len(self)), "index out of range"
        return self.data[idx]

    def slice_genome(self, fractions=0.2, at_idx=None):
        """Creates GenomePart class objects by slicing the genome

        :param fractions: (list/float) fractions of genome, residual fraction (1-sum(fractions) is used too
        :param at_idx: (list/int) defines specific location of spots at which genome is sliced
        :return: GenomePart object of genome fractions
        """
        if type(at_idx) == int:
            at_idx = [at_idx]
        if type(fractions) == int:
            fractions = [fractions]
        idxs = []
        parts = []
        if fractions == [1]:
            return [GenomePart(self, 0, len(self.data)+1)]
        elif at_idx is None:
            idxs.append(np.random.randint(0, len(self)))
        if fractions is None:
            idxs = at_idx
        else:
            for frac in fractions:
                new_idx = idxs[-1]+int(len(self)*frac)
                if new_idx > len(self):
                    new_idx -= len(self)
                idxs.append(new_idx)
        parts.append(GenomePart(self, idxs[-1], idxs[0]))
        for i in range(len(idxs)-1):
            parts.append(GenomePart(self, idxs[i], idxs[i+1]))
        return parts


class GenomePart:
    def __init__(self, genome, start, end):
        """GenomePart represents fraction of genome defined by Genome class object

        :param genome: Genome class object
        :param start: start index of genome covered by GenomePart
        :param end: end index of genome covered by GenomePart
        """
        self.genome = genome
        self.img = genome.img
        self.meth = genome.meth
        self.start = start
        self.end = end
        if start < end:
            self.data = genome.data[start:end]
        else:
            self.data = np.concatenate((genome.data[start:],
                                        genome.data[:end]))
        self.DNA, self.labels = strip_labels(unpack_DNA(self.data, single=False))
        if self.img:
            self.dna_out = vec_to_img(self.DNA)
        else:
            self.dna_out = self.DNA

    def __len__(self):
        return len(self.labels)


# DNA UTILS

def img_to_vec(img):
    """Transforms DNA representation from img format (one-hot) to
    vectorial format
    Inputs:
        img (np.array): one hot representation of the DNA sequence
    Outputs:
        vec (np.array): vectorial represenation of the DNA sequence
    """
    assert (img.max(axis=1) == 1).all(), f'expected one-hot encoding'
    vec = np.argmax(img, axis=1).reshape(-1, 1)
    return vec


def merge_dict(merge_size, meth=False):
    """Create a dictionary of merged k-mers dependent upon merge size
    Inputs:
        merge_size (int): size of k-mers from which dictionary is
        derived
        meth (bool): include methylated nucleotide classes
    Output:
        token_to_vec (dict): token to k-mer vector dictionary
        token_to_str (dict): token to k-mer string dictionary
        str_to_token (dict): k-mer string to token dictionary
    """
    nt_size = 8 if meth else 4
    values = np.zeros((nt_size**merge_size, merge_size), dtype=np.int)
    keys = np.arange(nt_size**merge_size)
    for l in range(merge_size):
        values[:, l] = keys//nt_size**l % nt_size
    strings = values.astype('<U1').astype(np.chararray)
    strings = np.add.reduce([s for s in strings.T]).astype(f'<U{merge_size}')
    strings = np.char.replace(strings, '0', 'a')
    strings = np.char.replace(strings, '1', 't')
    strings = np.char.replace(strings, '2', 'c')
    strings = np.char.replace(strings, '3', 'g')
    token_to_vec = {k: v for k, v in zip(keys, values.tolist())}
    token_to_str = {k: v for k, v in zip(keys, strings)}
    str_to_token = {v: k for k, v in zip(keys, strings)}
    return token_to_vec, token_to_str, str_to_token


def vec_to_img(vec, dtype=np.int8):
    """Transforms DNA representation from vectorial format to image
    (one-hot) format
    Inputs:
        vec (np.array): vectorial represenation of the DNA sequence
    Outputs:
        img (np.array): one hot representation of the DNA sequence
    """
    img = np.zeros((vec.shape[0], vec.max()+1), dtype=dtype)
    img[np.arange(len(img)), vec.reshape(-1)] = 1

    return img


def strip_labels(data, img=False, single=True):
    """Split DNA sequence column from labels column
    Inputs:
        data (np.array): Array holding DNA sequence and labels
        img (bool): Defines input data as img (True) or vector (False).
        single (bool): data is unpacked
    Outputs:
        dna (np.array): DNA sequence
        labels (np.array): labels
    """
    mul = 1 if single else 2
    if img:
        nt_size = 4 * mul
    else:
        nt_size = 1 * mul
    if data.shape[1] > nt_size:
        return data[:, :nt_size], data[:, nt_size:]

    return data[:, :nt_size], None


def add_complement(dna, img=False):
    """Adds reverse complement to sequence
    Inputs:
        dna (np.array): DNA sequence
        img (bool): Defines input data as img (True) or vector (False).
    Outputs:
        dna (np.array): Unpacked DNA sequence
    """
    assert not is_complementary(dna), "DNA is already complementary"
    compl_dict = {0: 1, 1: 0, 2: 3, 3: 2}
    if img:
        dna_ext = np.concatenate((dna[:, :4],
                                  np.flip(dna[:, [1, 0, 3, 2]], axis=0)))
    else:
        compl_dna = dna[:, :1].copy()
        for k, v in compl_dict.items():
            compl_dna[dna == k] = v
        dna_ext = np.concatenate((dna[:, :1], np.flip(compl_dna, axis=0)))

    return dna_ext


def apply_meth(dna, meth):
    """Transforms DNA sequence to methylated DNA sequence
    Input:
        dna (np.array): array of DNA sequence
        meth (np.array): Methylation labels (len == len(dna))
    Output:
        dna_meth (np.array): array of methylated DNA sequence
    """
    dna_meth = dna.copy()
    img = True if dna.shape[1] == 4 else False
    assert is_complementary(dna, img=img), f'DNA needs to be unpacked first'
    mask = meth.sum(axis=1) > 0
    if img:
        assert dna.shape == meth.shape, f'{dna.shape} != {meth.shape}'
        dna_meth[mask] = 0
        dna_meth = np.concatenate((dna, meth), axis=1)
    else:
        dna_meth[mask] = dna[mask]+4

    return dna_meth


def is_complementary(dna, img=False):
    """Evaluates whether reverse complement is part of given DNA
    sequence
    Inputs:
        dna (np.array): DNA sequence
        img (bool): Defines input data as img (True) or vector (False).
    Outputs:
        bool : evaluation result
    """
    if len(dna) % 2 != 0:
        return False
    if img:
        dna = img_to_vec(dna)
    mid_idx = len(dna)//2
    left_cond = dna[:mid_idx]
    right_cond = np.flip(dna[mid_idx:], axis=0)
    right_cond = 1+right_cond//2*2 - right_cond % 2

    return np.equal(left_cond, right_cond).all()


def pack_DNA(data, labels_new=None, img=False, single=True):
    """folds unrolled DNA sequence and labels into complementary, packed
    unit.
    Inputs:
        data (np.array): Array holding DNA sequence and labels
        labels_new (bool): New labels to be added (len() == len(data))
        img (bool): Defines input data as img (True) or vector (False).
        single (bool): DNA is to be packed in single column (ensures
        complementarity sequence)
    Outputs:
        data_packed (np.array): Packed data array
    """
    dna, labels = strip_labels(data, img=img, single=single)
    # assert is_complementary(dna, img), "DNA is not complementary"
    mid_idx = data.shape[0]//2
    if single:
        data_packed = dna[:mid_idx]
    else:
        data_packed = np.concatenate((dna[:mid_idx], np.flip(dna[mid_idx:],
                                                             axis=0)), axis=1)

    if labels_new is not None:
        if labels is not None:
            labels = np.concatenate((labels, labels_new), axis=1)
        else:
            labels = labels_new
    if labels is not None:
        sense = labels[:mid_idx].reshape(-1, 1)
        antisense = np.flip(labels[mid_idx:], axis=0).reshape(-1, 1)
        labels_packed = np.concatenate((sense, antisense), axis=1)
        data_packed = np.concatenate((data_packed, labels_packed), axis=1)
        return data_packed

    return data_packed


def unpack_DNA(data, labels_new=None, img=False, single=True):
    """unfolds packed DNA sequence and labels
    unit.
    Inputs:
        data (np.array): Array holding DNA sequence and labels
        labels_new (bool): New labels to be added (len() == len(data))
        img (bool): Defines input data as img (True) or vector (False).
        single (bool): DNA is to be unpacked from single column (first)
        (ensures complementarity sequence)
    Outputs:
        data_unpacked (np.array): Packed data array
    """
    dna, labels = strip_labels(data, img=img, single=single)
    labels_unpacked = None
    if single:
        dna_unpacked = add_complement(dna, img)
    else:
        dna_idx = dna.shape[1]/2
        assert dna_idx == int(dna_idx), f'dim 1 of DNA has to be even'
        dna_unpacked = np.concatenate((dna[:, :int(dna_idx)],
                                       np.flip(dna[:, int(dna_idx):], axis=0)))

    if labels_new is not None:
        ln_idx = labels_new.shape[1]/2
        assert ln_idx == int(ln_idx), f'dim 1 of new labels has to be even'
        print(f"{ln_idx} new layer(s) of labels added")
        sense = labels_new[:, :int(ln_idx)]
        antisense = np.flip(labels_new[:, int(ln_idx):], axis=0)
        labels_unpacked = np.concatenate((sense, antisense))
    if labels is not None:
        l_idx = labels.shape[1]/2
        assert l_idx == int(l_idx), f'dim 1 of labels has to be even'
        if labels_new is not None:
            sense = labels[:, :int(l_idx)]
            asense = np.flip(labels[:, int(l_idx):], axis=0)
            labels_temp = np.concatenate((sense, asense))
            labels_unpacked = np.concatenate((labels_unpacked,
                                              labels_temp), axis=1)
        else:
            sense = labels[:, :int(l_idx)]
            asense = np.flip(labels[:, int(l_idx):], axis=0)
            labels_unpacked = np.concatenate((sense, asense))

    if labels_unpacked is not None:
        data_unpacked = np.concatenate((dna_unpacked, labels_unpacked), axis=1)
    else:
        data_unpacked = dna_unpacked

    return data_unpacked


def unbinarize_DNA(dna_bin, img=False, quick=False):
    """Transforms array representation of DNA to sequence
    Inputs:
        dna_bin (np.array): Array representation of DNA
        img (bool): Defines img (one-hot) or vectorial representation
        of input
        quick (bool): Defines image representation to not be composed
        out of degenerate sequences, speeds up process
    Outputs:
        dna_str (string): DNA sequence
    """
    img_dict = {1: {1: {1: {1: 'N', 0: 'H'}, 0: {1: 'D', 0: 'W'}},
                    0: {1: {1: 'V', 0: 'M'}, 0: {1: 'R', 0: 'A'}}},
                0: {1: {1: {1: 'B', 0: 'Y'}, 0: {1: 'K', 0: 'T'}},
                    0: {1: {1: 'S', 0: 'C'}, 0: {1: 'G', 0: 'N'}}}}
    img_dict_simple = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    int_dict = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'R', 5: 'Y',
                6: 'S', 7: 'W', 8: 'K', 9: 'M', 10: 'B', 11: 'D',
                12: 'H', 13: 'V', 14: 'N'}
    if img:
        dna_str = np.full(dna_bin.shape[0], 'N', dtype='|S1')
        dna_bin_stripped = dna_bin[:, :4]
        if quick:
            for idx, nt_img in enumerate(dna_bin_stripped):
                dna_str[idx] = img_dict_simple[nt_img.argmax()]
        else:
            for idx, nt_img in enumerate(dna_bin_stripped):
                value = img_dict[nt_img[0]][nt_img[1]][nt_img[2]][nt_img[3]]
                dna_str[idx] = value
    else:
        dna_str = np.full(dna_bin.shape[0], 'N', dtype='|S1')
        for idx, nt_img in enumerate(dna_bin):
            dna_str[idx] = int_dict[nt_img[0]]

    return dna_str.tostring().decode('utf-8')

# CRITERION


class FocalLoss(torch.nn.Module):
    """Implementation of FocalLoss
    __init__:
        gamma (float): gamma variable (gamma == 0 is default CE loss)
        alpha (float): weights for class imbalance
        size_average (bool): Outputs mean(loss)"""
    def __init__(self, gamma=0.5, alpha=None, size_average=False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.t_type = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        if isinstance(alpha, (float, int, np.long)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        if self.alpha is not None:
            print(f'Alpha: {self.alpha}')

    def forward(self, input, target):
        """Compute FocalLoss
        Input:
            input (Tensor): output logits of model
            target (Tensor): class targets
        Output:
            loss (Tensor): computed loss"""
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target.type(self.t_type))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).type(self.t_type))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss

# LOGGER FUNCTION


class Logger(object):
    def __init__(self, metrics, test=True, valid_keys=None):
        """Object which stores and calculates metrics produced by a neural
        network during training

        Attributes:
            metrics (list): lists all metrics stored during training. list can
                include ['acc','AUC', 'loss', 'P-R']
            test (bool): store metrics of test set (default: True)
            valid_keys (list): list of labels for each valid_loader used during
        """
        self.i = {'train': 0}
        self.log_auc, self.log_acc = False, False
        self.log_p_r = False
        self.metrics = {'train': {}}
        self.key = 'train'
        if test:
            self.metrics['test'] = {}
            self.i['test'] = 0
        self.valid_keys = valid_keys
        if valid_keys is not None:
            for valid_key in valid_keys:
                self.metrics[valid_key] = {}
                self.i[valid_key] = 0
        if 'acc' in metrics:
            self.log_acc = True
            for key in self.metrics:
                self.metrics[key].update({'acc': []})
        if 'AUC' in metrics:
            self.log_auc = True
            for key in self.metrics:
                self.metrics[key].update({'auc': []})
        if 'P-R' in metrics:
            self.log_p_r = True
            for key in self.metrics:
                self.metrics[key].update({'p-r': []})
        for key in self.metrics:
            self.metrics[key].update({'loss': [0]})

    def log_loss(self, loss, key=None, smooth=False):
        """Logs loss metric

        Attributes:
            loss (float): training loss
            key (string): label of processed data
        """
        if key is None:
            key = self.key
        self.i[key] += 1
        if smooth:
            update = (self.metrics[key]['loss'][-1] * (self.i[key] - 1)
                      + loss) / self.i[key]
        else:
            update = loss
        self.metrics[key]['loss'].append(update)

    def log_metrics(self, y_true, y_hat, key=None):
        """Logs non-loss metrics

        Attributes:
            y_true (array): array containing true labels
            y_hat (array): array containing predicted labels
            key (string): label of processed data
        """
        if key is None:
            key = self.key
        if y_true.shape != y_hat.shape:
            y_true_hot = pd.get_dummies(pd.Series(y_true.reshape(-1)))
            zeros = pd.DataFrame(np.zeros(y_hat.shape))
            y_true_hot = y_true_hot.combine_first(zeros).astype(y_true_hot.dtypes.iloc[0]).values
        else:
            y_true_hot = y_true
        if self.log_auc:
            auc = [roc_auc_score(y_true_hot[:, i],
                                 y_hat[:, i]) for i in range(len(y_hat[0])) if y_true_hot[:, i].sum() != 0]
            self.metrics[key]['auc'].append(auc)
        if self.log_acc:
            acc = sum(np.argmax(y_hat, axis=1) == y_true) / len(y_true)
            self.metrics[key]['acc'].append(acc)
        if self.log_p_r:
            p_r = [average_precision_score(y_true_hot[:, i], y_hat[:, i]) for i in range(len(y_hat[0])) if
                   y_true_hot[:, i].sum() != 0]
            self.metrics[key]['p-r'].append(p_r)

    def output_metric(self, key=None, metric='loss'):
        """Prints last recorded value of metric

        Attributes:
            key (string): label of processed data
            metric (string): key of metric to be printed
        """
        if key is None:
            key = self.key
        return self.metrics[key][metric][-1]

    def output_metrics(self):
        """Prints last recorded values of all metrics
        """
        print('')
        for key in sorted(self.metrics):
            print('{}:'.format(key), end='')
            for k, v in self.metrics[key].items():
                if type(v[-1]) is list:
                    print('\t' + k + ': ' + ''.join('{:5.3f} '.format(vs) for vs in v[-1]), end='')
                else:
                    print('\t{}: {:5.3f}'.format(k, v[-1]), end='')
            print('\n', end='')
