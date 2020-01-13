import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, conv_size=7, pre_conv=False,
                 tgt_len=None, mem_len=None):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.pre_conv = pre_conv
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.conv_size = conv_size
            
        if conv_size != 0:
            if self.pre_conv:
                self.pre_motif_net = nn.Conv1d(d_model, d_model, conv_size, padding=conv_size//2)
            else:
                self.motif_net_q = nn.Conv1d(d_head, d_head, conv_size, padding=conv_size//2)
                self.motif_net_k = nn.Conv1d(d_head, d_head, conv_size, padding=conv_size//2)
                self.motif_net_v = nn.Conv1d(d_head, d_head, conv_size, padding=conv_size//2)
        
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_conv and (self.conv_size > 0):
                cat = cat.permute(1,2,0).contiguous()
                cat = self.pre_motif_net(cat).permute(2,0,1)
            w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_new = w
            if self.pre_conv and (self.conv_size > 0):
                w_new = w.permute(1,2,0).contiguous()
                w_new = self.pre_motif_net(w_new).permute(2,0,1)
            w_heads = self.qkv_net(w_new)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        if (self.conv_size == 0) or self.pre_conv:
            w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)      # qlen x bsz x n_head x d_head
            w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)     # qlen x bsz x n_head x d_head
            w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)     # qlen x bsz x n_head x d_head
        
        else:
            w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head).permute(2,1,3,0)     # qlen x bsz x n_head x d_head
            w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head).permute(2,1,3,0)      # qlen x bsz x n_head x d_head
            w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head).permute(2,1,3,0)      # qlen x bsz x n_head x d_head

            w_head_q = w_head_q.reshape(self.n_head * bsz, self.d_head, qlen)  # qlen x bsz x n_head x d_head
            w_head_k = w_head_k.reshape(self.n_head * bsz, self.d_head, klen)      # qlen x bsz x n_head x d_head
            w_head_v = w_head_v.reshape(self.n_head * bsz, self.d_head, klen)

            w_head_q = self.motif_net_q(w_head_q).view(self.n_head, bsz, self.d_head, qlen).permute(3,1,0,2)
            w_head_k = self.motif_net_k(w_head_k).view(self.n_head, bsz, self.d_head, klen).permute(3,1,0,2)
            w_head_v = self.motif_net_v(w_head_v).view(self.n_head, bsz, self.d_head, klen).permute(3,1,0,2)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        # r_w_bias: n_head x d_head
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        
        # r_r_bias: n_head x d_head
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)
        # [qlen x klen x bsz x n_head]
        self.attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(self.attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask,
                               mems=mems)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        self.emb_layers.append(
            nn.Embedding(n_token, d_embed, sparse=False)
        )
        if d_proj != d_embed:
            self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
    def forward(self, inp):
        embed = self.emb_layers[0](inp) # inp: qlen x bs -> embed: qlen x bs x d_embed
        if self.d_proj != self.d_embed:
            embed = F.linear(embed, self.emb_projs[0])

        embed.mul_(self.emb_scale)

        return embed
    
class ConvEmbeddings(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, conv_size, cutoffs):
        super().__init__()
        
        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        self.emb_layers.append(
            nn.Conv1d(4, d_embed, conv_size, padding=conv_size//2)
        )
        if d_proj != d_embed:
            self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
    def forward(self, inp):
        embed = self.emb_layers[0](inp.permute(1,2,0).contiguous()).permute(2,0,1).contiguous()
        if self.d_proj != self.d_embed:
            embed = F.linear(embed, self.emb_projs[0])

        embed.mul_(self.emb_scale)

        return embed   

class MemTransformerLM(nn.Module):
    def __init__(self, n_token_in, n_token_out, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, conv_size=7, conv_emb=False, pre_conv=False, tie_weight=True, d_embed=None,
                 tie_projs=[False], tgt_len=None, mem_len=None, ext_ds=None,
                 cutoffs=[], same_length=False, clamp_len=-1):
        super().__init__()
        self.n_token_in = n_token_in
        self.n_token_out = n_token_out

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.conv_size = conv_size
        self.pre_conv = pre_conv
        self.conv_emb = conv_emb
        
        if conv_emb:
            self.word_emb = ConvEmbeddings(n_token_in, d_embed, d_model, conv_size, cutoffs,)
        else:
            self.word_emb = AdaptiveEmbedding(n_token_in, d_embed, d_model, cutoffs,)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_ds = ext_ds
        self.max_klen = tgt_len +  mem_len

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout, conv_size=conv_size,
                    pre_conv=pre_conv, tgt_len=tgt_len, mem_len=mem_len,
                    dropatt=dropatt)
            )

        self.out_layer = nn.Linear(d_model, n_token_out)
        # use adaptive softmax (including standard softmax)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token_out, d_embed, d_model,
                                                cutoffs)

        if tie_weight:
            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
        
        if tie_projs:
            for i, tie_proj in enumerate(tie_projs):
                if tie_proj and d_model != d_embed:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                elif tie_proj:
                    self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, mem_len, ext_ds):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_ds = ext_ds

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        if self.conv_emb:
            qlen, bsz, _ = dec_inp.size()
        else:
            qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None].type(torch.bool) # -1
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen),
                                       diagonal=1 + mlen + self.ext_ds).byte()[:, :, None].type(torch.bool)

        hids = []
        
        ## attn
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                             self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)
        ##
        
        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems, criterion=None, last=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        tgt_len = target.size(0) 
        if not mems:
            mems = self.init_mems()
        if self.same_length and not last:
            tgt_len_adj = tgt_len - (self.ext_ds)
        else:
            tgt_len_adj = tgt_len

        target = target[:tgt_len_adj]
        hidden, new_mems = self._forward(data, mems=mems)
        if new_mems is not None:
            new_mems = [mem[:tgt_len_adj] for mem in new_mems]
        pred_hid = hidden[:tgt_len_adj] # do not evaluate scores downstream

        """loss, logit = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1),
                                criterion)
        loss = loss.view(tgt_len, -1)"""

        logit = self.out_layer(pred_hid.view(-1, pred_hid.size(-1)))
        if criterion is None:
            loss = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.view(-1).unsqueeze(1)).squeeze(1) # Problem with gather for merge
        else:
            loss = criterion(logit, target.view(-1))

        logits = [logit]
        targets = [target]

        targets = [t.view(-1).cpu().data.numpy() for t in targets]
        preds = [F.softmax(lgt, dim=1).cpu().data.numpy() for lgt in logits]

        if new_mems is None:
            return [loss] + [preds] + [targets]
        else:
            return [loss] + [preds] + [targets] + new_mems
