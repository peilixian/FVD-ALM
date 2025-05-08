import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import copy
from torch.autograd import Variable

class MHSA_Intra(nn.Module):


    def __init__(self, dim_in, heads, pos_enc_type='relative', use_pos=True):
        super(MHSA_Intra, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in
        self.heads = heads
        self.dim_head = self.dim_inner // self.heads


        self.scale = self.dim_head ** -0.5

        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()

    def forward(self, input):
        B, C, T = input.shape
        query = self.conv_query(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3,
                                                                                     2).contiguous()  # (B, h, T, dim_head) # Qi = Wq * ai
        key = self.conv_key(input).view(B, self.heads, self.dim_head, T)  # (B, h, dim_head, T) #Ki = Wk * ai
        value = self.conv_value(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3,
                                                                                     2).contiguous()  # (B, h, T, dim_head) # Vi = Wv * ai

        query *= self.scale
        sim = torch.matmul(query, key)  # (B, h, T, T)
        attn = F.softmax(sim, dim=-1)  # (B, h, T, T)
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.matmul(attn, value)  # (B, h, T, dim_head)
        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T)  # (B, C, T)
        output = input + self.bn(self.conv_out(output))
        return output

class Memory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_mu = args.mu_queue_len  # 5
        self.n_class = args.num_class  # 20
        self.out_dim = args.feature_size  # 2048
        # memork bank大小：CxSxD 20x5x2048
        self.register_buffer("cls_mu_queue", torch.zeros(self.n_class, self.n_mu, self.out_dim))  # 20 5 2048
        # torch.zeros:用于创建一个指定大小的全零张量（tensor），这里指创建一个20x5x2048形状的全零张量，
        # register_buffer不会注册到模型参数中model.parameters()会注册到模型model.state_dict()
        # 创建了一个20x5x2048形状的全零张量，命名为cls_mu_queue,并且不会被梯度回传
        self.register_buffer("cls_sc_queue", torch.zeros(self.n_class, self.n_mu))  # 20 5

    @torch.no_grad()
    def _update_queue(self, inp_mu, inp_sc, cls_idx, coe):
        for idx in cls_idx:
            self._sort_permutation(inp_mu, inp_sc, idx, coe)

    @torch.no_grad()
    def _sort_permutation(self, inp_mu, inp_sc, idx, coe):
        concat_sc = torch.cat([self.cls_sc_queue[idx, ...], inp_sc[..., idx]],
                              0)  # （13）idx代表更新哪一类比如第3类 拼接第3类分数队列idx=2 对应的00000 和代表性片段对应的第二类得分0.2288 0.2277 0.2277 0.22770.2277 0.2277 0.2277 0.2277
        concat_mu = torch.cat([self.cls_mu_queue[idx, ...], inp_mu], 0)  # 拼接memory bank中第idx类片段特征和代表性片段特征 13*2048
        sorted_sc, indices = torch.sort(concat_sc, descending=True)  # sorted_sc：降序排序后的分数队列 indices：排序对应的原来的索引顺序
        sorted_mu = torch.index_select(concat_mu, 0, indices[:self.n_mu])  # 按照indices得到第idx类对应的得分前5的片段特征
        clsmu = self.cls_mu_queue[idx, ...]
        self.cls_mu_queue[idx, ...] = (1 - coe) * clsmu + coe * sorted_mu  # 更新第idx类对应的特征队列
        self.cls_sc_queue[idx, ...] = sorted_sc[:self.n_mu]  # 更新第idx类对应的分数队列

    @torch.no_grad()
    def _init_queue(self, mu_queue, sc_queue, lbl_queue, coe):
        for mu, sc, lbl in zip(mu_queue, sc_queue, lbl_queue):
            lbl = lbl.cpu()
            idxs = np.where(lbl == 1)[0].tolist()
            self._update_queue(mu, sc, idxs, coe)

    @torch.no_grad()
    def _return_queue(self, cls_idx):
        mus = []
        for idx in cls_idx:
            mus.append(self.cls_mu_queue[idx][None, ...])
        mus = torch.cat(mus, 1)
        return mus

    @torch.no_grad()
    def _neg_queue(self, cls_idx):
        if len(cls_idx) ==1:
            for idx in cls_idx:
                mu_feats1 = self.cls_mu_queue[:idx,:,:]
                mu_feats2 = self.cls_mu_queue[idx+1:,:,:]
                mu_feats = torch.cat((mu_feats1,mu_feats2),0)
        else:
            idx = cls_idx[0]
            idx1 = cls_idx[1]
            mu_feats1 = self.cls_mu_queue[:idx,:,:]
            mu_feats2 = self.cls_mu_queue[idx+1:idx1+1,:,:]
            mu_feats3 = self.cls_mu_queue[idx1+1:,:,:]
            mu_feats = torch.cat((mu_feats1,mu_feats2,mu_feats3),0)
        return mu_feats

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)


def calculate_l1_norm(f):  # 1*138*2048
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)  # 1*138*1
    f = f / (f_norm + 1e-9)  # 1*138*2048
    return f


def random_walk(x, y, w):  # (bipartite random walk (BiRW)二分随机游走模块：获得更新特征)原视频特征1*138*2048  代表性片段特征1*8*2048
    x_norm = calculate_l1_norm(x)  # 1*138*2048
    y_norm = calculate_l1_norm(y)  # 1*8*2048
    eye_x = torch.eye(x.size(1)).float().to(x.device)  # 138*138 对角线元素为1，其余全为0

    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)  # 1*8*2048 x 1*138*2048 ->1*8*138
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)  # 1*8*138 / 1*8*1 ->1*8*138
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])  # 1*8*138 x 1*8*138 ->1*138*138
    mat_inv_x = torch.inverse(eye_x - (w ** 2) * affinity_mat)  # 1*138*138
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd',
                                 [latent_z, y]) + x  # 1*8*138 x 1*138*2048 ->1*138*2048 + 1*138*2048->1*138*2048
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])  # 1*138*138 * 1*138*2048 ->1*138*2048

    return refined_x


def Contrastive(x, element_logits, labels, is_back=False):
    if is_back:
        labels = torch.cat(
            (labels, torch.ones_like(labels[:, [0]])), dim=-1)
    else:
        labels = torch.cat(
            (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
    sim_loss = 0.
    n_tmp = 0.
    _, n, c = element_logits.shape

    for i in range(0, 3 * 2, 2):
        atn1 = F.softmax(element_logits[i], dim=0)
        atn2 = F.softmax(element_logits[i + 1], dim=0)

        n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
        n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
        Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
        Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
        Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

        d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
        d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

        sim_loss = sim_loss + 0.5 * torch.sum(
            torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss + 0.5 * torch.sum(
            torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
        n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
    sim_loss = sim_loss / n_tmp
    return sim_loss

class Modality_Enhancement_Module(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim = 1024
        self.AE_e = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim // 2, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.AE_d = nn.Sequential(
            nn.Conv1d(embed_dim // 2, n_feature, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv1 = nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
    def forward(self,vfeat,ffeat):
        fusion_feat = self.AE_e(ffeat)
        new_feat = self.AE_d(fusion_feat)
        channel_attn = self.channel_conv1(vfeat)
        bit_wise_attn = self.channel_conv1(ffeat)
        
        filter_feat = torch.sigmoid(channel_attn)*torch.sigmoid(bit_wise_attn)*vfeat
        
        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat,new_feat,vfeat

class Optical_convolution(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.opt_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())

    def forward(self, ffeat):
        opt_wise_attn = self.opt_wise_attn(ffeat)
        filter_ffeat = torch.sigmoid(opt_wise_attn) * ffeat
        opt_attn = self.attention(filter_ffeat)
        return opt_attn, filter_ffeat

class TFE_DC_Module(nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        
        embed_dim = 1024
        self.layer1 = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=2 ** 0, dilation=2 ** 0),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        self.layer2 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 1, dilation=2 ** 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        self.layer3 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 2, dilation=2 ** 2),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
    def forward(self, x):
        out = self.layer1(x)
        out_attention1 = self.attention(torch.sigmoid(out)*x)
        
        out = self.layer2(out)
        out_attention2 = self.attention(torch.sigmoid(out)*x)
        
        out = self.layer3(out)
        out_feature = torch.sigmoid(out)*x
        out_attention3 = self.attention(out_feature)
        
        out_attention = (out_attention1+out_attention2+out_attention3)/3.0

        return out_attention, out_feature, out, x


class TFEDCN(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss()
        embed_dim=2048
        mid_dim=1024
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio
        self.m = args['opt'].m
        self.n_mu = args['opt'].mu_num
        self.em_iter = args['opt'].em_iter
        self.mu = nn.Parameter(torch.randn(self.n_mu, embed_dim))  # 8*2048
        torch_init.xavier_uniform_(self.mu)
        self.mu_k = nn.Parameter(torch.randn(self.n_mu, embed_dim))  # 8*2048
        torch_init.xavier_uniform_(self.mu)
        self.mu_k.requires_grad = False
        self.MHSA_Intra = MHSA_Intra(dim_in=embed_dim, heads=8)
        self.vAttn = getattr(model,args['opt'].AWM)(1024,args)
        self.fAttn = getattr(model,args['opt'].TCN)(1024,args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class+1, 1))
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
        self.batch_avg=nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        
        self.apply(weights_init)
        self.vAttn_k = getattr(model, args['opt'].AWM)(1024, args)
        for vAttnparam_q, vAttnparam_k in zip(
                self.vAttn.parameters(), self.vAttn_k.parameters()
        ):
            vAttnparam_k.data.copy_(vAttnparam_q.data)  # initialize
            vAttnparam_k.requires_grad = False
        self.fAttn_k = getattr(model, args['opt'].TCN)(1024, args)
        for fAttnparam_q, fAttnparam_k in zip(
                self.fAttn.parameters(), self.fAttn_k.parameters()
        ):
            fAttnparam_k.data.copy_(fAttnparam_q.data)  # initialize
            fAttnparam_k.requires_grad = False
        self.MHSA_Intra_k = MHSA_Intra(dim_in=embed_dim, heads=8)
        for MHSAparam_q, MHSAparam_k in zip(
                self.MHSA_Intra.parameters(), self.MHSA_Intra_k.parameters()
        ):
            MHSAparam_k.data.copy_(MHSAparam_q.data)  # initialize
            MHSAparam_k.requires_grad = False
        self.fusion_k = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        for fuparam_q, fuparam_k in zip(
                self.fusion.parameters(), self.fusion_k.parameters()
        ):
            fuparam_k.data.copy_(fuparam_q.data)  # initialize
            fuparam_k.requires_grad = False
        self.classifierk = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))
        for classq, classk in zip(
                self.classifier.parameters(), self.classifierk.parameters()
        ):
            classk.data.copy_(classq)
            classk.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for vAttnparam_q, vAttnparam_k in zip(
                self.vAttn.parameters(), self.vAttn_k.parameters()
        ):
            vAttnparam_k.data = vAttnparam_k.data * self.m + vAttnparam_q.data * (1.0 - self.m)
        for fAttnparam_q, fAttnparam_k in zip(
                self.fAttn.parameters(), self.fAttn_k.parameters()
        ):
            fAttnparam_k.data = fAttnparam_k.data * self.m + fAttnparam_q.data * (1.0 - self.m)
        for MHSAparam_q, MHSAparam_k in zip(
                self.MHSA_Intra.parameters(), self.MHSA_Intra_k.parameters()
        ):
            MHSAparam_k.data = MHSAparam_k.data * self.m + MHSAparam_q.data * (1.0 - self.m)
        for fuparam_q, fuparam_k in zip(
                self.fusion.parameters(), self.fusion_k.parameters()
        ):
            fuparam_k.data = fuparam_k.data * self.m + fuparam_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
                self.classifier.parameters(), self.classifierk.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def EM(self, mu, x):
        # propagation -> make mu as video-specific mu
        norm_x = calculate_l1_norm(x)  # 最后一维进行二范数运算1*138*2048->1*138*2048
        for _ in range(self.em_iter):
            norm_mu = calculate_l1_norm(mu)  # 1*8*2048->1*8*2048
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0,
                                 1)  # 1*8*2048 x 1*138*2048 ->1*8*138
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)  # 1*8*138 / 1*8*1 ->1*8*138
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])  # 1*8*138 x 1*138*2048 ->1*8*2048
        return mu

    def EM2(self, mu, x):
        # propagation -> make mu as video-specific mu
        norm_x = calculate_l1_norm(x)  # 最后一维进行二范数运算1*138*2048->1*138*2048
        for _ in range(self.em_iter + 2):
            norm_mu = calculate_l1_norm(mu)  # 1*8*2048->1*8*2048
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0,
                                 1)  # 1*8*2048 x 1*138*2048 ->1*8*138
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)  # 1*8*138 / 1*8*1 ->1*8*138
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])  # 1*8*138 x 1*138*2048 ->1*8*2048
        return mu

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()
        f_atn,ffeat,n_ffeat,o_ffeat = self.fAttn(feat[:,1024:,:])
        v_atn,vfeat,n_rfeat,o_rfeat = self.vAttn(feat[:,:1024,:],ffeat)
        x_atn = (f_atn+v_atn)/2
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = self.fusion(nfeat)
        nfeat = self.MHSA_Intra(nfeat)
        mu = self.mu[None, ...].repeat(b, 1, 1)
        mu = self.EM(mu, nfeat.transpose(-1, -2))
        reallocated_x = random_walk(nfeat.transpose(-1, -2), mu, 0.5)
        x_cls = self.classifier(nfeat)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            f_atnk, ffeatk, n_ffeatk, o_ffeatk = self.fAttn_k(feat[:, 1024:, :])
            v_atnk, vfeatk, n_rfeatk, o_rfeatk = self.vAttn_k(feat[:, :1024, :], ffeatk)
            x_atnk = (f_atnk + v_atnk) / 2
            nfeatk = torch.cat((vfeatk, ffeatk), 1)
            nfeatk = self.fusion_k(nfeatk)
            nfeatk = self.MHSA_Intra_k(nfeatk)
            xk_cls = self.classifierk(nfeatk)
        r_cls = self.classifier(reallocated_x.transpose(-1, -2))
        mu_cls = self.classifier(mu.transpose(-1, -2))

        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2),'attn':x_atn.transpose(-1, -2),
                'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2),'mu': mu,
                'r_cas': r_cls.transpose(-1, -2), 'mu_cas': mu_cls.transpose(-1, -2),'cask':xk_cls.transpose(-1, -2),
                'n_rfeat':n_rfeat.transpose(-1,-2),'o_rfeat':o_rfeat.transpose(-1,-2),'n_ffeat':n_ffeat.transpose(-1,-2),
                'o_ffeat':o_ffeat.transpose(-1,-2)
                }


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, memory, **args):
        feat, element_logits, r_element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['r_cas'],outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        element_logitsk = outputs['cask']
        norm_cas = calculate_l1_norm(element_logits)
        norm_rcas = calculate_l1_norm(r_element_logits)
        
        mutual_loss=0.5*F.mse_loss(v_atn,f_atn.detach())+0.5*F.mse_loss(f_atn,v_atn.detach())
        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, f_atn,include_min=True)
        r_element_logits_supp = self._multiply(r_element_logits, f_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        loss_mil_orig_r, _ = self.topkloss(r_element_logits,
                                           labels,
                                           is_back=True,
                                           rat=args['opt'].k,
                                           reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        loss_mil_supp_r, _ = self.topkloss(r_element_logits_supp,
                                           labels,
                                           is_back=False,
                                           rat=args['opt'].k,
                                           reduce=None)
        actionloss = loss_mil_orig + loss_mil_orig_r
        backloss = loss_mil_supp + loss_mil_supp_r
        num_itr = labels.shape[0]
        craloss_stack = []
        for i in range(num_itr):
            label = labels[i, ...].unsqueeze(0).cpu()
            idxs = np.where(label == 1)[1].tolist()
            q = element_logits[i, ...].unsqueeze(0)
            q = torch.mean(q, 1)
            q = nn.functional.normalize(q, dim=1)  # 1 21
            k = element_logitsk[i, ...].unsqueeze(0)
            k = torch.mean(k, 1)
            k = nn.functional.normalize(k, dim=1)
            if len(idxs) == 1:
                for idx in idxs:
                    negcas1 = element_logitsk[:idx, :, :]
                    negcas2 = element_logitsk[idx + 1:, :, :]
                    neg = torch.cat((negcas1, negcas2), 0)
            else:
                idx = idxs[0]
                idx1 = idxs[1]
                negcas1 = element_logitsk[:idx, :, :]
                negcas2 = element_logitsk[idx + 1:idx1 + 1, :, :]
                negcas3 = element_logitsk[idx1 + 1:, :, :]
                neg = torch.cat((negcas1, negcas2, negcas3), 0)
            neg = torch.mean(neg, 0).unsqueeze(0)
            neg = neg.permute(0, 2, 1)
            neg = nn.functional.normalize(neg, dim=1)
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,nck->nk', [q, neg])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= 0.07
            labelss = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            craloss = self.celoss(logits, labelss).reshape(1)
            craloss_stack.append(craloss)
        craloss_out = torch.tensor([item.cpu().detach().numpy() for item in craloss_stack]).squeeze(1).cuda()
        spl_loss = self.lossspl(norm_cas, norm_rcas)

        #loss_3_supp_Contrastive = Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.abs().mean()
        # guide loss
        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.abs().mean()
        # guide loss
        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()
        
        
        # total loss
        total_loss = (actionloss.mean() +
                      backloss.mean() +
                      spl_loss +
                      craloss_out.mean()+
                      #args['opt'].alpha3*loss_3_supp_Contrastive+
                      + args['opt'].alpha1 * (f_loss_norm + v_loss_norm)
                      + args['opt'].alpha2 * f_loss_guide
                      + args['opt'].alpha3 * v_loss_guide
                      + args['opt'].alpha4 * mutual_loss
                      + args['opt'].alpha4 * loss_norm/3
                      + args['opt'].alpha4 * loss_guide/3
                      )

        return total_loss


    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(1,0)
        neg = nn.functional.normalize(neg, dim=1).unsqueeze(0)
        l_pos =torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg =torch.einsum('nc,nck->nk', [q,neg])
        logits = torch.cat([l_pos,l_neg],dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
        con_loss = self.celoss(logits,labels)
        return con_loss
    def lossspl(self, pred, soft_label):
        soft_label = F.softmax(soft_label / 0.2, -1)
        soft_label = Variable(soft_label.detach().data, requires_grad=False)
        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / 0.2, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind