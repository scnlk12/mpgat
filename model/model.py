import torch
from torch import nn
import numpy as np
import datetime
import torch.nn.functional as F
import math


class GMAN(nn.Module):
    def __init__(self, model_dim, P, Q, T, L, K, d, lap_mx, LAP, num_node, embed_dim, skip_dim=256):
        '''
        TE: [batch_size, P + Q, 2] (time-of-day, day-of-week)
        SE: [N, K * d]
        P:  number of history steps
        Q:  number of prediction steps
        T:  one day is divided into T steps
        L:  number of STAtt blocks in the encoder/decoder
        K:  number of attention heads
        d:  dimension of each attention head outputs
        return:  [batch_size, Q, N]
        '''
        super().__init__()
        self.model_dim = model_dim
        self.P = P
        self.Q = Q
        self.T = T
        self.L = L
        self.K = K
        self.d = d
        self.lap_mx = lap_mx
        self.LAP = LAP
        self.num_node = num_node
        self.embed_dim = embed_dim
        self.skip_dim = skip_dim

        # 对节点编码
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        # 网络结构
        # Use P (history steps) instead of T (total time steps in a day) for adaptive embedding
        self.STE_emb = STEmbedding(model_dim, K, d, lap_mx, num_node, P)
        self.ST_Att = nn.ModuleList(
            [
                STAttBlock(K, d, LAP, num_node)
                for _ in range(L)
            ]
        )

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=K * d, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(L)
        ])

        # 修复: 使用P而不是硬编码的12,适应不同的历史窗口长度
        self.end_conv1 = nn.Conv2d(
            in_channels=P, out_channels=Q, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=1, kernel_size=1, bias=True,
        )

    def forward(self, x, TE):
        D = self.K * self.d
        # input: transform from model_dim (3) to K*d (64)
        x = self.STE_emb(x, TE)

        skip = 0

        # encoder
        for i, att in enumerate(self.ST_Att):
            x = att(x, TE, self.LAP, self.node_embeddings)
            skip += self.skip_convs[i](x.permute(0, 3, 2, 1))

        # output
        skip = self.end_conv1(F.leaky_relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.leaky_relu(skip.permute(0, 3, 2, 1)))

        return skip.permute(0, 3, 2, 1).squeeze(-1)


class STEmbedding(nn.Module):
    def __init__(self, model_dim, K, d, lap_mx, num_node, time_step, drop=0.):
        super().__init__()
        self.K = K
        self.d = d
        self.lap_mx = lap_mx

        self.x_forward = nn.Sequential(
            nn.Linear(model_dim, self.K * self.d),
            nn.LeakyReLU(),
            nn.Linear(self.K * self.d, 4 * self.K * self.d),
            nn.LeakyReLU(),
            nn.Linear(4 * self.K * self.d, self.K * self.d),
        )

        # Ea (STAEformer论文) 感觉类似位置编码
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(time_step, num_node, self.K * self.d))
        )

        self.dropout = nn.Dropout(drop)

    def forward(self, x, TE):
        B, _, _, _ = TE.shape
        # Ea
        adp_emb = self.adaptive_embedding.expand(
            size=(B, *self.adaptive_embedding.shape)
        )  # (batch_size, in_steps, num_nodes, adaptive_embedding_dim)
        x = self.x_forward(x)
        # 时空adaptive embedding 类似于位置编码
        x += adp_emb
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, fea, residual_add=True):
        super(FeedForward, self).__init__()
        self.residual_add = residual_add
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[0])

    def forward(self, inputs):
        # x (B, T, 1, F)
        x = self.ln(inputs)
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                # x = F.relu(x)
                x = F.leaky_relu(x)
        if self.residual_add:
            x += inputs
        return x


class STAttBlock(nn.Module):
    def __init__(self, K, d, L_tilde, num_node):
        super().__init__()
        self.K = K
        self.d = d
        self.L_tilde = L_tilde
        self.num_node = num_node

        self.spatialAtt = spatialAttention(K, d)
        self.temporalAtt = temporalAttention(K, d)
        self.causal_inception = Inception_Temporal_Layer(self.num_node, self.K * self.d, self.K * self.d, self.K * self.d)
        self.gated_fusion = gatedFusion(K * d)

    def forward(self, x, te, lap_matrix, node_embedding):
        x_raw = x
        ht1 = self.temporalAtt(x, te)
        ht2 = self.causal_inception(x.permute(0, 2, 1, 3), te).transpose(1, 2)
        ht = self.gated_fusion(ht1, ht2)

        hs = self.spatialAtt(ht, lap_matrix, node_embedding)

        return torch.add(x_raw, hs)


class spatialAttention(nn.Module):
    '''
    spatial attention module
    x: [batch_size, num_step, N, D]
    STE: [batch_size, num_step, N, D]
    K: number of attention heads
    d: dimension of each attention head outputs
    return: [batch_size, num_step, N, K * d]
    '''

    def __init__(self, K, d, dropout=0.1):
        super().__init__()
        self.K = K
        self.d = d

        # 单线性层
        self.FC_Q = nn.Linear(K * d, K * d)
        self.FC_K = nn.Linear(K * d, K * d)
        self.FC_V = nn.Linear(K * d, K * d)

        # pre-norm
        self.norm1 = nn.LayerNorm(K * d)
        self.norm2 = nn.LayerNorm(K * d)

        # 可学习结构权重 alpha, beta
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # 多头合并
        self.out_proj = nn.Linear(self.K * self.d, self.K * self.d)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, 4 * self.K * self.d),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(4 * self.K * self.d, self.K * self.d),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, lapmatrix, node_embeddings, k=3):
        x_raw = x
        B, T, N, D = x.shape

        # x = self.norm1(x)

        query = self.FC_Q(x)
        key = self.FC_K(x)
        value = self.FC_V(x)

        # 多头
        query = query.view(B, T, N, self.K, self.d)
        key = key.view(B, T, N, self.K, self.d)
        value = value.view(B, T, N, self.K, self.d)
        query = query.permute(0, 1, 3, 2, 4).contiguous().view(B * T * self.K, N, self.d)
        key = key.permute(0, 1, 3, 2, 4).contiguous().view(B * T * self.K, N, self.d).transpose(-2, -1)
        value = value.permute(0, 1, 3, 2, 4).contiguous().view(B * T * self.K, N, self.d)

        attention = torch.matmul(query, key) / math.sqrt(self.d)
        raw = attention
        # mask (consider graph structure)
        # torch.mm 矩阵乘法
        supports = F.softmax(F.leaky_relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # 二值化support矩阵 （感觉可以改阈值）
        # supports = supports > 0.5
        supports = supports.unsqueeze(0).expand(B, -1, -1)
        lap = lapmatrix.unsqueeze(0).expand(B, -1, -1)

        adj = (lapmatrix != 0).float()
        mask = (adj == 0) * (-1e9)
        mask = mask.unsqueeze(0).expand(B, -1, -1)

        supports = supports.unsqueeze(1).unsqueeze(2)  # [B,1,1,N,N]
        supports = supports.expand(B, T, self.K, N, N).contiguous().view(B * T * self.K, N, N)
        lap = lap.unsqueeze(1).unsqueeze(2)  # [B,1,1,N,N]
        lap = lap.expand(B, T, self.K, N, N).contiguous().view(B * T * self.K, N, N)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(B, T, self.K, N, N).contiguous().view(B * T * self.K, N, N)

        # attentionS = attention * supports
        # attentionL = attention * lap
        # attention = attentionS + attentionL
        # TODO logits = raw + self.alpha * attentionS + self.beta * attentionL + mask
        logits = raw + self.alpha * supports + self.beta * lap + mask

        attention = torch.softmax(logits, dim=-1)
        attention = self.attn_dropout(attention)

        out = torch.matmul(attention, value)
        out = out.view(B, T, self.K, N, self.d).permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, self.K * self.d)

        x1 = self.out_proj(out)
        x1 = self.proj_dropout(x1)
        # add
        x1 = x_raw + x1
        x1 = self.norm1(x1)

        # y = x + FFN( LN(x) )
        return self.norm2(x1 + self.two_layer_feed_forward(x1))


class Inception_Temporal_Layer(nn.Module):
    def __init__(self, num_stations, In_channels, Hid_channels, Out_channels, kernels=[2, 3, 6, 7]):
        super(Inception_Temporal_Layer, self).__init__()

        self.num_stations = num_stations
        # self.act = nn.LeakyReLU()
        self.d = In_channels

        self.conv = CausalConv1d(In_channels, In_channels, 1, dilation=1, groups=1)

        # Multi-scale branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(Hid_channels, Hid_channels, k),
                nn.LeakyReLU(),
            )
            for k in kernels
        ])

        # Fuse multi-branch output (depthwise 1×1 conv is cleaner)
        self.merge = nn.Conv2d(
            in_channels=1 + len(kernels),  # depth dimension (branches)
            out_channels=1,
            kernel_size=(1, 1),
            groups=1,
            bias=False
        )

        # self.conv1_1 = CausalConv1d(5 * Hid_channels, Out_channels, 1, groups=1)
        self.projection = nn.Linear(Hid_channels, Out_channels)

    def forward(self, inputs, TE):

        # (Batch_size, Number_Station, Seq_len, In_channel)
        B, N, T, C = inputs.shape
        inputs = inputs.reshape(B * N, T, -1).transpose(1, 2)
        inputs = self.conv(inputs)
        # ---- multi-scale branches ----
        outs = [branch(inputs) for branch in self.branches]
        outputs = torch.stack([inputs] + outs, dim=1)

        # ---- merge branches using Conv2d ----
        merged = self.merge(outputs).squeeze(1)  # [B*N, C, T]

        # ---- reshape back to [B, N, T, C] ----
        merged = merged.transpose(1, 2).reshape(B, N, T, C)
        outputs = self.projection(merged)

        return outputs


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        results = super(CausalConv1d, self).forward(inputs)
        padding = self.padding[0]
        # padding = 0
        if padding != 0:
            return results[:, :, :-padding]
        return results


class temporalAttention(nn.Module):
    '''
    temporal attention module
    x: [batch_size, num_step, N, D]
    STE: [batch_size, num_step, N, D]
    K: number of attention heads
    d: dimension of each attention head outputs
    return: [batch_size, num_step, N, K * d]
    '''

    def __init__(self, K, d, kernel_size=3, mask=True, dropout=0.1):
        super().__init__()

        self.K = K
        self.d = d
        self.mask = mask

        # 单层线性层
        self.FC_Q = nn.Linear(K * d, K * d)
        self.FC_K = nn.Linear(K * d, K * d)
        self.FC_V = nn.Linear(K * d, K * d)

        self.FC_Q4HT = nn.Linear(2, K * d)
        self.FC_K4HT = nn.Linear(2, K * d)
        self.FC_V4HT = nn.Linear(2, K * d)

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        # self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(K * d, K * d, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key
        self.conv1Ds_aware_temporal_context = nn.ModuleList([
            nn.Conv2d(K * d, K * d, (1, kernel_size), padding=(0, self.padding)),
            nn.Conv2d(K * d, K * d, (1, kernel_size), padding=(0, self.padding))
        ])

        # pre-norm
        self.norm1 = nn.LayerNorm(K * d)
        self.norm2 = nn.LayerNorm(K * d)

        # 多头合并
        self.out_proj = nn.Linear(self.K * self.d, self.K * self.d)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, 4 * self.K * self.d),
            nn.LeakyReLU(),
            nn.Linear(4 * self.K * self.d, self.K * self.d),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, TE, use_te=True):
        x_raw = x
        B, T, N, D = x.shape
        expected_D = self.K * self.d
        assert D == expected_D, f"last dim D must equal K*d ({expected_D}), got {D}"
        # pre-norm
        # x = self.norm1(x)

        # Post-Norm架构：
        # 1. 先做attention子层 (x -> attn(x) -> x + attn(x) -> norm1)
        # 2. 再做FFN子层   ((x + attn(x)) -> ffn(x + attn(x)) -> x + attn(x) + ffn(...) -> norm2)

        query = self.FC_Q(x)
        key = self.FC_K(x)
        value = self.FC_V(x)

        # TE (B, T, N, 2) -> (B, N, T, 2)
        query4HT = self.FC_Q4HT(TE)
        key4HT = self.FC_K4HT(TE)

        # 时间编码
        if use_te:
            query += query4HT
            key += key4HT

        for conv in self.conv1Ds_aware_temporal_context:
            query = conv(query.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            key = conv(key.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # 多头
        query = query.view(B, T, N, self.K, self.d)
        key = key.view(B, T, N, self.K, self.d)
        value = value.view(B, T, N, self.K, self.d)
        query = query.permute(0, 2, 3, 1, 4).contiguous().view(B * N * self.K, T, self.d)
        key = key.permute(0, 2, 3, 1, 4).contiguous().view(B * N * self.K, T, self.d).transpose(-2, -1)
        value = value.permute(0, 2, 3, 1, 4).contiguous().view(B * N * self.K, T, self.d)

        attention = torch.matmul(query, key) / math.sqrt(self.d)
        attention = torch.softmax(attention, dim=-1)
        # attention (B * N * K * T, B * N * K * T) value (B * N * K * T, d)
        attention = self.attn_dropout(attention)
        # out (B * N * K * T, d)
        out = torch.matmul(attention, value)
        out = out.view(B, N, self.K, T, self.d).permute(0, 3, 1, 2, 4).contiguous().view(B, T, N, self.K * self.d)
        # output linear projection and dropout
        x1 = self.out_proj(out)
        x1 = self.proj_dropout(x1)
        # add
        x1 = x_raw + x1
        x1 = self.norm1(x1)  # post-norm for attention

        # y = x + FFN( LN(x) )
        return self.norm2(x1 + self.two_layer_feed_forward(x1))


class gatedFusion(nn.Module):
    '''
    gated fusion module
    HS: [batch_size, num_step, N, D]
    HT: [batch_size, num_step, N, D]
    D: output dims
    return: [batch_size, num_step, N, D]
    '''

    def __init__(self, D):
        super().__init__()
        self.D = D

        self.FC_HT1 = FeedForward([D, D])
        self.FC_HT2 = FeedForward([D, D])

        self.gate = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.LeakyReLU(),
            nn.Linear(D, D),
        )

    def forward(self, HT1, HT2):
        # residual 结构保留原始信息
        HT1 = HT1 + self.FC_HT1(HT1)
        HT2 = HT2 + self.FC_HT2(HT2)
        concatenated = torch.cat((HT1, HT2), dim=-1)
        gate_value = torch.sigmoid(self.gate(concatenated))
        fused_output = gate_value * HT1 + (1 - gate_value) * HT2
        return fused_output