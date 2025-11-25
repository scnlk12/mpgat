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
        # self.num_node = 358
        #
        # self.embed_dim = 50
        #
        # self.skip_dim = 256
        self.num_node = num_node
        self.embed_dim = embed_dim
        self.skip_dim = skip_dim

        # 对节点编码
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        # 两层网络
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, self.K * self.d),
            nn.LayerNorm(self.K * self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.K * self.d, self.K * self.d),
            nn.LayerNorm(self.K * self.d),
        )

        # 网络结构
        self.STE_emb = STEmbedding(model_dim, K, d, lap_mx)
        self.ST_Att = nn.ModuleList(
            [
                STAttBlock(K, d, LAP)
                for _ in range(L)
            ]
        )

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=K * d, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(L)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=1, kernel_size=1, bias=True,
        )

    def forward(self, x, TE):
        D = self.K * self.d
        # input
        x = self.feed_forward(x)

        # STE
        STE_P = []

        skip = 0

        # encoder
        for i, att in enumerate(self.ST_Att):
            x = att(x, STE_P, TE, self.LAP, self.node_embeddings)
            skip += self.skip_convs[i](x.permute(0, 3, 2, 1))

        # output
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))

        return skip.permute(0, 3, 2, 1).squeeze(-1)


class STEmbedding(nn.Module):
    def __init__(self, model_dim, K, d, lap_mx, drop=0.):
        super().__init__()
        self.K = K
        self.d = d
        self.lap_mx = lap_mx

        lape_dim = 32

        self.x_forward = nn.Sequential(
            nn.Linear(self.K * self.d, self.K * self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.K * self.d, self.K * self.d),
        )

        # self.spatial_embedding = LaplacianPE(lape_dim, self.K * self.d)
        # self.se_forward = FC(input_dims=[self.K * self.d, self.K * self.d], units=[self.K * self.d, self.K * self.d], activations=[F.relu, None])

        # self.te_forward = nn.Sequential(
        #     nn.Linear(48, self.K * self.d),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.K * self.d, self.K * self.d),
        # )
        # self.te_forward = FC(input_dims=[48, self.K * self.d], units=[self.K * self.d, self.K * self.d], activations=[F.relu, None])

        # self.tod_embedding = nn.Embedding(288, 24)
        # self.dow_embedding = nn.Embedding(7, 24)

        # self.tem_embedding = TemEmbedding(self.K * self.d)

        # Ea (STAEformer论文)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(12, 358, self.K * self.d))
        )

        # 位置编码 positional embedding 
        # self.position_encoding = PositionalEncoding(self.K * self.d)

        self.dropout = nn.Dropout(drop)

        # self.linear_emb = nn.Linear(4 * self.K * self.d, self.K * self.d)

    def forward(self, x, TE):
        # return torch.add(torch.add(self.SE, TE), adp_emb)
        # one-hot编码方式
        # TE = self.tem_embedding(TE)

        B, T, N, D = TE.shape
        # Ea 
        # adp_emb = self.adaptive_embedding.expand(
        #         size=(B, *self.adaptive_embedding.shape)
        #     )  # (batch_size, in_steps, num_nodes, adaptive_embedding_dim)

        # Ea
        adp_emb = self.adaptive_embedding.expand(
            size=(B, *self.adaptive_embedding.shape)
        )  # (batch_size, in_steps, num_nodes, adaptive_embedding_dim)

        x_raw = x
        x = self.x_forward(x)

        # x += self.position_encoding(x_raw)
        # x += self.SE
        # 时间（周期）编码 + 空间编码（节点Laplacian结构）+ 时空adaptive embedding
        x += adp_emb
        # x += self.spatial_embedding(self.lap_mx)
        # x += TE
        x = self.dropout(x)

        return x


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class TemEmbedding(nn.Module):
    def __init__(self, D):
        super(TemEmbedding, self).__init__()
        self.ff_te = FeedForward([295, D, D])

    def forward(self, TE, T=288):
        '''
        TE: [B, T, N, 2]
        return: [B, T, N, D]
        '''
        B, Times, N, D = TE.shape
        TE = TE[:, :, 0, :]
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # [B,T,7]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # [B,T,288]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B,T,295]
        TE = TE.unsqueeze(dim=2)  # [B,T,1,295]
        TE = TE.repeat(1, 1, N, 1)
        TE = self.ff_te(TE)  # [B,T,1,F]

        return TE  # [B,T,N,F]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.ModuleList([nn.LayerNorm(fea[i + 1]) for i in range(self.L)])

    def forward(self, inputs):
        # x (B, T, 1, F)
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            x = self.ln[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
        return x


class STAttBlock(nn.Module):
    def __init__(self, K, d, L_tilde):
        super().__init__()
        self.K = K
        self.d = d

        self.L_tilde = L_tilde

        self.spatialAtt = spatialAttention(K, d)
        self.temporalAtt = temporalAttention(K, d)

        self.gated_fusion = gatedFusion(K * d)

        # add & norm 
        self.norm = nn.LayerNorm(K * d)

        self.causal_inception = Inception_Temporal_Layer(358, self.K * self.d, self.K * self.d, self.K * self.d)

    def forward(self, x, STE, TE, lapMatrix, node_embedding):
        x_raw = x
        HT1 = self.temporalAtt(x, STE, TE)
        HT1 = self.norm(HT1)

        HT2 = self.causal_inception(x.permute(0, 2, 1, 3), TE).transpose(1, 2)
        HT2 = self.norm(HT2)
        HT = self.gated_fusion(HT1, HT2)

        HS, _ = self.spatialAtt(HT, lapMatrix, node_embedding)

        return torch.add(x_raw, HS)


class spatialAttention(nn.Module):
    '''
    spatial attention module
    x: [batch_size, num_step, N, D]
    STE: [batch_size, num_step, N, D]
    K: number of attention heads
    d: dimension of each attention head outputs
    return: [batch_size, num_step, N, K * d]
    '''

    def weighted_k_hop_matrix(adj_matrix, k):
        """
        计算带权邻接矩阵的 k-hop 矩阵
        :param adj_matrix: 带权邻接矩阵 (n x n)
        :param k: hop 数
        :return: k-hop 矩阵 (n x n)
        """
        result = np.linalg.matrix_power(adj_matrix, k)  # 计算 A^k
        return result

    def __init__(self, K, d):
        super().__init__()
        self.K = K
        self.d = d

        # 1-alpha使用自身更新 
        # self.alpha = 0.7

        # self.FC_Q = nn.Linear(K * d, K * d)
        # self.FC_K = nn.Linear(K * d, K * d)
        # self.FC_V = nn.Linear(K * d, K * d)
        self.FC_Q = FeedForward([K * d, K * d])
        self.FC_K = FeedForward([K * d, K * d])
        self.FC_V = FeedForward([K * d, K * d])

        self.dropout = nn.Dropout(p=0.1)

        # 自身更新
        # self.FC_ii_V = nn.Linear(K * d, K * d)

        # embed_dim = 64
        # self.num_node = 358
        # self.node_embeddings = nn.Parameter(torch.randn(358, 10), requires_grad=True)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, self.K * self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.K * self.d, self.K * self.d),
        )

    def forward(self, x, lapmatrix, node_embeddings):
        D = self.K * self.d
        x_raw = x
        # B, T, n, D = x.shape

        batch_size = x.shape[0]

        query = self.FC_Q(x)
        key = self.FC_K(x)
        value = self.FC_V(x)
        # value_ii = self.FC_ii_V(x)

        query = torch.concat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.concat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.concat(torch.split(value, self.K, dim=-1), dim=0)
        # value_ii = torch.concat(torch.split(value_ii, self.K, dim=-1), dim=0)

        key = key.transpose(2, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)

        # mask (consider graph structure)
        # torch.mm 矩阵乘法
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)

        # 二值化support矩阵 （感觉可以改阈值）
        # supports = supports > 0.5
        supports = supports.unsqueeze(0).unsqueeze(0)

        k = 3

        # print("lapmatrix.shape", lapmatrix.shape)
        matrix = lapmatrix
        lapmatrix = torch.tensor(lapmatrix.todense(),
                                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).unsqueeze(
            0).unsqueeze(0)
        lapmatrixk = torch.tensor(np.linalg.matrix_power(matrix.todense(), k),
                                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).unsqueeze(
            0).unsqueeze(0)

        # attention = attention.masked_fill_(support_bool, -torch.inf)

        # attention 增加两个邻接矩阵A 基于距离的A + AGCRN中的A tanh（ZZT） 以及他们的topK矩阵
        # print("attention.shape", attention.shape)
        # print("supports.shape", supports.shape)
        # print("lapmatrix.shape", lapmatrix.shape)
        # attention = attention * supports * lapmatrix * lapmatrixk
        attentionS = attention * supports
        attentionL = attention * lapmatrix
        # attentionK = attention * lapmatrixk
        # attention = attentionS + attentionK + attentionL
        attention = attentionS + attentionL

        attention = nn.Softmax(dim=-1)(attention)
        # print('attention.size()', attention.size())
        attention = self.dropout(attention)
        attention = attention.float()

        # attention_ii = torch.zeros(attention.shape[0], attention.shape[1], attention.shape[2], attention.shape[2],
        #    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        # 创建对角线索引
        # indices = torch.arange(attention.shape[2])

        # 利用广播和高级索引填充对角线元素
        # attention_ii[:, :, indices, indices] = (1 - self.alpha)

        # x = self.alpha * torch.matmul(attention, value) + torch.matmul(attention_ii, value_ii)
        x = torch.matmul(attention, value)
        # g = torch.sigmoid(torch.matmul(attention, value) + value_ii)
        # x = g * torch.matmul(attention, value) + (1 - g) * value_ii
        x = torch.concat(torch.split(x, batch_size, dim=0), dim=-1)
        x = self.two_layer_feed_forward(x)

        attention = torch.concat(torch.split(attention, batch_size, dim=0), dim=1)

        return torch.add(x, x_raw), attention


# ASTGCN论文中的spatial gcn
class cheb_conv_with_SAt(nn.Module):
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
    '''

    def __init__(self, num_of_filters, K, cheb_polynomials, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv_with_SAt, self).__init__(**kwargs)
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials

        # self.Theta = self.params.get('Theta', allow_deferred_init=True)

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        Theta = torch.empty(self.K, num_of_features, 64)
        nn.init.xavier_uniform_(Theta)
        # Theta._finish_deferred_init()

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices, 64,
                                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            for k in range(self.K):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention[:, time_step, :, :]

                # shape of theta_k is (F, num_of_filters)
                theta_k = Theta[k]

                # shape is (batch_size, V, F)
                rhs = torch.bmm(T_k_with_at.float(), graph_signal)
                # rhs = rhs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                theta_k = theta_k.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))
        return torch.relu(torch.concat(outputs, dim=-1))


class Inception_Temporal_Layer(nn.Module):
    def __init__(self, num_stations, In_channels, Hid_channels, Out_channels):
        super(Inception_Temporal_Layer, self).__init__()

        self.conv = CausalConv1d(In_channels, In_channels, 1, dilation=1, groups=1)
        # 改变下层卷积核和上层卷积核的大小 1 ✖️ 7 1 ✖️ 6 1 ✖️ 3 1 ✖️ 2
        # 原卷积核大小为 1 ✖️ 3 1 ✖️ 2 1 ✖️ 2
        self.temporal_conv1 = CausalConv1d(In_channels, Hid_channels, 7, dilation=1, groups=1)
        # init.xavier_normal_(self.temporal_conv1.weight)

        self.temporal_conv4 = CausalConv1d(Hid_channels, Hid_channels, 6, dilation=1, groups=1)

        self.temporal_conv2 = CausalConv1d(Hid_channels, Hid_channels, 3, dilation=1, groups=1)
        # init.xavier_normal_(self.temporal_conv2.weight)

        self.temporal_conv3 = CausalConv1d(Hid_channels, Hid_channels, 2, dilation=1, groups=1)
        # init.xavier_normal_(self.temporal_conv3.weight)

        self.conv1_1 = CausalConv1d(5 * Hid_channels, Out_channels, 1, groups=1)

        self.num_stations = num_stations
        self.act = nn.LeakyReLU(inplace=True)
        self.d = In_channels

        self.FC_Q = FeedForward([In_channels, In_channels])
        self.FC_K = FeedForward([In_channels, In_channels])
        self.FC_V = FeedForward([In_channels, In_channels])

        self.FC_Q4HT = FeedForward([2, In_channels])
        self.FC_K4HT = FeedForward([2, In_channels])
        self.FC_V4HT = FeedForward([2, In_channels])

        self.dropout = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(In_channels)

    def forward(self, inputs, TE):
        query = self.FC_Q(inputs)
        key = self.FC_K(inputs)
        value = self.FC_V(inputs)

        # query = query.transpose(1, 2)
        # key = key.transpose(1, 2).transpose(2, 3)
        # value = value.transpose(1, 2)
        key = key.transpose(2, 3)

        query4HT = self.FC_Q4HT(TE.transpose(1, 2))
        key4HT = self.FC_K4HT(TE.transpose(1, 2))
        key4HT = key4HT.transpose(2, 3)

        attention = torch.matmul(query4HT, key4HT)
        attention /= (self.d ** 0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, value)
        x = x + inputs
        x1 = self.norm1(x)

        # (Batch_size, Number_Station, Seq_len, In_channel)
        B, N, T, _ = inputs.shape
        inputs = inputs.reshape(B * N, T, -1).transpose(1, 2)
        # inputs = torch.stack([self.conv(inputs[:, s_i].transpose(1, 2)).transpose(1, 2)
        #                         for s_i in range(self.num_stations)], dim=1)
        inputs = self.conv(inputs)
        # output_1 = torch.stack([self.temporal_conv1(inputs[:, s_i].transpose(1, 2)).transpose(1, 2)
        #                         for s_i in range(self.num_stations)], dim=1)
        # output_1 = self.act(output_1)
        output_1 = self.temporal_conv1(inputs)
        output_1 = self.act(output_1)

        output_2 = self.temporal_conv2(inputs)
        output_2 = self.act(output_2)

        output_3 = self.temporal_conv3(inputs)
        output_3 = self.act(output_3)

        output_4 = self.temporal_conv4(inputs)
        output_4 = self.act(output_4)

        outputs = torch.cat([inputs, output_1, output_4, output_2, output_3], dim=-2)

        outputs = self.conv1_1(outputs)
        outputs = outputs.reshape(B, N, T, -1)

        return outputs + x1


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

    def __init__(self, K, d, mask=True):
        super().__init__()

        self.K = K
        self.d = d
        self.mask = mask

        # self.FC_Q = nn.Linear(K * d, K * d)
        # self.FC_K = nn.Linear(K * d, K * d)
        # self.FC_V = nn.Linear(K * d, K * d)
        self.FC_Q = FeedForward([K * d, K * d])
        self.FC_K = FeedForward([K * d, K * d])
        self.FC_V = FeedForward([K * d, K * d])

        self.FC_Q4HT = FeedForward([2, K * d])
        self.FC_K4HT = FeedForward([2, K * d])
        self.FC_V4HT = FeedForward([2, K * d])

        # pems04 
        kernel_size = 3

        self.padding = (kernel_size - 1) // 2

        # self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(K * d, K * d, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key
        self.conv1Ds_aware_temporal_context = nn.ModuleList([
            nn.Conv2d(K * d, K * d, (1, kernel_size), padding=(0, self.padding)),
            nn.Conv2d(K * d, K * d, (1, kernel_size), padding=(0, self.padding))
        ])

        dropout = 0.1

        self.norm1 = nn.LayerNorm(K * d)

        self.dropout = nn.Dropout(p=dropout)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, self.K * self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.K * self.d, self.K * self.d),
        )

        self.norm2 = nn.LayerNorm(K * d)

        # self-attention 和 temb 融合
        self.mlp = nn.Sequential(
            nn.Linear(24, K * d),
            nn.ReLU(),
            nn.Linear(K * d, 2)  # 输出两个权重
        )

    def forward(self, x, STE, TE):
        D = self.K * self.d

        x_raw = x
        batch_size = x.shape[0]
        b, L, n, d = x.shape

        query = self.FC_Q(x)

        # key 和 value要改掉
        key = self.FC_K(x)
        value = self.FC_V(x)

        for conv in self.conv1Ds_aware_temporal_context:
            query = conv(query.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            key = conv(key.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        query = torch.concat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.concat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.concat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)

        query4HT = self.FC_Q4HT(TE.transpose(1, 2))
        key4HT = self.FC_K4HT(TE.transpose(1, 2))

        query4HT = torch.concat(torch.split(query4HT, self.K, dim=-1), dim=0)
        key4HT = torch.concat(torch.split(key4HT, self.K, dim=-1), dim=0)
        # value4HT = self.FC_V4HT(TE.transpose(1, 2))
        # print("query4HT.size()", query4HT.shape)
        # print("key4HT.shape", key4HT.shape)
        key4HT = key4HT.transpose(2, 3)

        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)

        # attention4HT = torch.matmul(query4HT, key4HT)
        attention4HT = torch.matmul(query4HT, key)
        attention4HT /= (self.d ** 0.5)

        # print("attention.shape", attention.shape)
        # print("attention4HT.shape", attention4HT.shape)

        # num_step = x.shape[1]
        # mask = torch.ones(num_step, num_step, dtype=bool, device=query.device).tril()
        # attention = attention.masked_fill_(~mask, -torch.inf)

        # mask attention score
        # if self.mask:
        #     num_step = x.shape[1]
        #     mask = torch.ones(num_step, num_step, dtype=bool, device=query.device).tril()
        #     attention = attention.masked_fill_(~mask, -torch.inf)
        # 1. 点积
        # attention = attention * attention4HT
        # 2. 求和
        # attention = torch.add(attention, attention4HT)
        # 3. mlp求权值
        # combined = torch.cat((attention, attention4HT), dim=-1)  # 拼接
        # print("combined.shape", combined.shape)
        # weights = torch.softmax(self.mlp(combined), dim=-1)       # 生成权重
        # print("weights.shape", weights.shape)
        # attention = weights[..., 0].unsqueeze(-1) * attention + weights[..., 1].unsqueeze(-1) * attention4HT

        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2)
        x = torch.concat(torch.split(x, batch_size, dim=0), dim=-1)
        x = x + x_raw
        x1 = self.norm1(x)

        x = self.two_layer_feed_forward(x1)
        x = self.norm2(x1 + x)

        return x


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

        # self.feed_forward = nn.Linear(D, D)

        # self.FC_xs = FC(input_dims=D, units=D, activations=None)
        self.FC_HT1 = FeedForward([D, D])
        # self.FC_xt = FC(input_dims=D, units=D, activations=None)
        self.FC_HT2 = FeedForward([D, D])

        self.gate = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
        )

        # 定义滤波器卷积和门控卷积
        # self.filter_conv = nn.Conv1d(D * 2, D, kernel_size=1)
        # self.gate_conv = nn.Conv1d(D * 2, D, kernel_size=1)

        # self.two_layer_feed_forward = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None])

    def forward(self, HT1, HT2):
        HT1 = self.FC_HT1(HT1)
        # XS = self.FC_xs(XS)
        HT2 = self.FC_HT2(HT2)
        # XT = self.FC_xt(XT)

        # z = torch.sigmoid(torch.add(XS, XT))
        # H = torch.add(torch.multiply(z, XS), torch.multiply(1-z, XT))

        concatenated = torch.cat((HT1, HT2), dim=-1)
        # print('concatenated.shape', concatenated.shape)

        # Compute gate value (sigmoid to get values between 0 and 1)
        gate_value = torch.sigmoid(self.gate(concatenated))

        # Fuse inputs using the gate
        fused_output = gate_value * HT1 + (1 - gate_value) * HT2
        # H = torch.cat((XS, XT), axis=-1)
        # H = self.two_layer_feed_forward(H)

        # # 滤波器卷积
        # filter = self.filter_conv(concatenated)
        # filter = torch.tanh(filter)  # 使用 Tanh 激活函数

        # # 门控卷积
        # gate = self.gate_conv(concatenated)
        # gate = torch.sigmoid(gate)  # 使用 Sigmoid 激活函数

        # # 门控融合 增加非线性
        # fused_output = filter * gate  # 逐元素相乘

        # return H
        return fused_output


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


class Adaptive_Fusion(nn.Module):
    def __init__(self, head, dims):
        super(Adaptive_Fusion, self).__init__()
        self.h = head
        self.d = dims
        D = head * dims
        self.D = D
        features = D

        self.qlfc = FeedForward([features, features])
        # self.qlfc = nn.Linear(D, D)
        self.khfc = FeedForward([features, features])
        # self.khfc = nn.Linear(D, D)
        self.vhfc = FeedForward([features, features])
        # self.vhfc = nn.Linear(D, D)
        self.ofc = FeedForward([features, features])
        # self.ofc = nn.Linear(D, D)

        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        # self.ff = FeedForward([features, features, features], True)
        self.ff = nn.Linear(D, D)

    def forward(self, xl, xh, Mask=False):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,N,F]
        return: [B,T,N,F]
        '''
        # xl += te
        # xh += te

        query = self.qlfc(xl)  # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh))  # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh))  # [B,T,N,F]

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]
        keyh = torch.cat(torch.split(keyh, self.d, -1), 0).permute(0, 2, 3, 1)  # [k*B,N,d,T]
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)  # [k*B,N,T,d]

        attentionh = torch.matmul(query, keyh)  # [k*B,N,T,T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)  # [T,T]
            mask = torch.tril(mask)  # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)  # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)  # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)  # scaled
        attentionh = F.softmax(attentionh, -1)  # [k*B,N,T,T]

        value = torch.matmul(attentionh, valueh)  # [k*B,N,T,d]

        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)  # [B,T,N,F]
        value = self.ofc(value)
        value = value + xl

        value = self.ln(value)

        return self.ff(value)


class transformAttention(nn.Module):
    '''
    transform attention module
    x: [batch_size, P, N, D]
    K: number of attention heads
    d: dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''

    def __init__(self, K, d):
        super().__init__()
        self.K = K
        self.d = d

        self.feed_forward = nn.Linear(self.K * self.d, self.K * self.d)
        # self.FC_q = FC(input_dims=self.K * self.d, units=self.K * self.d, activations=F.relu)
        # self.FC_k = FC(input_dims=self.K * self.d, units=self.K * self.d, activations=F.relu)
        # self.FC_v = FC(input_dims=self.K * self.d, units=self.K * self.d, activations=F.relu)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, K * d),
            nn.ReLU(inplace=True),
            nn.Linear(K * d, K * d),
        )
        # self.two_layer_feed_forward = FC(input_dims=self.K * self.d, units=self.K * self.d, activations=F.relu)

    def forward(self, x, STE_P, STE_Q):
        D = self.K * self.d

        query = self.feed_forward(STE_Q)
        # query = self.FC_q(STE_Q)
        key = self.feed_forward(STE_P)
        # key = self.FC_k(STE_P)
        value = self.feed_forward(x)
        # value = self.FC_v(x)

        batch_size = x.shape[0]

        query = torch.concat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.concat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.concat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.transpose(1, 2)
        x = torch.concat(torch.split(x, batch_size, dim=0), dim=-1)
        x = self.two_layer_feed_forward(x)

        return x


def compute_topK_patches(i, sequence1, sequence2_list, memory_items_length):
    distance_dtw = []
    for j in range(memory_items_length):
        dist, _ = fastdtw(sequence1[i, :, :], sequence2_list[j].detach().cpu().numpy()[i, :, :])
        distance_dtw.append(dist)
    # 从小到大排序
    distance_dtw_with_index = sorted(enumerate(distance_dtw), reverse=False, key=lambda x: x[1])
    # topK_patchs_index
    topK_patchs_index = [index for index, value in distance_dtw_with_index[:int(memory_items_length * 2 / 3)]]
    # topK_patchs
    topK_patchs = [sequence2_list[idx][i, :, :] for idx in topK_patchs_index]
    return topK_patchs
