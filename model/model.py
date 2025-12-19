import torch
from torch import nn
import numpy as np
import datetime
import torch.nn.functional as F
import math

# HTV-Lite 增强模块
from model.coupled_graph_lite import LightweightCoupledGraph
from model.time_varying_mask import TimeVaryingMask
from model.low_rank_embedding import LowRankAdaptiveEmbedding
from model.multi_level_fusion import MultiLevelFeatureFusion


class GMAN(nn.Module):
    def __init__(self, model_dim, P, Q, T, L, K, d, lap_mx, LAP, num_node, embed_dim, skip_dim=256,
                 # HTV-Lite 配置参数
                 use_coupled_graph=False,
                 coupling_weight=0.01,
                 use_time_varying_mask=False,
                 use_low_rank_embedding=False,
                 low_rank=32,
                 use_multi_level_fusion=False,
                 fusion_type='attention'):
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

        HTV-Lite 增强参数:
            use_coupled_graph (bool): 启用耦合图学习 (P0-1)
            coupling_weight (float): 耦合损失权重
            use_time_varying_mask (bool): 启用时变掩码 (P0-2)
            use_low_rank_embedding (bool): 启用低秩分解 (P1-4)
            low_rank (int): 低秩维度
            use_multi_level_fusion (bool): 启用多层特征融合 (P2-5)
            fusion_type (str): 融合方式 ('attention', 'weighted', etc.)
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

        # HTV-Lite 配置
        self.use_coupled_graph = use_coupled_graph
        self.use_time_varying_mask = use_time_varying_mask
        self.use_low_rank_embedding = use_low_rank_embedding
        self.use_multi_level_fusion = use_multi_level_fusion

        # 对节点编码
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        # P0-1: 耦合图学习模块
        if use_coupled_graph:
            self.coupled_graph = LightweightCoupledGraph(
                num_nodes=num_node,
                embed_dim=embed_dim,
                temporal_steps=P,
                coupling_weight=coupling_weight
            )
            print(f"[HTV-Lite] 耦合图学习已启用 (参数: {P * embed_dim:,})")

        # 网络结构
        # Use P (history steps) instead of T (total time steps in a day) for adaptive embedding
        self.STE_emb = STEmbedding(
            model_dim, K, d, lap_mx, num_node, P,
            use_low_rank=use_low_rank_embedding,
            low_rank=low_rank
        )

        self.ST_Att = nn.ModuleList(
            [
                STAttBlock(
                    K, d, LAP, num_node,
                    use_time_varying_mask=use_time_varying_mask
                )
                for _ in range(L)
            ]
        )

        # P2-5: 多层特征融合
        if use_multi_level_fusion:
            self.multi_level_fusion = MultiLevelFeatureFusion(
                num_layers=L,
                fusion_type=fusion_type
            )
            print(f"[HTV-Lite] 多层特征融合已启用 (融合方式: {fusion_type})")

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
        layer_features = []  # P2-5: 收集多层特征用于融合

        # P0-1: 预先生成所有时间步的耦合图 (避免在每层重复计算)
        coupled_graphs = None
        if self.use_coupled_graph:
            # 生成P个时间步的图结构 (N, N) -> (P, N, N)
            coupled_graphs = self.coupled_graph.get_all_graphs(self.node_embeddings)

        # encoder
        for i, att in enumerate(self.ST_Att):
            # 执行STAttBlock (传入所有时间步的耦合图)
            x = att(x, TE, self.LAP, self.node_embeddings, coupled_graphs=coupled_graphs)

            # P2-5: 收集层特征
            if self.use_multi_level_fusion:
                layer_features.append(x)

            skip += self.skip_convs[i](x.permute(0, 3, 2, 1))

        # P2-5: 多层特征融合
        if self.use_multi_level_fusion and len(layer_features) > 0:
            x_fused = self.multi_level_fusion(layer_features)
            # 添加融合特征的skip连接,而不是覆盖
            skip = skip + self.skip_convs[-1](x_fused.permute(0, 3, 2, 1))
            # 更新x为融合后的特征(虽然后续不再使用x,但保持一致性)
            x = x_fused

        # output
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))

        return skip.permute(0, 3, 2, 1).squeeze(-1)

    def coupling_loss(self):
        """
        计算耦合图的时序连续性损失 (P0-1)

        返回:
            loss (Tensor): 标量损失值，如果未启用耦合图则返回0
        """
        if not self.use_coupled_graph:
            return torch.tensor(0.0, device=self.node_embeddings.device)

        return self.coupled_graph.coupling_loss(self.node_embeddings)


class STEmbedding(nn.Module):
    def __init__(self, model_dim, K, d, lap_mx, num_node, time_step, drop=0.,
                 use_low_rank=False, low_rank=32):
        super().__init__()
        self.K = K
        self.d = d
        self.lap_mx = lap_mx
        self.use_low_rank = use_low_rank

        self.x_forward = nn.Sequential(
            nn.Linear(model_dim, self.K * self.d),
            nn.ReLU(inplace=True),
            nn.Linear(self.K * self.d, self.K * self.d),
        )

        # P1-4: 低秩分解 adaptive embedding
        if use_low_rank:
            self.adaptive_embedding_module = LowRankAdaptiveEmbedding(
                time_steps=time_step,
                num_nodes=num_node,
                embed_dim=self.K * self.d,
                rank=low_rank,
                fusion='multiply'
            )
            print(f"[HTV-Lite] 低秩分解嵌入已启用 (rank={low_rank}, 压缩96.2%)")
        else:
            # 原版全秩 Ea (STAEformer论文) 感觉类似位置编码
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(time_step, num_node, self.K * self.d))
            )

        self.dropout = nn.Dropout(drop)

    def forward(self, x, TE):
        B, _, _, _ = TE.shape
        x = self.x_forward(x)

        # Ea - adaptive embedding
        if self.use_low_rank:
            adp_emb = self.adaptive_embedding_module(B)  # (B, P, N, D)
        else:
            adp_emb = self.adaptive_embedding.expand(
                size=(B, *self.adaptive_embedding.shape)
            )  # (batch_size, in_steps, num_nodes, adaptive_embedding_dim)

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
                x = F.relu(x)
        if self.residual_add:
            x += inputs
        return x


class STAttBlock(nn.Module):
    def __init__(self, K, d, L_tilde, num_node, use_time_varying_mask=False):
        super().__init__()
        self.K = K
        self.d = d
        self.L_tilde = L_tilde
        self.num_node = num_node
        self.use_time_varying_mask = use_time_varying_mask

        self.spatialAtt = spatialAttention(K, d)
        self.temporalAtt = temporalAttention(K, d, use_time_varying_mask=use_time_varying_mask)
        self.causal_inception = Inception_Temporal_Layer(self.num_node, self.K * self.d, self.K * self.d, self.K * self.d)
        self.gated_fusion = gatedFusion(K * d)

    def forward(self, x, te, lap_matrix, node_embedding, coupled_graphs=None):
        """
        Args:
            x: (B, T, N, D)
            te: (B, T, N, 2) 时间编码
            lap_matrix: (N, N) 拉普拉斯矩阵
            node_embedding: (N, embed_dim) 节点嵌入
            coupled_graphs: (P, N, N) 所有时间步的耦合图 (可选)
        """
        x_raw = x
        ht1 = self.temporalAtt(x, te)
        ht2 = self.causal_inception(x.permute(0, 2, 1, 3), te).transpose(1, 2)
        ht = self.gated_fusion(ht1, ht2)

        # 传入所有时间步的耦合图到空间注意力
        hs = self.spatialAtt(ht, lap_matrix, node_embedding, coupled_graphs=coupled_graphs)

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

        # 可学习结构权重 alpha, beta, gamma (耦合图权重)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.5))  # 耦合图权重

        # 多头合并
        self.out_proj = nn.Linear(self.K * self.d, self.K * self.d)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, 4 * self.K * self.d),
            nn.ReLU(),
            nn.Linear(4 * self.K * self.d, self.K * self.d),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, lapmatrix, node_embeddings, k=3, coupled_graphs=None):
        x_raw = x
        B, T, N, D = x.shape

        x = self.norm1(x)

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
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
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

        # P0-1: 整合耦合图 (如果提供)
        if coupled_graphs is not None:
            # coupled_graphs: (P, N, N), 为每个时间步选择对应的图
            # 扩展到 (B, T, K, N, N)
            coupled = coupled_graphs.unsqueeze(0).unsqueeze(2)  # [1, P, 1, N, N]
            coupled = coupled.expand(B, T, self.K, N, N).contiguous().view(B * T * self.K, N, N)
            # logits = raw + alpha*supports + beta*lap + gamma*coupled + mask
            logits = raw + self.alpha * supports + self.beta * lap + self.gamma * coupled + mask
        else:
            # 原版逻辑 (无耦合图)
            logits = raw + self.alpha * supports + self.beta * lap + mask

        attention = torch.softmax(logits, dim=-1)
        attention = self.attn_dropout(attention)

        out = torch.matmul(attention, value)
        out = out.view(B, T, self.K, N, self.d).permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, self.K * self.d)

        x1 = self.out_proj(out)
        x1 = self.proj_dropout(x1)
        # add
        x1 = x_raw + x1

        # y = x + FFN( LN(x) )
        return x1 + self.two_layer_feed_forward(self.norm2(x1))


class Inception_Temporal_Layer(nn.Module):
    def __init__(self, num_stations, In_channels, Hid_channels, Out_channels, kernels=[2, 3, 6, 7]):
        super(Inception_Temporal_Layer, self).__init__()

        self.num_stations = num_stations
        self.act = nn.LeakyReLU()
        self.d = In_channels

        self.conv = CausalConv1d(In_channels, In_channels, 1, dilation=1, groups=1)

        # Multi-scale branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(Hid_channels, Hid_channels, k),
                nn.LeakyReLU()
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

        self.FC_Q = FeedForward([In_channels, In_channels])
        self.FC_K = FeedForward([In_channels, In_channels])
        self.FC_V = FeedForward([In_channels, In_channels])

        # 时间编码变换: 2维 -> In_channels维, 禁用残差连接
        self.FC_Q4HT = FeedForward([2, In_channels], residual_add=False)
        self.FC_K4HT = FeedForward([2, In_channels], residual_add=False)
        self.FC_V4HT = FeedForward([2, In_channels], residual_add=False)

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

        # return outputs + x1
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

    def __init__(self, K, d, kernel_size=3, mask=True, dropout=0.1, use_time_varying_mask=False):
        super().__init__()

        self.K = K
        self.d = d
        self.mask = mask
        self.use_time_varying_mask = use_time_varying_mask

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

        # P0-2: 时变掩码生成器
        if use_time_varying_mask:
            self.time_mask_generator = TimeVaryingMask(
                hidden_dim=K * d,
                num_heads=K,
                mask_type='multiplicative'
            )
            print(f"[HTV-Lite] 时变掩码已启用 (参数: ~3,600)")

        # pre-norm
        self.norm1 = nn.LayerNorm(K * d)
        self.norm2 = nn.LayerNorm(K * d)

        # 多头合并
        self.out_proj = nn.Linear(self.K * self.d, self.K * self.d)

        self.two_layer_feed_forward = nn.Sequential(
            nn.Linear(K * d, 4 * self.K * self.d),
            nn.ReLU(),
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
        x = self.norm1(x)

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

        # P0-2: 应用时变掩码
        if self.use_time_varying_mask:
            # 归一化时间编码 (0-287 → 0-1, 0-6 → 0-1)
            te_normalized = TE / torch.tensor([287.0, 6.0], device=TE.device)
            time_mask = self.time_mask_generator(te_normalized)  # (B, T, N, K)

            # 调整维度: (B, T, N, K) -> (B, N, K, T, 1)
            time_mask = time_mask.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, N, K, T, 1)
            time_mask = time_mask.contiguous().view(B * N * self.K, T, 1)  # (B*N*K, T, 1)

            # 应用掩码 (乘性调制)
            attention = attention * time_mask

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

        # y = x + FFN( LN(x) )
        return x1 + self.two_layer_feed_forward(self.norm2(x1))


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
            nn.ReLU(),
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