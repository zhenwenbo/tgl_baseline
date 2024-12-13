import torch
import torch.nn as nn

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(TimeSeriesEmbedding, self).__init__()
        # 时间编码
        self.time_embedding = nn.Linear(1, d_model)
        self.value_embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, d_model), nn.ReLU()  # 可选额外的位置编码层
        )
        
        # Transformer 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        
        # 池化和投影
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化
        self.projection = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # 假设输入 x 的 shape 为 [batch_size, seq_len, 2]
        time = x[:, :, 0].unsqueeze(-1)  # 提取时间列
        value = x[:, :, 1].unsqueeze(-1)  # 提取数值列
        
        time_embed = self.time_embedding(time)
        value_embed = self.value_embedding(value)
        
        # 合并时间和数值的嵌入
        embed = self.pos_encoder(torch.cat((time_embed, value_embed), dim=-1))
        
        # Transformer 编码
        embed = embed.permute(1, 0, 2)  # 转换为 [seq_len, batch_size, d_model]
        encoded = self.transformer_encoder(embed)
        
        # 转换回 [batch_size, seq_len, d_model]
        encoded = encoded.permute(1, 0, 2)
        
        # 全局池化
        pooled = self.pooling(encoded.permute(0, 2, 1)).squeeze(-1)
        
        # 投影到最终嵌入维度
        output = self.projection(pooled)
        return output


timeE = TimeSeriesEmbedding()
x = torch.arange(0, 100, dtype = torch.float32, device = 'cuda:0')
t = torch.arange(100, 200, dtype = torch.float32, device = 'cuda:0')

x = torch.stack((x,t)).reshape(1, -1, 2)