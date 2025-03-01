import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from typing import Optional

# 定义卷积模块类，来自mmcv.cnn（保持不变）
class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, norm_cfg: Optional[dict] = None, act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

# 定义上下文锚点注意力 (Context Anchor Attention) 模块（保持不变）
class CAA(nn.Module):
    def __init__(self, channels: int, h_kernel_size: int = 11, v_kernel_size: int = 11, norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001), act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels, norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels, norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))  # 计算注意力系数
        x = x * attn_factor  # 应用注意力系数
        return x

# 定义交叉注意力 (Cross-Attention) 模块
class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_value_projection = nn.Linear(key_dim, query_dim)  # 共享key和value的projection

    def forward(self, query, key, value):
        # 通过投影生成查询、键和值
        query = self.query_projection(query)
        key_value = self.key_value_projection(key)
        
        # 交叉注意力
        attn_output, _ = self.multihead_attention(query=query, key=key_value, value=value)
        return attn_output

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.1, pre_trained_model_name="bert-base-uncased"):
        super(SentimentClassifier, self).__init__()
        
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        
        # 加载预训练的 ViT 模型
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 定义 CAA 模块用于图像特征增强
        self.caa = CAA(channels=self.vit.config.hidden_size)
        
        # 定义 Cross Attention Layer
        # 这里将文本特征作为查询，图像特征作为键和值
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.bert.config.hidden_size, 
            key_dim=self.vit.config.hidden_size, 
            value_dim=self.vit.config.hidden_size
        )
        
        # 可选：定义门控机制
        self.gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 计算 BERT 和 ViT 模型的特征维度并合并
        combined_feature_size = self.bert.config.hidden_size + self.vit.config.hidden_size  # 修改为 hidden_size + hidden_size
        
        # Dropout 层防止过拟合
        self.drop = nn.Dropout(p=dropout_prob)
        
        # 分类层
        self.fc = nn.Linear(combined_feature_size, n_classes)
        
        # 注册缓冲区（如果需要保留）
        self.register_buffer('pre_features', torch.zeros(64, 1536))
        self.register_buffer('pre_weight1', torch.ones(64, 1)) 

    def forward(self, input_ids, attention_mask, image):
        # 获取 BERT 模型的输出
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用 BERT 的 pooler_output 作为文本特征
        text_pooler = bert_outputs.pooler_output  # [batch_size, hidden_size]
        
        # 通过 ViT 模型获取图像特征
        image_features = self.vit(pixel_values=image).pooler_output  # [batch_size, hidden_size]
        
        # 通过 CAA 模块增强图像特征
        # 假设 CAA 模块需要的输入形状是 [batch_size, channels, 1, 1]
        image_features = self.caa(image_features.unsqueeze(-1).unsqueeze(-1))  # [batch_size, hidden_size, 1, 1]
        
        # 调整图像特征的形状为 [batch_size, hidden_size]
        image_features = image_features.view(image_features.size(0), image_features.size(1))  # [batch_size, hidden_size]

        # 可选：计算门控值
        gate_value = self.gate(text_pooler)  # [batch_size, 1]
        
        # 使用交叉注意力融合文本和图像特征
        # 注意：MultiheadAttention 的输入形状为 [seq_len, batch_size, embed_dim]
        # 这里将文本特征作为查询，图像特征作为键和值
        cross_attn_output = self.cross_attention(
            text_pooler.unsqueeze(0),      # [1, batch_size, hidden_size]
            image_features.unsqueeze(0),   # [1, batch_size, hidden_size]
            image_features.unsqueeze(0)    # [1, batch_size, hidden_size]
        )  # [1, batch_size, hidden_size]
        
        # 移除序列长度维度，得到 [batch_size, hidden_size]
        cross_attn_output = cross_attn_output.squeeze(0)  # [batch_size, hidden_size]
        
        # 根据门控值调整图像特征的贡献
        # 当 gate_value 接近 1 时，更多依赖文本特征
        # 当 gate_value 接近 0 时，更多依赖图像特征
        combined_features = torch.cat((cross_attn_output, text_pooler), dim=1)  # [batch_size, 2 * hidden_size]
        
        # 可选：使用门控值调整最终特征
        # combined_features = gate_value * text_pooler + (1 - gate_value) * cross_attn_output
        
        # Dropout 层
        x = self.drop(combined_features)
        
        # 分类层的输出
        logits = self.fc(x)  # [batch_size, n_classes]
        
        return logits, combined_features
