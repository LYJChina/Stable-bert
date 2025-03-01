import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from typing import Optional

# 保持 ConvModule 和 CAA 类不变
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
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        x = x * attn_factor
        return x

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
    def __init__(self, n_classes, dropout_prob=0, pre_trained_model_name="bert-base-uncased"):
        super(SentimentClassifier, self).__init__()
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        # 加载预训练的 ViT 模型
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.bert_hidden_size = self.bert.config.hidden_size  # 768
        self.vit_hidden_size = self.vit.config.hidden_size    # 768

        # 定义 CAA 模块用于图像特征增强
        self.caa = CAA(channels=self.vit.config.hidden_size)

        # 定义 Cross Attention Layer
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.vit.config.hidden_size, 
            key_dim=self.bert.config.hidden_size, 
            value_dim=self.bert.config.hidden_size
        )

        # 定义门控机制，确保模型更关注文本特征
        self.gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

        # 合并文本和图像特征后的维度
        combined_feature_size = self.vit.config.hidden_size + self.bert.config.hidden_size  # 768 + 768 = 1536
        
        # 增加 Dropout 层防止过拟合
        self.drop = nn.Dropout(p=dropout_prob)

        # 分类层
        self.fc = nn.Linear(combined_feature_size, n_classes)
        self.register_buffer('pre_features', torch.zeros(64, 1536))
        self.register_buffer('pre_weight1', torch.ones(64, 1)) 
        # # 初始化门控层的偏置，使其偏向于文本特征9/*+
        
        # nn.init.constant_(self.gate[0].bias, 2.0)  # sigmoid(2) ≈ 0.88，偏向于文本

    def forward(self, input_ids, attention_mask, image):
        # 获取 BERT 模型的输出
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 使用 BERT 的 pooler_output 作为文本特征
        text_pooler = bert_outputs.pooler_output  # [batch_size, hidden_size]

        # 通过 ViT 模型获取图像特征
        image_features = self.vit(pixel_values=image).pooler_output  # [batch_size, hidden_size]

        # 通过 CAA 模块增强图像特征
        image_features = self.caa(image_features.unsqueeze(-1).unsqueeze(-1))  # [batch_size, hidden_size, 1, 1]

        # 调整图像特征的形状为 [batch_size, hidden_size]
        image_features = image_features.view(image_features.size(0), image_features.size(1))  # [batch_size, hidden_size]

        # 计算门控值
        gate_value = self.gate(text_pooler)  # [batch_size, 1]

        # 使用交叉注意力融合图像和文本特征
        # 将图像特征作为查询，文本特征作为键和值
        # 注意：MultiheadAttention 的输入形状为 [seq_len, batch_size, embed_dim]
        cross_attn_output = self.cross_attention(
            image_features.unsqueeze(0),      # [1, batch_size, hidden_size]
            text_pooler.unsqueeze(0),         # [1, batch_size, hidden_size]
            text_pooler.unsqueeze(0)          # [1, batch_size, hidden_size]
        )  # [1, batch_size, hidden_size]

        # 移除序列长度维度，得到 [batch_size, hidden_size]
        cross_attn_output = cross_attn_output.squeeze(0)  # [batch_size, hidden_size]

        # 根据门控值调整特征的贡献并拼接，得到 [batch_size, 1536]
        combined_features = torch.cat((cross_attn_output, text_pooler), dim=1)  # [batch_size, 1536]

        # Dropout 层
        x = self.drop(combined_features)

        # 分类层的输出
        logits = self.fc(x)  # [batch_size, n_classes]

        return logits, combined_features