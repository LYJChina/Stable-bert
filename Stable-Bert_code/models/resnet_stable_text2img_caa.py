import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torchvision.models import ResNet50_Weights
from typing import Optional
import torch.nn.functional as F
# 保持 ConvModule 类不变
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

# 保持 CAA 类不变
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

        
# 保持 ConvModule 类不变
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

# 保持 CAA 类不变
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

# 修改 CrossAttentionLayer 类以适应文本查询图像
class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_projection = nn.Linear(key_dim, query_dim)
        self.value_projection = nn.Linear(value_dim, query_dim)

    def forward(self, query, key, value):
        # 通过投影生成查询、键和值
        query = self.query_projection(query)  # [batch_size, query_len, query_dim]
        key = self.key_projection(key)        # [batch_size, key_len, query_dim]
        value = self.value_projection(value)  # [batch_size, value_len, query_dim]

        # 调整维度为 [seq_len, batch_size, embed_dim]，多头注意力要求的形状
        query = query.transpose(0, 1)  # [batch_size, query_len, embed_dim] -> [query_len, batch_size, embed_dim]
        key = key.transpose(0, 1)      # [batch_size, key_len, embed_dim] -> [key_len, batch_size, embed_dim]
        value = value.transpose(0, 1)  # [batch_size, value_len, embed_dim] -> [value_len, batch_size, embed_dim]

        # 交叉注意力
        attn_output, _ = self.multihead_attention(query=query, key=key, value=value)

        # 将输出还原为 [batch_size, seq_len, embed_dim] 形式
        attn_output = attn_output.transpose(0, 1)  # [seq_len, batch_size, embed_dim] -> [batch_size, seq_len, embed_dim]

        return attn_output

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, pre_trained_model_name='bert-base-uncased', resnet_model_name='resnet50', dropout_prob=0.3):
        super(SentimentClassifier, self).__init__()

        # 初始化BERT模型
        self.bert = BertModel.from_pretrained(pre_trained_model_name)

        # 初始化ResNet模型，并去除全连接层
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 输出 [batch_size, 2048, H, W]

        # CAA模块
        self.caa = CAA(channels=2048)

        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 [batch_size, 2048, 1, 1]

        # 投影层，将图像特征降维到与文本特征相同的维度
        self.image_projection = nn.Linear(2048, self.bert.config.hidden_size)  # [batch_size, 768]
        self.text_projection = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)  # 保持维度不变

        # Cross Attention Layer：文本查询图像
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.bert.config.hidden_size,  # 768 for bert-base-uncased
            key_dim=self.image_projection.out_features,  # 768
            value_dim=self.image_projection.out_features,  # 768
            num_heads=8
        )

        combined_feature_size = self.bert.config.hidden_size * 2  # 768 * 2 = 1536

        # 保留缓冲区
        # 注意：这里假设批量大小为64，如果批量大小可能变化，需要调整缓冲区的使用方式
        self.register_buffer('pre_features', torch.zeros(128, combined_feature_size))
        self.register_buffer('pre_weight1', torch.ones(128, 1)) 

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(combined_feature_size, combined_feature_size),
            nn.Sigmoid()
        )

        # Dropout和全连接层
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(combined_feature_size, n_classes)

    def forward(self, input_ids, attention_mask, image):
        # 获取BERT文本特征
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]

        # 获取ResNet图像特征
        image_features = self.feature_extractor(image)  # [batch_size, 2048, H, W]
        image_features = self.caa(image_features)      # [batch_size, 2048, H, W]
        image_features = self.global_pool(image_features).view(image_features.size(0), -1)  # [batch_size, 2048]

        # 将图像特征进行投影
        image_features = self.image_projection(image_features)  # [batch_size, 768]

        # 将文本特征进行投影（保持维度不变）
        text_features = self.text_projection(text_features)  # [batch_size, seq_length, 768]

        # 使用文本特征作为查询，图像特征作为键和值
        image_features_expanded = image_features.unsqueeze(1)  # [batch_size, 1, 768]

        cross_attn_output = self.cross_attention(
            query=text_features,          # [batch_size, seq_length, 768]
            key=image_features_expanded,  # [batch_size, 1, 768]
            value=image_features_expanded # [batch_size, 1, 768]
        )  # 输出: [batch_size, seq_length, 768]

        # 对交叉注意力后的文本特征进行池化（平均池化）
        cross_attn_output = cross_attn_output.mean(dim=1)  # [batch_size, 768]

        # 对文本特征进行池化（例如，使用平均池化）
        text_features_pooled = text_features.mean(dim=1)  # [batch_size, 768]

        # 将交叉注意力后的输出与池化后的文本特征拼接
        combined_features = torch.cat((cross_attn_output, text_features_pooled), dim=1)  # [batch_size, 1536]

        # 应用门控机制
        gate_values = self.gate(combined_features)  # [batch_size, 1536]
        combined_features = combined_features * gate_values  # 加权融合

        # Dropout层
        x = self.drop(combined_features)  # [batch_size, 1536]
        # 全连接层进行分类
        logits = self.fc(x)  # [batch_size, n_classes]
        return logits, x