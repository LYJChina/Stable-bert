import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from typing import Optional

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
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))  # 计算注意力系数
        x = x * attn_factor  
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_value_projection = nn.Linear(key_dim, query_dim)  # 共享key和value的projection

    def forward(self, query, key, value):
        query = self.query_projection(query)
        key_value = self.key_value_projection(key)
        
        attn_output, _ = self.multihead_attention(query=query, key=key_value, value=value)
        return attn_output

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.1, pre_trained_model_name="bert-base-uncased"):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
        self.caa = CAA(channels=self.vit.config.hidden_size)
        
        self.text_attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=8, dropout=dropout_prob)
        self.text_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.text_ffn = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size * 4, self.bert.config.hidden_size)
        )
        self.text_ffn_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.cross_attention = CrossAttentionLayer(query_dim=self.vit.config.hidden_size, key_dim=self.bert.config.hidden_size, value_dim=self.bert.config.hidden_size)
        combined_feature_size = self.bert.config.hidden_size + self.vit.config.hidden_size
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(combined_feature_size, n_classes)
        self.register_buffer('pre_features', torch.zeros(16, 1536))
        self.register_buffer('pre_weight1', torch.ones(16, 1)) 

    def forward(self, input_ids, attention_mask, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        text_pooler = bert_outputs.pooler_output  # [batch_size, hidden_size]

        image_features = self.vit(pixel_values=image).pooler_output  # [batch_size, hidden_size]
        image_features = self.caa(image_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [batch_size, hidden_size]

        text_features = text_features.transpose(0, 1)  # [seq_len, batch_size, hidden_size]

        attn_output, _ = self.text_attention(text_features, text_features, text_features)  # [seq_len, batch_size, hidden_size]
        attn_output = self.text_norm(attn_output + text_features)  # 残差连接和层归一化
        ffn_output = self.text_ffn(attn_output)  # [seq_len, batch_size, hidden_size]
        ffn_output = self.text_ffn_norm(ffn_output + attn_output)  # 残差连接和层归一化

        ffn_output = ffn_output.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        text_features = torch.mean(ffn_output, dim=1)  # [batch_size, hidden_size]

        cross_attn_output = self.cross_attention(image_features.unsqueeze(0), text_features.unsqueeze(0), text_features.unsqueeze(0))  # [1, batch_size, hidden_size]
        cross_attn_output = cross_attn_output.squeeze(0)  # [batch_size, hidden_size]
        combined_features = torch.cat((cross_attn_output, image_features), dim=1)  # [batch_size, 2 * hidden_size]

        x = self.drop(combined_features)
        logits = self.fc(x)
        return logits, combined_features
