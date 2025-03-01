import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights
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
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))  
        x = x * attn_factor  
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_value_projection = nn.Linear(key_dim, query_dim)  
    def forward(self, query, key, value):
        query = self.query_projection(query)
        key = self.key_value_projection(key)
        value = self.key_value_projection(value)
        
        attn_output, _ = self.multihead_attention(query=query, key=key, value=value)
        return attn_output

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.1, pre_trained_model_name="bert-base-uncased", resnet_version='resnet18'):
        super(SentimentClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        
        if resnet_version == 'resnet18':
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            resnet_hidden_size = 512
        else:
            raise NotImplementedError(f"ResNet version '{resnet_version}' is not implemented.")
        
        self.resnet.fc = nn.Identity()  
        self.resnet.avgpool = nn.Identity()  
        
        self.caa = CAA(channels=resnet_hidden_size)

        self.cross_attention = CrossAttentionLayer(
            query_dim=self.bert.config.hidden_size, 
            key_dim=resnet_hidden_size, 
            value_dim=resnet_hidden_size
        )
        
        self.image_proj = nn.Linear(resnet_hidden_size, self.bert.config.hidden_size)
        
        self.gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        combined_feature_size = self.bert.config.hidden_size + self.bert.config.hidden_size  # 768 + 768 = 1536
        
        self.drop = nn.Dropout(p=dropout_prob)
        
        self.fc = nn.Linear(combined_feature_size, n_classes)
        
        self.register_buffer('pre_features', torch.zeros(64, combined_feature_size))
        self.register_buffer('pre_weight1', torch.ones(64, 1)) 

    def forward(self, input_ids, attention_mask, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        text_pooler = bert_outputs.pooler_output  # [batch_size, hidden_size]
        
        image_features = self.resnet(image)  # [batch_size, 512, 7, 7]
        
        image_features = self.caa(image_features)  # [batch_size, 512, 7, 7]
        
        batch_size = image_features.size(0)
        image_features = image_features.view(batch_size, 512, -1)  # [batch_size, 512, 49]
        image_features = image_features.permute(0, 2, 1)  # [batch_size, 49, 512]
        image_features = self.image_proj(image_features)  # [batch_size, 49, 768]
        image_features = image_features.permute(1, 0, 2)  # [49, batch_size, 768]
        
        query = text_pooler.unsqueeze(0)      # [1, batch_size, hidden_size]
        key = image_features                   # [49, batch_size, hidden_size]
        value = image_features                 # [49, batch_size, hidden_size]
        
        cross_attn_output = self.cross_attention(query, key, value)  # [1, batch_size, hidden_size]
        cross_attn_output = cross_attn_output.squeeze(0)             # [batch_size, hidden_size]
        
        gate_value = self.gate(text_pooler)  # [batch_size, 1]
        
        combined_features = torch.cat((cross_attn_output, text_pooler), dim=1)  # [batch_size, 2 * hidden_size]
        
        x = self.drop(combined_features)
        
        logits = self.fc(x)  # [batch_size, n_classes]
        
        return logits, combined_features
