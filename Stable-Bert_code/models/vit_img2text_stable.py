import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from typing import Optional

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=8)
        self.linear = nn.Linear(query_dim, value_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return self.linear(attn_output)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.1, pre_trained_model_name="bert-base-uncased"):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.vit.config.hidden_size, 
            key_dim=self.bert.config.hidden_size, 
            value_dim=self.bert.config.hidden_size
        )
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
        # image_features = torch.mean(self.vit(pixel_values=image).last_hidden_state, dim=1)  # [batch_size, hidden_size]
        cross_attn_output = self.cross_attention(
            image_features.unsqueeze(0), 
            text_pooler.unsqueeze(0), 
            text_pooler.unsqueeze(0)
        )  # [1, batch_size, hidden_size]
        cross_attn_output = cross_attn_output.squeeze(0)  # [batch_size, hidden_size]
        combined_features = torch.cat((cross_attn_output, text_pooler), dim=1)  # [batch_size, 2 * hidden_size]
        x = self.drop(combined_features)
        logits = self.fc(x)
        return logits, combined_features