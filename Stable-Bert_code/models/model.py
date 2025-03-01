import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

# # vit+bert+cat
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, dropout_prob=0.1, pre_trained_model_name="bert-base-uncased"):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.register_buffer('pre_features', torch.zeros(16, 1536))
        self.register_buffer('pre_weight1', torch.ones(16, 1))
        combined_feature_size = self.bert.config.hidden_size + self.vit.config.hidden_size

        self.drop = nn.Dropout(p=dropout_prob)

        self.fc = nn.Linear(combined_feature_size, n_classes)

    def forward(self, input_ids, attention_mask, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.pooler_output
        
        vit_outputs = self.vit(pixel_values=image)
        image_features = vit_outputs.pooler_output
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Dropout 层
        x = self.drop(combined_features)
        flatten_features = combined_features
        logits = self.fc(x)
        return logits, x

# resnet+bert+cat
# import torch
# import torch.nn as nn
# from transformers import BertModel
# from torchvision import models

# class SentimentClassifier(nn.Module):
#     def __init__(self, n_classes, dropout_prob=0.1, pre_trained_model_name="bert-base-uncased"):
#         super(SentimentClassifier, self).__init__()
        
#         
#         self.bert = BertModel.from_pretrained(pre_trained_model_name)
        
#         self.resnet = models.resnet18(pretrained=True)
        
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
#         self.register_buffer('pre_features', torch.zeros(16, 1280))# vit 为1536，resnet为1280
#         self.register_buffer('pre_weight1', torch.ones(16, 1))
        
#         combined_feature_size = self.bert.config.hidden_size + 512  

#         self.drop = nn.Dropout(p=dropout_prob)

#         self.fc = nn.Linear(combined_feature_size, n_classes)

#     def forward(self, input_ids, attention_mask, image):
#         bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         text_features = bert_outputs.pooler_output  # [batch_size, hidden_size]
    
#         image_features = self.resnet(image)  # [batch_size, 512, 1, 1]
#         image_features = image_features.view(image_features.size(0), -1)  
        
#         combined_features = torch.cat((text_features, image_features), dim=1)  # [batch_size, hidden_size + 512]

#         x = self.drop(combined_features)
#         flatten_features = x
#         logits = self.fc(x)
#         return logits, flatten_features
