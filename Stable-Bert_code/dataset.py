import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225]),
])

class TwitterDataset(Dataset):
    def __init__(
        self,
        tweets: np.array,
        labels: np.array,
        sentiment_targets: np.array,
        image_ids: np.array,
        tokenizer,
        max_len: int,
        image_dir: str,
        transforms=None,
    ):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.sentiment_targets = sentiment_targets
        self.max_len = max_len
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]
        sentiment_target = self.sentiment_targets[item]
        encoding = self.tokenizer.encode_plus(
            tweet,
            text_pair=sentiment_target,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        image_id = self.image_ids[item]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        if self.transforms:
            image = self.transforms(image)

        return {
            "review_text": tweet,
            "sentiment_targets": sentiment_target,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image": image,
            "targets": torch.tensor(label, dtype=torch.long),
        }
