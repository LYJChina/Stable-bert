import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import os
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    ViTModel,
)

train_tsv = "/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015/train.tsv"
dev_tsv = "/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015/dev.tsv"
test_tsv = "/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015/test.tsv"
IMAGE_DIR = "/home/yinxx23/data/qgfx/IJCAI2019_data/IJCAI2019_data/twitter2015_images"

PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 80
BATCH_SIZE = 32
DROPOUT_PROB = 0.1
NUM_CLASSES = 3
DEVICE = "cuda:5"
EPOCHS = 10
LEARNING_RATE = 5e-5
ADAMW_CORRECT_BIAS = True
NUM_WARMUP_STEPS = 0
NUM_RUNS = 10
RANDOM_SEEDS = list(range(NUM_RUNS))


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

test_df = pd.read_csv(test_tsv, sep="\t")
train_df = pd.read_csv(train_tsv, sep="\t")
val_df = pd.read_csv(dev_tsv, sep="\t")

test_df = test_df.rename(
    {
        "index": "sentiment",
        "#1 ImageID": "image_id",
        "#2 String": "tweet_content",
        "#2 String.1": "target",
    },
    axis=1,
)
train_df = train_df.rename(
    {
        "#1 Label": "sentiment",
        "#2 ImageID": "image_id",
        "#3 String": "tweet_content",
        "#3 String.1": "target",
    },
    axis=1,
).drop(["index"], axis=1)
val_df = val_df.rename(
    {
        "#1 Label": "sentiment",
        "#2 ImageID": "image_id",
        "#3 String": "tweet_content",
        "#3 String.1": "target",
    },
    axis=1,
).drop(["index"], axis=1)

train_df['sentiment'] = train_df['sentiment'].astype(int)
val_df['sentiment'] = val_df['sentiment'].astype(int)
test_df['sentiment'] = test_df['sentiment'].astype(int)

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

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
            # "caption": caption,  # 移除 caption
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image": image,
            "targets": torch.tensor(label, dtype=torch.long),
        }

def create_data_loader(df, tokenizer, max_len, batch_size, image_dir, seed):
    ds = TwitterDataset(
        tweets=df.tweet_content.to_numpy(),
        labels=df.sentiment.to_numpy(),
        sentiment_targets=df.target.to_numpy(),
        image_ids=df.image_id.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        image_dir=image_dir,
        transforms=image_transforms,
    )
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")


import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
# 构建和实例化分类器
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        combined_feature_size = self.bert.config.hidden_size + self.vit.config.hidden_size

        self.drop = nn.Dropout(p=DROPOUT_PROB)
        self.fc = nn.Linear(combined_feature_size, n_classes)

    def forward(self, input_ids, attention_mask, image):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.pooler_output

        vit_outputs = self.vit(pixel_values=image)
        image_features = vit_outputs.pooler_output

        combined_features = torch.cat((text_features, image_features), dim=1)

        x = self.drop(combined_features)
        logits = self.fc(x)
        return logits

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        images = d["image"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets).item()
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions / n_examples, np.mean(losses)

def format_eval_output(rows):
    tweets, targets, labels, predictions = zip(*rows)
    results_df = pd.DataFrame()
    results_df["tweet"] = tweets
    results_df["target"] = targets
    results_df["label"] = labels
    results_df["prediction"] = predictions
    return results_df

def eval_model(model, data_loader, loss_fn, device, n_examples, detailed_results=False):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_labels = []
    all_predictions = []
    rows = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            images = d["image"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())
            all_labels.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

            if detailed_results:
                rows.extend(
                    zip(
                        d["review_text"],
                        d["sentiment_targets"],
                        targets.cpu().numpy(),
                        preds.cpu().numpy(),
                    )
                )

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )

        if detailed_results:
            return (
                correct_predictions / n_examples,
                np.mean(losses),
                precision,
                recall,
                f1,
                format_eval_output(rows),
            )

    return correct_predictions / n_examples, np.mean(losses), precision, recall, f1

results_per_run = {}
for run_number in range(NUM_RUNS):
    seed = RANDOM_SEEDS[run_number]
    set_seed(seed)

    train_data_loader = create_data_loader(
        train_df, tokenizer, MAX_LEN, BATCH_SIZE, IMAGE_DIR, seed
    )
    val_data_loader = create_data_loader(
        val_df, tokenizer, MAX_LEN, BATCH_SIZE, IMAGE_DIR, seed
    )
    test_data_loader = create_data_loader(
        test_df, tokenizer, MAX_LEN, BATCH_SIZE, IMAGE_DIR, seed
    )

    data = next(iter(train_data_loader))
    model = SentimentClassifier(NUM_CLASSES)
    model.to(device)
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    images = data["image"].to(device)
    model(input_ids, attention_mask, images)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS} -- RUN {run_number}")
        print("-" * 10)
        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_df)
        )

        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")
        val_acc, val_loss, val_precision, val_recall, val_f1 = eval_model(
            model, val_data_loader, loss_fn, device, len(val_df)
        )

        print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f} precision {val_precision:.4f} recall {val_recall:.4f} F1 {val_f1:.4f}")

    test_acc, _, test_precision, test_recall, test_f1, detailed_results = eval_model(
        model, test_data_loader, loss_fn, device, len(test_df), detailed_results=True
    )
    macro_f1 = f1_score(
        detailed_results.label.astype(int), detailed_results.prediction.astype(int), average="macro"
    )

    print(f"TEST ACC = {test_acc:.4f}")
    print(f"PRECISION = {test_precision:.4f}")
    print(f"RECALL = {test_recall:.4f}")
    print(f"WEIGHTED F1 = {test_f1:.4f}")
    print(f"MACRO F1 = {macro_f1:.4f}")

    results_per_run[run_number] = {
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "weighted_f1": test_f1,
        "macro_f1": macro_f1
    }

with open(f'./vit_bert_base_17_text2img.json', 'w+') as f:
    json.dump(results_per_run, f)

print(f"AVERAGE ACCURACY = {np.mean([_['accuracy'] for _ in results_per_run.values()]):.4f}")
print(f"AVERAGE PRECISION = {np.mean([_['precision'] for _ in results_per_run.values()]):.4f}")
print(f"AVERAGE RECALL = {np.mean([_['recall'] for _ in results_per_run.values()]):.4f}")
print(f"AVERAGE WEIGHTED F1 = {np.mean([_['weighted_f1'] for _ in results_per_run.values()]):.4f}")
print(f"AVERAGE MACRO F1 = {np.mean([_['macro_f1'] for _ in results_per_run.values()]):.4f}")
