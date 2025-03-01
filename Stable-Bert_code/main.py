import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import os
import random
import numpy as np
import pandas as pd
import torch
from transformers import logging
from transformers import BertTokenizer
from dataset import TwitterDataset
from dataloader import create_data_loader, set_seed
from train import train_epoch, configure_optimizer_and_scheduler
from eval import eval_model
from sklearn.metrics import f1_score
from models.resnet_stable_caa_text2img import SentimentClassifier
from config import parse_args
from transformers import PreTrainedTokenizerFast
import json
def main():
    args = parse_args()
    logging.set_verbosity_error()
    train_df = pd.read_csv(args.train_tsv, sep="\t")
    val_df = pd.read_csv(args.dev_tsv, sep="\t")
    test_df = pd.read_csv(args.test_tsv, sep="\t")
    train_df = train_df.rename({"#1 Label": "sentiment", "#2 ImageID": "image_id", "#3 String": "tweet_content", "#3 String.1": "target",}, axis=1).drop(["index"], axis=1)
    val_df = val_df.rename({"#1 Label": "sentiment","#2 ImageID": "image_id","#3 String": "tweet_content","#3 String.1": "target",}, axis=1).drop(["index"], axis=1)
    test_df = test_df.rename({"index": "sentiment","#1 ImageID": "image_id","#2 String": "tweet_content","#2 String.1": "target",}, axis=1)
    train_df['sentiment'] = train_df['sentiment'].astype(int)
    val_df['sentiment'] = val_df['sentiment'].astype(int)
    test_df['sentiment'] = test_df['sentiment'].astype(int)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device}.")

    results_per_run = {}
    for run_number in range(args.num_runs):
        seed = args.random_seeds[run_number]
        print("\nseed = ",seed)
        set_seed(seed)
        train_data_loader = create_data_loader(
            train_df, tokenizer, args.max_len, args.batch_size,  args.image_dir, seed
        )
        val_data_loader = create_data_loader(
            val_df, tokenizer, args.max_len, args.batch_size,  args.image_dir, seed
        )
        test_data_loader = create_data_loader(
            test_df, tokenizer, args.max_len, args.batch_size,  args.image_dir, seed
        )

        model = SentimentClassifier(args.num_classes)
        model.to(device)
        optimizer, scheduler = configure_optimizer_and_scheduler(
            model, train_data_loader, args.epochs, args.learning_rate, args.num_warmup_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs} -- RUN {run_number}")
            print("-" * 10)
            train_acc, train_loss = train_epoch(
                model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_df), epoch, args
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

    with open(f'./resnet_bert_stable_caa_text2img_15.json', 'w+') as f:
        json.dump(results_per_run, f)

    print(f"AVERAGE ACCURACY = {np.mean([_['accuracy'] for _ in results_per_run.values()]):.4f}")
    print(f"AVERAGE PRECISION = {np.mean([_['precision'] for _ in results_per_run.values()]):.4f}")
    print(f"AVERAGE RECALL = {np.mean([_['recall'] for _ in results_per_run.values()]):.4f}")
    print(f"AVERAGE WEIGHTED F1 = {np.mean([_['weighted_f1'] for _ in results_per_run.values()]):.4f}")
    print(f"AVERAGE MACRO F1 = {np.mean([_['macro_f1'] for _ in results_per_run.values()]):.4f}")

if __name__ == "__main__":
    main()
