import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from dataset import TwitterDataset

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

            outputs, x = model(input_ids=input_ids, attention_mask=attention_mask, image=images)

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
            all_labels, all_predictions, average='weighted', zero_division=0
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
