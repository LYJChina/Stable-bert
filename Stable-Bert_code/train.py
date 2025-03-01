import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from dataset import TwitterDataset
from torch.autograd import Variable
from reweighting import weight_learner
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch, args):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        images = d["image"].to(device)

        outputs, cfeatures = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
        pre_features = model.pre_features.to(device)
        pre_weight1 = model.pre_weight1.to(device)
        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, d)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).to(device))

        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)
        _, preds = torch.max(outputs, dim=1)
        # print("weight1_shape = ", weight1.shape)
        # print("loss_shape = ", loss_fn(outputs, targets).shape)
        loss_fn1 = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        loss = loss_fn1(outputs, targets).view(1, -1).mm(weight1).view(1)
        # loss = torch.mean(loss).view(1)
        correct_predictions += torch.sum(preds == targets).item()
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions / n_examples, np.mean(losses)
def configure_optimizer_and_scheduler(model, train_data_loader, epochs, learning_rate, num_warmup_steps):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler
