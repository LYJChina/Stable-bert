import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import TwitterDataset, image_transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def create_data_loader(df, tokenizer, max_len, batch_size,  image_dir, seed):
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
