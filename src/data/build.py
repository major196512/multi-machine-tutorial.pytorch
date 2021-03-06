import torch
import numpy as np

from .datasets import DatasetFromDict
from .sampler import IterSampler
from .loader import cifar_10_loader
from ..utils.env import seed_all_rng

__all__ = ['build_train_loader']

def build_train_loader(images_per_batch, world_size=1, num_workers=4, seed=None, shuffle=True):
    assert images_per_batch > world_size
    assert images_per_batch % world_size == 0

    data = cifar_10_loader(train=True)
    dataset = DatasetFromDict(data)
    sampler = IterSampler(size=len(dataset), shuffle=shuffle, seed=seed)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_batch // world_size, drop_last=True
            )  # drop last so the batch always have the same size

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader

def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
