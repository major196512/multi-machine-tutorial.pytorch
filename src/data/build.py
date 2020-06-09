import torch

from .datasets import DatasetFromDict
from .sampler import IterSampler
from .loader import cifar_10_loader

__all__ = ['build_train_loader']

def build_train_loader(images_per_batch, world_size=1, seed=None, shuffle=True):
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
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

def trivial_batch_collator(batch):
    return batch
