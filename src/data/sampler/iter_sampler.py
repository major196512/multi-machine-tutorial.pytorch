import torch
from torch.utils.data.sampler import Sampler

import itertools
from typing import Optional
from ... import dist

class IterSampler(Sampler):
    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = dist.shared_random_seed()
        self._seed = int(seed)

        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)
