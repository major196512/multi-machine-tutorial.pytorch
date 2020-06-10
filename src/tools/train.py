import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

import logging
from ..engine import launch, default_argument_parser

logging.basicConfig(level=logging.INFO)

def main(args):
    max_iter = 100
    images_per_batch = 8
    num_workers = 2
    device='cuda'

    from ..utils.env import seed_all_rng
    from ..dist import shared_random_seed, synchronize, get_world_size, get_rank, get_local_rank, is_main_process
    from ..data import build_train_loader
    from ..modeling import ToyModel

    seed = args.seed
    if seed == -1 : seed = shared_random_seed()
    seed_all_rng(seed)

    model = ToyModel(device=device)
    if get_world_size() > 1:
        model = DistributedDataParallel(model, device_ids=[get_local_rank()], broadcast_buffers=False)

    loader = build_train_loader(images_per_batch=images_per_batch, num_workers=num_workers, world_size=get_world_size(), shuffle=True, seed=seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for curr_iter in range(max_iter):
        optimizer.zero_grad()
        loader_iter = iter(loader)
        data = next(loader_iter)

        img = torch.cat([d['img'][None, :] for d in data], dim=0).to(device)
        ann = torch.tensor([d['ann'] for d in data]).long().to(device)
        output = model(img)
        loss = loss_fn(output, ann)
        loss.backward()
        optimizer.step()

        if is_main_process() and curr_iter % 10 == 0:
            print(f'iteration({curr_iter}) : loss({loss.item():.5f})')

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(f"Command Line Args:{args}")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_ip=args.dist_ip,
        dist_port=args.dist_port,
        args=(args,),
    )
