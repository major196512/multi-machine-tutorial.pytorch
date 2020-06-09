import torch
import logging
from ..engine import launch, default_argument_parser

logging.basicConfig(level=logging.INFO)

'''
cuda_device(0) : tensor([[139., 143., 146.,  ..., 112.,  98.,  95.],
        [170., 169., 177.,  ..., 255., 255., 255.],
        [ 44.,  50.,  53.,  ..., 109., 106., 108.],
        ...,
        [101.,  96.,  94.,  ...,  76., 137., 109.],
        [225., 177., 164.,  ..., 172., 128., 204.],
        [143., 142., 143.,  ..., 254., 255., 255.]])
cuda_device(0) : [tensor(6.), tensor(8.), tensor(7.), tensor(5.), tensor(5.), tensor(3.), tensor(2.), tensor(2.)]
'''

def main(args):
    images_per_batch = 8
    train_shuffle = True

    from ..utils.env import seed_all_rng
    from ..dist import shared_random_seed, synchronize, get_world_size
    from ..data import build_train_loader

    seed = args.seed
    if seed == -1 : seed = shared_random_seed()
    seed_all_rng(seed)

    loader = build_train_loader(images_per_batch=images_per_batch, world_size=get_world_size(), shuffle=train_shuffle, seed=seed)
    loader_iter = iter(loader)
    data = next(loader_iter)

    img_list = torch.cat([d['img'][None, :] for d in data], dim=0)
    ann_list = [d['ann'] for d in data]
    print(f'cuda_device({torch.cuda.current_device()}) : {img_list}')
    synchronize()
    print(f'cuda_device({torch.cuda.current_device()}) : {ann_list}')

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
