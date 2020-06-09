import torch
import logging
from ..engine import launch, default_argument_parser

logging.basicConfig(level=logging.INFO)

def main(args):
    from ..utils.env import seed_all_rng
    from ..dist import shared_random_seed, get_rank, get_world_size, is_main_process, synchronize
    from ..dist import gather, all_gather

    seed_all_rng(shared_random_seed())
    world_size = get_world_size()
    local_rank = get_rank()

    if is_main_process() :
        print('---------------------------------------------')
        print('            All Gather Method                ')
        print('---------------------------------------------')
    local_tensor = torch.tensor([torch.rand(world_size)[local_rank]])

    print(f'cuda_device({torch.cuda.current_device()}) : local({local_tensor})')
    tensor = all_gather(local_tensor)
    print(f'cuda_device({torch.cuda.current_device()}) : gather({tensor})')

    synchronize()

    if is_main_process() :
        print('---------------------------------------------')
        print('              Gather Method                  ')
        print('---------------------------------------------')
    local_tensor = torch.tensor([torch.rand(world_size)[local_rank]])
    print(f'cuda_device({torch.cuda.current_device()}) : local({local_tensor})')
    tensor = gather(local_tensor, dst=0)
    print(f'cuda_device({torch.cuda.current_device()}) : gather({tensor})')

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
