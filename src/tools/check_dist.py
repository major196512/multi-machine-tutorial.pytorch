import torch
import logging
from ..engine import launch, default_argument_parser

logging.basicConfig(level=logging.INFO)

def main(args):
    from ..dist import shared_random_seed, get_rank
    print(f'cuda_device({torch.cuda.current_device()}) Distributed Rank({get_rank()}) Shared Seed({shared_random_seed()})')

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
