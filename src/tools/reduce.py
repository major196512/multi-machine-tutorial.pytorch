import torch
import logging
from ..engine import launch, default_argument_parser

logging.basicConfig(level=logging.INFO)

def main(args):
    from ..dist import is_main_process, reduce_dict

    local_loss = {
        'loss1' : torch.rand(1).cuda(),
        'loss2' : torch.rand(1).cuda()
    }
    print(f'local({local_loss})')

    avg_loss = reduce_dict(local_loss)
    if is_main_process() :
        print('---------------------------------------------')
        print('          Reduce Average Loss                ')
        print('---------------------------------------------')
        print(f'reduce({avg_loss})')

    sum_loss = reduce_dict(local_loss, average=False)
    if is_main_process() :
        print('---------------------------------------------')
        print('            Reduce Sum Loss                  ')
        print('---------------------------------------------')
        print(f'reduce({sum_loss})')

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
