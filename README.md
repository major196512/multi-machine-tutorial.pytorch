<p align="center"><img width="40%" src="./img/pytorch.png"></p>

# Multi Machine Tutorial for Pytorch
It works TCP communication for multi-gpu processing.
They automatically find unused port address and 

## Requirements
* pytorch : 1.5
* CUDA : 10.1

## Multi GPU in Single Machine
For only multi-gpu processing in single machine, you only need to clarify `num-gpu` argument.
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2
```


## Multi Machines
### Main Machine
[For collective communication](https://tutorials.pytorch.kr/intermediate/dist_tuto.html#collective-communication) in pytorch, it needs to execute process in main machine.
They automatically set main machine ip address and unused port number for TCP communication.
`num-machine`
set `machine-rank` to zero.
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2 --num-machine 2 --machine-rank 0
```

When you want to use a fixed port number, just clarify `dist-port` argument.
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2 --num-machine 2 --machine-rank 0 --dist-port xxxxx
```

### Other Machines
In other machines, you clarify `machine-rank` within the range of 1~(num_machine-1).
And you must set `dist-ip` and `dist-port` arguments which is the same with main machine values.

```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2 --num-machine 2 --machine-rank 1 --dist-ip xxx.xxx.xxx.xxx --dist-port xxxxx
```

## Test
Examples for collective communication functions in single machine.
It also can be executed in multi-machine settings.
### Gather
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.gather --num-gpu 2
```

### Reduce
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.reduce --num-gpu 2
```
