# Multi Machine Tutorial for Pytorch
<p align="center"><img width="40%" src="./img/pytorch.png"></p>

## Introduction
xxxxx

## Requirements
* pytorch : 1.5
* CUDA : 10.1

## Base

### Single Machine
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2
```

```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2 --dist_port 47515
```

### Multi Machines
* Main Machine

```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2 --num-machine 2 --machine-rank 0
```

* Other Machines

```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.check_dist --num-gpu 2 --num-machine 2 --machine-rank 1 --dist_ip xxx.xxx.xxx.xxx --dist_port xxxxx
```

## Test

### Gather
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.gather --num-gpu 2
```

### Reduce
```bash
CUDA_VISIBLE_DEVICES='0,1' python -m src.tools.reduce --num-gpu 2
```
