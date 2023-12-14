# command: OMP_NUM_THREADS=32 torchrun --nnodes 1 --nproc_per_node 8 ./run_training.py

import yaml

from trainer import Trainer
from configs import *

import os
import torch
from networks import TransformerModel
from torch.distributed import init_process_group, destroy_process_group
from data import get_train_dataset
import random


# import pickle
# torchrun --nnodes 1 --nproc_per_node 1 ./training/run_training.py 
def ddp_setup():
    init_process_group(backend="nccl")
    print(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_train_objs(cfg,):
    model_config = ModelConfig(**cfg['model_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])

    train_set = get_train_dataset(sequence_length=model_config.max_seq_len)

    model = TransformerModel(model_config)
    if cfg['compile']:
        model = torch.compile(model)

    optimizer = create_optimizer(model, opt_cfg)
    if cfg['compile']:
        model = torch.compile(model)
    lr_scheduler = None
    if 'lr_scheduler_config' in cfg:
        lr_scheduler_config = LRSchedulerConfig(**cfg['lr_scheduler_config'])
        warmup_config = None
        if 'warmup_config' in cfg:
            warmup_config = WarmUpConfig(**cfg['warmup_config'])

        lr_scheduler = create_lr_scheduler(optimizer, lr_scheduler_config, warmup_config)

    return model, optimizer, train_set, None, lr_scheduler


def main(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'), yaml.FullLoader)
    ddp_setup()
    print(cfg)

    trainer_config = TrainerConfig(**cfg['trainer_config'])

    model, optimizer, train_data, test_data, lr_scheduler = get_train_objs(cfg)

    trainer = Trainer(trainer_config, model, optimizer, train_data, test_data, lr_scheduler, cfg)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main('./configs/transformer_config.yaml')
