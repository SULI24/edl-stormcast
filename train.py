# Inspired from https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/cuboid_transformer/sevir/train_cuboid_sevir.py
import warnings
# from typing import Union, Dict
# from shutil import copyfile
# from copy import deepcopy
# import inspect
# import pickle
# import numpy as np
import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
# import torchmetrics
# import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
# from einops import rearrange
from pytorch_lightning import Trainer, seed_everything

from models.lightning_unet import UNetSEVIRPLModule
from sevir.config import cfg
# from utils.optim import SequentialLR, warmup_lambda
# from utils.utils import get_parameter_names
from utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
# from sevir.layout import layout_to_in_out_slice
# from utils.visualization.sevir_vis_seq import save_example_vis_results
# from utils.metrics import SEVIRSkillScore

# _curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
# exps_dir = os.path.join(_curr_dir, "experiments")
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
pytorch_state_dict_name = "_sevir.pt"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['unet', 'gan', 'earthformer'], type=str)
    parser.add_argument('--save', default='tmp_sevir', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on SEVIR.')
    parser.add_argument('--edl', action='store_true',
                        help='Use Evidential Deep Learning on model.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "earthformer_sevir_v1.yaml"))
    print(args.cfg)
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        dataset_oc = OmegaConf.to_object(UNetSEVIRPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = UNetSEVIRPLModule.get_sevir_datamodule(
        dataset_oc=dataset_oc,
        micro_batch_size=micro_batch_size,
        num_workers=8,)
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = UNetSEVIRPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = UNetSEVIRPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        pretrained_ckpt_name = pytorch_state_dict_name
        if not os.path.exists(os.path.join(pretrained_checkpoints_dir, pretrained_ckpt_name)):
            s3_download_pretrained_ckpt(ckpt_name=pretrained_ckpt_name,
                                        save_dir=pretrained_checkpoints_dir,
                                        exist_ok=False)
        state_dict = torch.load(os.path.join(pretrained_checkpoints_dir, pretrained_ckpt_name),
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        state_dict = pl_ckpt_to_pytorch_state_dict(checkpoint_path=trainer.checkpoint_callback.best_model_path,
                                                   map_location=torch.device("cpu"),
                                                   delete_prefix_len=len("torch_nn_module."))
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        trainer.test(ckpt_path="best",
                     datamodule=dm)

if __name__ == "__main__":
    main()