# import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
from torchmetrics import Metric
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
# import argparse
# from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
# from earthformer.config import cfg
from utils.optim import SequentialLR, warmup_lambda
from utils.utils import get_parameter_names
# from utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from sevir.layout import layout_to_in_out_slice
from utils.visualization.sevir_vis_seq import save_example_vis_results
from utils.metrics import SEVIRSkillScore

from .unet import UNet
from .gan import Generator, Discriminator
from .cuboid_transformer import CuboidTransformerModel
from sevir.torch_wrap import SEVIRLightningDataModule

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
# pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
# pytorch_state_dict_name = "earthformer_sevir.pt"

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def KLDiv(out_seq, gamma, v, alpha, beta, omega=0.01, kl=False):
    
    error = (out_seq - gamma).abs()
    
    if kl:
        loss = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
    else:
        loss = 2*v+alpha
    
    return (loss*error).mean()

def KLL_NIG(y, output):
    gamma, v, alpha, beta = torch.split(output, 1, -1)
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


def modified_mse(gamma, nu, alpha, beta, target, reduction='mean'):
    """
    Lipschitz MSE loss of the "Improving evidential deep learning via multi-task learning."

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        target ([FloatTensor]): true labels.
        reduction (str, optional): . Defaults to 'mean'.

    Returns:
        [FloatTensor]: The loss value. 
    """
    mse = (gamma-target)**2
    c = get_mse_coef(gamma, nu, alpha, beta, target).detach()
    modified_mse = mse*c
    if reduction == 'mean': 
        return modified_mse.mean()
    elif reduction == 'sum':
        return modified_mse.sum()
    else:
        return modified_mse


def get_mse_coef(gamma, nu, alpha, beta, y):
    """
    Return the coefficient of the MSE loss for each prediction.
    By assigning the coefficient to each MSE value, it clips the gradient of the MSE
    based on the threshold values U_nu, U_alpha, which are calculated by check_mse_efficiency_* functions.

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        y ([FloatTensor]): true labels.

    Returns:
        [FloatTensor]: [0.0-1.0], the coefficient of the MSE for each prediction.
    """
    alpha_eff = check_mse_efficiency_alpha(gamma, nu, alpha, beta, y)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta, y)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt()/delta).detach()
    return torch.clip(c, min=False, max=1.)


def check_mse_efficiency_alpha(gamma, nu, alpha, beta, y, reduction='mean'):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial alpha(numpy.array) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    
    """
    delta = (y-gamma)**2
    right = (torch.exp((torch.digamma(alpha+0.5)-torch.digamma(alpha))) - 1)*2*beta*(1+nu) / nu

    return (right).detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta, y):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial nu(torch.Tensor) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu+1)/nu
    return (beta*nu_1/alpha)




class AverageLoss(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: Tensor) -> None:
        
        self.loss_sum += loss
        self.total += 1

    def compute(self) -> Tensor:
        return self.loss_sum / self.total
    
class SEVIRPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(SEVIRPLModule, self).__init__()

        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])
            
        #loss fn
        self.loss_cfg = OmegaConf.to_object(oc.loss)
        self.edl = self.loss_cfg["edl"]
        if self.loss_cfg["loss_fn"] == 'mse' and not self.edl:
            self.loss_fn = F.mse_loss
        elif self.loss_cfg["loss_fn"] == 'mse' and self.edl:
            self.loss_fn = self.MSE_EDL
        elif self.loss_cfg["loss_fn"] == 'mse_to_nll':
            self.loss_fn = self.MSE_to_NLL
        elif self.loss_cfg["loss_fn"] == 'mse_plus_nll':
            self.loss_fn = self.MSE_plus_NLL
        elif self.loss_cfg["loss_fn"] == 'nll':
            self.loss_fn = KLL_NIG
        elif self.loss_cfg["loss_fn"] == 'lipschitz':
            self.loss_fn = self.lipschitz
        elif self.loss_cfg["loss_fn"] == 'custom_nll':
            self.loss_fn = self.custom_nll
            # Initialize all running statistics at 0.
            self.num_losses = 5
            self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_std_l = None        
        elif self.loss_cfg["loss_fn"] == 'custom_nll_v2':
            self.loss_fn = self.custom_nll
            # Initialize all running statistics at 0.
            self.num_losses = 6
            self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_std_l = None
        elif self.loss_cfg["loss_fn"] == 'custom_nll_v3':
            self.loss_fn = self.custom_nll
            # Initialize all running statistics at 0.
            self.num_losses = 2
            self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor)
            self.running_std_l = None
        
        self.edl_act = self.loss_cfg["edl_act"]
        self.kldiv = KLDiv
        self.lambda_increasing = self.loss_cfg["lambda_increasing"]
        self.coeff = lambda x: self.loss_cfg["slope"] * x if self.loss_cfg["slope"] * x < self.loss_cfg["lambda"] else self.loss_cfg["lambda"]
        if self.loss_cfg["late_start"] != 0:
            self.coeff = lambda x: 0 if x < self.loss_cfg["late_start"] else (self.loss_cfg["slope"] * (x - self.loss_cfg["late_start"]) if self.loss_cfg["slope"] * (x - self.loss_cfg["late_start"]) < self.loss_cfg["lambda"] else self.loss_cfg["lambda"])
        
        if model_cfg["name"] == "earthformer":
            self.torch_nn_module = CuboidTransformerModel(
                input_shape=model_cfg["input_shape"],
                target_shape=model_cfg["target_shape"],
                base_units=model_cfg["base_units"],
                block_units=model_cfg["block_units"],
                scale_alpha=model_cfg["scale_alpha"],
                enc_depth=model_cfg["enc_depth"],
                dec_depth=model_cfg["dec_depth"],
                enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
                dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
                dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
                downsample=model_cfg["downsample"],
                downsample_type=model_cfg["downsample_type"],
                enc_attn_patterns=enc_attn_patterns,
                dec_self_attn_patterns=dec_self_attn_patterns,
                dec_cross_attn_patterns=dec_cross_attn_patterns,
                dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
                dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
                num_heads=model_cfg["num_heads"],
                attn_drop=model_cfg["attn_drop"],
                proj_drop=model_cfg["proj_drop"],
                ffn_drop=model_cfg["ffn_drop"],
                upsample_type=model_cfg["upsample_type"],
                ffn_activation=model_cfg["ffn_activation"],
                gated_ffn=model_cfg["gated_ffn"],
                norm_layer=model_cfg["norm_layer"],
                # global vectors
                num_global_vectors=model_cfg["num_global_vectors"],
                use_dec_self_global=model_cfg["use_dec_self_global"],
                dec_self_update_global=model_cfg["dec_self_update_global"],
                use_dec_cross_global=model_cfg["use_dec_cross_global"],
                use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
                use_global_self_attn=model_cfg["use_global_self_attn"],
                separate_global_qkv=model_cfg["separate_global_qkv"],
                global_dim_ratio=model_cfg["global_dim_ratio"],
                # initial_downsample
                initial_downsample_type=model_cfg["initial_downsample_type"],
                initial_downsample_activation=model_cfg["initial_downsample_activation"],
                # initial_downsample_type=="stack_conv"
                initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
                initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
                initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
                initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
                # misc
                padding_type=model_cfg["padding_type"],
                z_init_method=model_cfg["z_init_method"],
                checkpoint_level=model_cfg["checkpoint_level"],
                pos_embed_type=model_cfg["pos_embed_type"],
                use_relative_pos=model_cfg["use_relative_pos"],
                self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
                # initialization
                attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
                ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
                conv_init_mode=model_cfg["conv_init_mode"],
                down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
                norm_init_mode=model_cfg["norm_init_mode"],
                edl = self.edl,
                edl_act = self.edl_act,
            )
        elif model_cfg["name"] == "unet":
            self.torch_nn_module = UNet(
                input_shape=model_cfg["input_shape"],
                target_shape=model_cfg["target_shape"],
                enc_nodes=model_cfg["enc_nodes"],
                center=model_cfg["center"],
                dec_nodes=model_cfg["dec_nodes"],
                activation=model_cfg["activation"],
                edl = self.edl,
                edl_act = self.edl_act,
            )
        else:
            raise NotImplementedError
        
        self.model_name = model_cfg["name"]
        self.total_num_steps = total_num_steps
        self.save_hyperparameters(oc)
        self.oc = oc

        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)
        
        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_loss = AverageLoss()
        self.valid_kldiv = AverageLoss()
        self.valid_total_loss = AverageLoss()
        self.valid_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_loss = AverageLoss()
        self.test_kldiv = AverageLoss()
        self.test_total_loss = AverageLoss()
        self.test_score = SEVIRSkillScore(
            mode=self.metrics_mode,
            seq_len=self.out_len,
            layout=self.layout,
            threshold_list=self.threshold_list,
            metrics_list=self.metrics_list,
            eps=1e-4,)
        
        
    def MSE_EDL(self, y, output):
        gamma, v, alpha, beta = torch.split(output, 1, -1)

        return F.mse_loss(y, gamma)
    
    def MSE_to_NLL(self, y, output):
        gamma, v, alpha, beta = torch.split(output, 1, -1)
        
        if self.current_epoch < self.loss_cfg["late_start"]:
            return F.mse_loss(y, gamma)
        else:
            return KLL_NIG(y, output)
        
    def MSE_plus_NLL(self, y, output):
        gamma, v, alpha, beta = torch.split(output, 1, -1)
        
        return F.mse_loss(y, gamma) + KLL_NIG(y, output) * .001
    
    def lipschitz(self, y, output):
        gamma, v, alpha, beta = torch.split(output, 1, -1)
        
        return modified_mse(gamma, v, alpha, beta, y) + KLL_NIG(y, output)
    
    def custom_nll(self, y, output):
        gamma, v, alpha, beta = torch.split(output, 1, -1)
        two_beta_lambda = 2 * beta * (1 + v)
        t1 = 0.5 * (torch.pi / v).log()
        t2 = alpha * two_beta_lambda.log()
        t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
        t4 = alpha.lgamma()
        t5 = (alpha + 0.5).lgamma()
        if self.loss_cfg["loss_fn"] == "custom_nll":
            unweighted_losses = [t1.mean(), -t2.mean(), t3.mean(), t4.mean(), -t5.mean()]
        elif self.loss_cfg["loss_fn"] == "custom_nll_v2":
            unweighted_losses = [t1.mean(), -t2.mean(), t3.mean(), t4.mean(), -t5.mean(), F.mse_loss(y, gamma)]
        elif self.loss_cfg["loss_fn"] == "custom_nll_v3":
            nll = t1 - t2 + t3 + t4 - t5
            unweighted_losses = [nll.mean(), F.mse_loss(y, gamma)]
        
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)
        if self.training:
            return torch.sum(L)
        
        self.running_mean_L = self.running_mean_L.to(self.device)
        self.running_mean_l = self.running_mean_l.to(self.device)
        self.running_S_l = self.running_S_l.to(self.device)
        if self.running_std_l is not None:
            self.running_std_l = self.running_std_l.to(self.device)
        
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.global_step == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.global_step <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
                self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.global_step == 0:
            mean_param = 0.0
        else:
            mean_param = (1. - 1 / (self.global_step + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have global_step = t - 1
        running_variance_l = self.running_S_l / (self.global_step + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        loss = sum(weighted_losses)
        return loss 

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.dataset = self.get_dataset_config()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_dataset_config():
        oc = OmegaConf.create()
        oc.dataset_name = "sevir"
        oc.img_height = 384
        oc.img_width = 384
        oc.in_len = 13
        oc.out_len = 12
        oc.seq_len = 25
        oc.plot_stride = 2
        oc.interval_real_time = 5
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.start_date = None
        oc.train_val_split_date = (2019, 1, 1)
        oc.train_test_split_date = (2019, 6, 1)
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ('csi', 'pod', 'sucr', 'bias')
        oc.threshold_list = (16, 74, 133, 160, 181, 219)
        return oc
    
    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.name = "earthformer"
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'
        cfg.checkpoint_level = 2
        # initial downsample and final upsample
        cfg.initial_downsample_type = "stack_conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_stack_conv_num_layers = 3
        cfg.initial_downsample_stack_conv_dim_list = [4, 16, cfg.base_units]
        cfg.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
        cfg.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 100
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 20
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "SEVIR"
        oc.monitor_lr = True
        oc.monitor_device = False
        # oc.track_grad_norm = -1
        oc.use_wandb = True
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [80, ]
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        elif self.oc.optim.method == 'adam':
            optimizer = torch.optim.Adam(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project="SEVIR_EDL", group=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            # track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # Distributed data parallel
            strategy="ddp_find_unused_parameters_true",
            # strategy="auto",
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_sevir_datamodule(dataset_oc,
                             micro_batch_size: int = 1,
                             num_workers: int = 8):
        dm = SEVIRLightningDataModule(
            seq_len=dataset_oc["seq_len"],
            sample_mode=dataset_oc["sample_mode"],
            stride=dataset_oc["stride"],
            batch_size=micro_batch_size,
            layout=dataset_oc["layout"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            # datamodule_only
            dataset_name=dataset_oc["dataset_name"],
            start_date=dataset_oc["start_date"],
            train_val_split_date=dataset_oc["train_val_split_date"],
            train_test_split_date=dataset_oc["train_test_split_date"],
            end_date=dataset_oc["end_date"],
            num_workers=num_workers,)
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(layout=self.layout,
                                                         in_len=self.in_len,
                                                         out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice
    
    def loss(self, output, out_seq):
        edl_params = []
        if self.edl:
            gamma, v, alpha, beta = torch.split(output, 1, -1)
            edl_params = [v, alpha, beta]
        else:
            gamma = output
        loss = self.loss_fn(out_seq, output)
        if self.edl:
            kldiv = self.kldiv(out_seq, gamma, v, alpha, beta, omega=torch.tensor(self.loss_cfg["omega"], requires_grad=False), kl=self.loss_cfg["kl"])
        else:
            kldiv = 0
        return gamma, loss, kldiv, edl_params
    
    def forward(self, in_seq, out_seq):
        output = self.torch_nn_module(in_seq)
        output, loss, kldiv, edl_params = self.loss(output, out_seq)
        return output, loss, kldiv, edl_params

    def training_step(self, batch, batch_idx):
        data_seq = batch['vil'].contiguous()
        x = data_seq[self.in_slice]
        y = data_seq[self.out_slice]
        y_hat, loss, kldiv, edl_params = self(x, y)
        micro_batch_size = x.shape[self.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)
        self.save_vis_step_end(
            data_idx=data_idx,
            in_seq=x,
            target_seq=y,
            pred_seq=y_hat,
            edl_params=edl_params,
            mode="train"
        )
        self.log('train_loss', loss,
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log('kldiv_loss', kldiv,
                 on_step=True, on_epoch=False, sync_dist=True)
        total_loss = loss + self.coeff(self.current_epoch) * kldiv
        self.log('total_train_loss', total_loss, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data_seq = batch['vil'].contiguous()
        x = data_seq[self.in_slice]
        y = data_seq[self.out_slice]
        micro_batch_size = x.shape[self.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            y_hat, loss, kldiv, edl_params = self(x, y)
            self.save_vis_step_end(
                data_idx=data_idx,
                in_seq=x,
                target_seq=y,
                pred_seq=y_hat,
                edl_params=edl_params,
                mode="val"
            )
            if self.oc.trainer.precision == 16:
                y_hat = y_hat.float()
            if not y.is_contiguous():
                y = y.contiguous()
            if not y_hat.is_contiguous():
                y_hat = y_hat.contiguous()
            step_mse = self.valid_mse(y_hat, y)
            step_mae = self.valid_mae(y_hat, y)
            self.valid_loss.update(loss)
            self.valid_kldiv.update(kldiv)
            total_loss = loss + self.coeff(self.current_epoch) * kldiv
            self.valid_total_loss.update(total_loss)
                    
            self.valid_score.update(y_hat, y)
            self.log('valid_frame_mse_step', step_mse,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('valid_frame_mae_step', step_mae,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            
            self.log('valid_loss_step', loss,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('valid_kldiv_step', kldiv,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('valid_total_loss_step', total_loss,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return None

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        
        valid_loss = self.valid_loss.compute()
        valid_kldiv = self.valid_kldiv.compute()
        valid_total_loss = self.valid_total_loss.compute()
        
        self.log('valid_frame_mse_epoch', valid_mse,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_frame_mae_epoch', valid_mae,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log('valid_loss_mse-nll_epoch', valid_loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_kldiv_epoch', valid_kldiv,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_total_loss_epoch', valid_total_loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.edl:
            self.log('KL_coeff', self.coeff(self.current_epoch), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        self.valid_loss.reset()
        self.valid_kldiv.reset()
        self.valid_total_loss.reset()
        valid_score = self.valid_score.compute()
        self.log("valid_loss_epoch", -valid_score["avg"]["csi"],
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_score_epoch_end(score_dict=valid_score, mode="val")
        self.valid_score.reset()
        self.save_score_epoch_end(score_dict=valid_score,
                                  mse=valid_mse,
                                  mae=valid_mae,
                                  mode="val")

    def test_step(self, batch, batch_idx):
        data_seq = batch['vil'].contiguous()
        x = data_seq[self.in_slice]
        y = data_seq[self.out_slice]
        micro_batch_size = x.shape[self.layout.find("N")]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            y_hat, loss, kldiv, edl_params = self(x, y)
            self.save_vis_step_end(
                data_idx=data_idx,
                in_seq=x,
                target_seq=y,
                pred_seq=y_hat,
                edl_params=edl_params,
                mode="test"
            )
            if self.oc.trainer.precision == 16:
                y_hat = y_hat.float()
            if not y.is_contiguous():
                y = y.contiguous()
            if not y_hat.is_contiguous():
                y_hat = y_hat.contiguous()
            step_mse = self.test_mse(y_hat, y)
            step_mae = self.test_mae(y_hat, y)
            self.test_loss.update(loss)
            self.test_kldiv.update(kldiv)
            total_loss = loss + self.coeff(self.current_epoch) * kldiv
            self.test_total_loss.update(total_loss)
            self.test_score.update(y_hat, y)
            
            self.log('test_frame_mse_step', step_mse,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('test_frame_mae_step', step_mae,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            
            self.log('test_loss_step', loss,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('test_kldiv_step', kldiv,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log('test_total_loss_step', total_loss,
                     prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return None

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        
        test_loss = self.test_loss.compute()
        test_kldiv = self.test_kldiv.compute()
        test_total_loss = self.test_total_loss.compute()
        
        self.log('test_frame_mse_epoch', test_mse,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_frame_mae_epoch', test_mae,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log('test_loss_mse-nll_epoch', test_loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_kldiv_epoch', test_kldiv,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_total_loss_epoch', test_total_loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.edl:
            self.log('KL_coeff', self.coeff(self.current_epoch), prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_loss.reset()
        self.test_kldiv.reset()
        self.test_total_loss.reset()
        test_score = self.test_score.compute()
        self.log_score_epoch_end(score_dict=test_score, mode="test")
        self.test_score.reset()
        self.save_score_epoch_end(score_dict=test_score,
                                  mse=test_mse,
                                  mae=test_mae,
                                  mode="test")

    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean, sync_dist=True)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean, sync_dist=True)

    def save_score_epoch_end(self,
                             score_dict: Dict,
                             mse: Union[np.ndarray, float],
                             mae: Union[np.ndarray, float],
                             mode: str = "val"):
        assert mode in ["val", "test"], f"Wrong mode {mode}. Must be 'val' or 'test'."
        if self.local_rank == 0:
            save_dict = deepcopy(score_dict)
            save_dict.update(dict(mse=mse, mae=mae))
            if self.scores_dir is not None:
                save_path = os.path.join(self.scores_dir, f"{mode}_results_epoch_{self.current_epoch}.pkl")
                f = open(save_path, 'wb')
                pickle.dump(save_dict, f)
                f.close()

    def save_vis_step_end(
            self,
            data_idx: int,
            in_seq: torch.Tensor,
            target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            edl_params: torch.Tensor,
            mode: str = "train",
            vis_hits_misses_fas: bool = True):
        r"""
        Parameters
        ----------
        data_idx:   int
            data_idx == batch_idx * micro_batch_size
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if self.edl:
                for i in range(3):
                    edl_params[i] = edl_params[i].detach().float().cpu().numpy()
            if data_idx in example_data_idx_list:
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    layout=self.layout,
                    plot_stride=self.oc.vis.plot_stride,
                    label=self.oc.logging.logging_prefix,
                    interval_real_time=self.oc.dataset.interval_real_time,
                    edl=self.edl,
                    edl_params=edl_params,
                    vis_hits_misses_fas=vis_hits_misses_fas)