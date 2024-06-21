import argparse
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models import MSEModule
from utils import seed_everything
seed_everything(42)

# dtype = torch.float
# # dtype = torch.double
# torch.set_default_dtype(dtype)
# default_dtype = torch.get_default_dtype()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments for training PyTorch models on SEVIR dataset with and without Evidential Deep Learning')
    parser.add_argument('-i', '--input_dir', type=str, default='', help='Directory where the training data is stored')
    parser.add_argument('-y', '--config_file', type=str, default='./config/mhd-0000.yaml', help='Path to the config file')
    parser.add_argument('-c', '--ckpt_path', type=str, default='', help='Directory to save checkpoints')
    parser.add_argument('-l', '--load_ckpt', type=str, default='', help='Checkpoint to load weights')
    parser.add_argument('-o','--log_dir', type=str, default='./logs', help='Directory to store logs')
    parser.add_argument('-d','--wandb_dir', type=str, default='', help='wandb_directory')
    parser.add_argument('-p','--wandb_project', type=str, default='', help='wandb_directory')
    parser.add_argument('-g','--wandb_group', type=str, default='', help='wandb_directory')
    parser.add_argument('-n','--no_wandb', action='store_true', help='turn off wandb')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=2)
    
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'vgg', 'mse+vgg', 'cgan'])
    
    args = parser.parse_known_args()

    return args

if __name__ == '__main__':

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse command line arguments
    args = parse_arguments()
    data_dir = args.input_dir
    ckpt_path = args.ckpt_path
    wandb_dir = args.wandb_dir
    wandb_project = args.wandb_project
    wandb_group = args.wandb_group
    use_wandb = not args.no_wandb

    # Load config file
    config = load_config(args.config_file)
    model_params = config['model_params']
    dataset_params = config['dataset_params']
    train_loader_params = config['train_loader_params']
    val_loader_params = config['val_loader_params']
    loss_params = config['loss_params']
    optimizer_params = config['optimizer_params']
    train_params = config['train_params']
    wandb_params = config['wandb_params']
    world_size = args.world_size


    # Set parameters to config values if not defined on command line
    if not data_dir:
        data_dir = dataset_params['data_dir']

    if not ckpt_path:
        ckpt_path = train_params['ckpt_path']

    if not wandb_dir:
        wandb_dir = wandb_params['wandb_dir']

    if not wandb_project:
        wandb_project = wandb_params['wandb_project']

    if not wandb_group:
        wandb_group = wandb_params['wandb_group']

    wandb_logger = WandbLogger(group=wandb_group, project=wandb_project, save_dir=wandb_dir, config=config)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, save_last=True, every_n_epochs=train_params['ckpt_freq'])

    mhd_lightning = MHDLightning(model_params, dataset_params, train_loader_params, val_loader_params, loss_params,
                                 optimizer_params, train_params, wandb_params, data_dir, ckpt_path, wandb_dir, wandb_project, wandb_group, world_size)
    
    mhd_lightning.run_training(wandb_logger)

