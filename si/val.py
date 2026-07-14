# Default imports
import argparse
from datetime import datetime
import torch
import os 

# Custom imports
from common.utils import get_yaml, save_yaml
from modules.train_module import TrainModule
from modules.ae_module import AutoencoderModule
from modules.combined_module import CombinedModule
from data.datamodule import ClimateDataModule

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
    if args.description is not None:
        trainconfig["description"] = args.description
    
    return config, modelconfig, trainconfig, dataconfig

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    description = trainconfig.get("description", "")
    name = modelconfig["model_name"] + "_" + description + "_" + str(seed) + "_" + now
    wandb_logger = WandbLogger(project=trainconfig["project"],
                               name=name,
                               mode=trainconfig["wandb_mode"])
    path = trainconfig["log_dir"] + name + "/"
    config['training']["log_dir"] = path

    config['training']['devices'] = 1
    config['training']['strategy'] = "auto"

    dataconfig['val_num_inferences'] = 50 

    os.makedirs(path, exist_ok=True) 
    save_yaml(config, path + "config.yml")

    autoencoder = dataconfig.get("autoencoder", False)
    is_combined = modelconfig.get("model_name", "") == "Combined"

    datamodule = ClimateDataModule(dataconfig=dataconfig)

    if is_combined:
        # CombinedModule loads forecaster + downscaler checkpoints internally.
        model = CombinedModule(config,
                               normalizer=datamodule.train_dataset)
    elif autoencoder:
        model = AutoencoderModule(config,
                                  normalizer=datamodule.train_dataset)
    else:
        model = TrainModule(config,
                            normalizer=datamodule.train_dataset)
    
    trainer = L.Trainer(devices = trainconfig["devices"],
                        num_nodes = trainconfig.get("num_nodes", 1),
                        accelerator = trainconfig["accelerator"],
                        strategy = trainconfig["strategy"],
                        check_val_every_n_epoch = trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps = trainconfig["log_every_n_steps"],
                        max_epochs = trainconfig["max_epochs"],
                        default_root_dir = path,
                        logger=wandb_logger,
                        accumulate_grad_batches=trainconfig.get("accumulate_grad_batches", 1),
                        num_sanity_val_steps=trainconfig.get("num_sanity_val_steps", 1),
                        precision=trainconfig["precision"],)
    
    if is_combined:
        # Weights are already loaded inside CombinedModule; don't pass ckpt_path.
        trainer.validate(model=model, datamodule=datamodule)
    elif trainconfig.get("checkpoint") is not None:
        trainer.validate(model=model,
                datamodule=datamodule,
                ckpt_path=trainconfig["checkpoint"],
                weights_only=False)
    else:
        trainer.validate(model=model,
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None)
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Path to the checkpoint to resume training')
    args = parser.parse_args()

    main(args)