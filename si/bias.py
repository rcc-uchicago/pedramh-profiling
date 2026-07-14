# Default imports
import argparse
import time
import torch
import os

# Custom imports
from common.utils import get_yaml, save_yaml, assemble_forcing, disassemble_input
from common.plotting import plot_reconstruction, plot_spectrum
from modules.train_module import TrainModule
from modules.combined_module import CombinedModule
from data.amip_new import GetDataset
from torch.utils.data import DataLoader, Subset

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything

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
        trainconfig["forecaster_checkpoint"] = args.checkpoint
    if args.description is not None:
        trainconfig["description"] = args.description
    
    return config, modelconfig, trainconfig, dataconfig

def plot_predictions(pred_feat_dict, target_feat_dict, log_dir, step):

    t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b h w -> h w 
    t2m_target = target_feat_dict['2m_temperature'][0].cpu()
    pr_6h_pred = pred_feat_dict['PRATEsfc_24h'][0].cpu()
    pr_6h_target = target_feat_dict['PRATEsfc_24h'][0].cpu()

    z500_pred = pred_feat_dict['geopotential'][0, -6, ...].cpu() # b l h w -> h w
    z500_target = target_feat_dict['geopotential'][0, -6, ...].cpu()
    u250_pred = pred_feat_dict['u_component_of_wind'][0, -9, ...].cpu()
    u250_target = target_feat_dict['u_component_of_wind'][0, -9, ...].cpu()
    t850_pred = pred_feat_dict['temperature'][0, -3, ...].cpu()
    t850_target = target_feat_dict['temperature'][0, -3, ...].cpu()
    q850_pred = pred_feat_dict['specific_total_water'][0, -3, ...].cpu()
    q850_target = target_feat_dict['specific_total_water'][0, -3, ...].cpu()

    plot_reconstruction(t2m_pred, # h w
                t2m_target,
                f'{log_dir}/t2m_{step}.png')
    plot_reconstruction(z500_pred,
                z500_target,
                f'{log_dir}/z500_{step}.png')
    plot_reconstruction(pr_6h_pred,
                pr_6h_target,
                f'{log_dir}/PRATEsfc_{step}.png')
    plot_reconstruction(u250_pred,
                u250_target,
                f'{log_dir}/u250_{step}.png')
    plot_reconstruction(t850_pred,
                t850_target,
                f'{log_dir}/t850_{step}.png')
    plot_reconstruction(q850_pred,
                q850_target,
                f'{log_dir}/q850_{step}.png')

    
    plot_spectrum(t2m_pred.unsqueeze(0),
                    t2m_target.unsqueeze(0),
                    f'{log_dir}/t2m_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(z500_pred.unsqueeze(0),
                    z500_target.unsqueeze(0),
                    f'{log_dir}/z500_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(pr_6h_pred.unsqueeze(0),
                    pr_6h_target.unsqueeze(0),
                    f'{log_dir}/PRATEsfc_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(u250_pred.unsqueeze(0),
                    u250_target.unsqueeze(0),
                    f'{log_dir}/u250_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(t850_pred.unsqueeze(0),
                    t850_target.unsqueeze(0),
                    f'{log_dir}/t850_spectrum_{step}.png',
                    num_t=1)
    plot_spectrum(q850_pred.unsqueeze(0),
                    q850_target.unsqueeze(0),
                    f'{log_dir}/q850_spectrum_{step}.png',
                    num_t=1)
    
def save_predictions(pred_feat_dict, target_feat_dict, log_dir, step):
    torch.save(pred_feat_dict, f'{log_dir}/predictions_{step}.pt')
    torch.save(target_feat_dict, f'{log_dir}/targets_{step}.pt')

def main(args):
    config=get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    ID = 0
    seed = trainconfig["seed"] + ID
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    is_combined = modelconfig.get("model_name", "") == "Combined"

    description = trainconfig.get("description", "")
    # Combined module bundles two checkpoints; anchor the log dir on the forecaster's.
    if is_combined:
        anchor_ckpt = trainconfig.get("checkpoint") or trainconfig["forecaster_checkpoint"]
    else:
        anchor_ckpt = trainconfig["checkpoint"]

    directory_path = os.path.dirname(anchor_ckpt)
    path = os.path.join(directory_path, f"bias_logs_{description}_{ID}/")

    os.makedirs(path, exist_ok=True) 
    print(f"Logging to: {path}")
    save_yaml(config, path + "config.yml")

    dataconfig['batch_size'] = 1

    dataset = GetDataset(dataconfig,
                         year_start=1996,
                         year_end=2001)

    return_calendar = dataconfig.get('return_calendar', False)

    # Step through dataset at forecast intervals (e.g. every 4th sample for 24h steps with 6h data)
    stride = dataconfig['timedelta_hours'] // dataconfig['data_timedelta_hours']
    
    device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"

    if is_combined:
        # CombinedModule loads forecaster + downscaler checkpoints internally.
        model = CombinedModule(config, normalizer=dataset).to(device)
    else:
        model = TrainModule(config, normalizer=dataset).to(device)
        state_dict = torch.load(anchor_ckpt, map_location=device, weights_only=False)['state_dict']
        model.load_state_dict(state_dict)
    model.eval()

    ensemble_size = 1
    invariant = model.invariant_input.to(device) # 1 c nlat nlon
    invariant = invariant.expand(ensemble_size, -1, -1, -1) # e c nlat nlon

    # Climatology resolution matches the prediction resolution:
    # - CombinedModule outputs at full (downscaler) resolution.
    # - Forecaster-only: low-res if a downsample is configured, else full-res.
    if (not is_combined) and model.downsample is not None:
        downsample_factor = model.downsample.downsample_factor
        clim_nlat, clim_nlon = model.nlat // downsample_factor, model.nlon // downsample_factor
    else:
        clim_nlat, clim_nlon = model.nlat, model.nlon

    num_steps = len(dataset) // stride
    print(f"Processing {num_steps} timesteps (stride={stride}) with ensemble size {ensemble_size}...")

    plot_every = 30
    plot_val = trainconfig.get("plot_val", False)
    #num_steps = 500

    # Strided subset preserves date ordering for the autoregressive rollout while
    # letting a multi-worker DataLoader prefetch HDF5 reads in parallel. The model
    # forward is still sequential, but I/O + host->device copies overlap with compute.
    start = 0
    strided_indices = list(range(start, num_steps * stride, stride))
    strided_dataset = Subset(dataset, strided_indices)
    num_workers = int(dataconfig.get("num_data_workers", 4))
    loader = DataLoader(
        strided_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # per-member running mean accumulators: e c h w / e c l h w
    climatology_surface = torch.zeros((ensemble_size, len(model.surface_variables), clim_nlat, clim_nlon), device=device)
    climatology_multilevel = torch.zeros((ensemble_size, len(model.multilevel_variables), model.nlevels, clim_nlat, clim_nlon), device=device)
    climatology_diagnostic = torch.zeros((ensemble_size, len(model.diagnostic_variables), clim_nlat, clim_nlon), device=device)

    log_every = 10
    start_time = time.time()

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            # DataLoader has already added the leading batch dim (size 1).
            if return_calendar:
                surface_t_b, upper_air_t_b, diagnostic_t_b, surface_t1_b, upper_air_t1_b, diagnostic_t1_b, varying_boundary_data_b, calendar_b = batch
                calendar = calendar_b.to(device, non_blocking=True).expand(ensemble_size, -1)
            else:
                surface_t_b, upper_air_t_b, diagnostic_t_b, surface_t1_b, upper_air_t1_b, diagnostic_t1_b, varying_boundary_data_b = batch
                calendar = None

            varying_boundary_data = varying_boundary_data_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1)

            if step_idx == 0:
                surface_t = surface_t_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1)
                upper_air_t = upper_air_t_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1, -1)
                diagnostic_t = diagnostic_t_b.to(device, non_blocking=True).expand(ensemble_size, -1, -1, -1)

                x = model.preprocess(surface_t, upper_air_t, diagnostic_t) # e c h w / e c l h w / e c h w

            c_grid = assemble_forcing(varying_boundary_data, invariant) # e c h w

            # TrainModule: y and y_last both low-res.
            # CombinedModule: y is low-res rollout state, y_last is full-res downscaled prediction.
            fwd_kwargs = {'return_model_last': True}
            if calendar is not None:
                fwd_kwargs['c_scalar'] = calendar
            y, y_last = model.forward(x, c_grid, **fwd_kwargs)

            surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(y_last, nlevels=model.nlevels)

            surface_pred_denorm = model.n.surface_inv_transform(surface_pred)
            multilevel_pred_denorm = model.n.upper_air_inv_transform(multilevel_pred)
            diagnostic_pred_denorm = model.n.diagnostic_inv_transform(diagnostic_pred)

            # per-member running mean update
            n = step_idx + 1
            climatology_surface += (surface_pred_denorm - climatology_surface) / n
            climatology_multilevel += (multilevel_pred_denorm - climatology_multilevel) / n
            climatology_diagnostic += (diagnostic_pred_denorm - climatology_diagnostic) / n

            x = y

            if (step_idx + 1) % log_every == 0 or step_idx == num_steps - 1:
                elapsed = time.time() - start_time
                steps_done = step_idx + 1
                avg_per_step = elapsed / steps_done
                remaining = avg_per_step * (num_steps - steps_done)
                print(
                    f"Step {steps_done}/{num_steps} | "
                    f"elapsed {elapsed:.1f}s | "
                    f"remaining {remaining:.1f}s | "
                    f"{avg_per_step:.2f}s/step",
                    flush=True,
                )

            if step_idx % plot_every == 0:
                print(f"Step {step_idx}/{num_steps}")
                # save intermediate climatology (ensemble mean)
                torch.save(climatology_surface.mean(dim=0).cpu(), path + f"climatology_surface_{step_idx + 1}.pt")
                torch.save(climatology_multilevel.mean(dim=0).cpu(), path + f"climatology_multilevel_{step_idx + 1}.pt")
                torch.save(climatology_diagnostic.mean(dim=0).cpu(), path + f"climatology_diagnostic_{step_idx + 1}.pt")

                # Targets stay at full resolution
                surface_t1_dev = surface_t1_b.to(device, non_blocking=True)
                upper_air_t1_dev = upper_air_t1_b.to(device, non_blocking=True)
                diagnostic_t1_dev = diagnostic_t1_b.to(device, non_blocking=True)

                # Match target resolution to prediction resolution.
                if (not is_combined) and model.downsample is not None:
                    surface_t1_dev, upper_air_t1_dev, diagnostic_t1_dev = model.downsample(surface_t1_dev, upper_air_t1_dev, diagnostic_t1_dev)

                surface_true_denorm = model.n.surface_inv_transform(surface_t1_dev)
                multilevel_true_denorm = model.n.upper_air_inv_transform(upper_air_t1_dev)
                diagnostic_true_denorm = model.n.diagnostic_inv_transform(diagnostic_t1_dev)

                # use first ensemble member for plotting
                pred_feat_dict = {}
                target_feat_dict = {}

                for c, surface_feat_name in enumerate(model.surface_variables):
                    pred_feat_dict[surface_feat_name] = surface_pred_denorm[:1, c] # 1 nlat nlon
                    target_feat_dict[surface_feat_name] = surface_true_denorm[:, c]

                for c, multilevel_feat_name in enumerate(model.multilevel_variables):
                    pred_feat_dict[multilevel_feat_name] = multilevel_pred_denorm[:1, c] # 1 nlevel nlat nlon
                    target_feat_dict[multilevel_feat_name] = multilevel_true_denorm[:, c]

                for c, diagnostic_feat_name in enumerate(model.diagnostic_variables):
                    pred_feat_dict[diagnostic_feat_name] = diagnostic_pred_denorm[:1, c] # 1 nlat nlon
                    target_feat_dict[diagnostic_feat_name] = diagnostic_true_denorm[:, c]

                if plot_val:
                    plot_predictions(pred_feat_dict, target_feat_dict, path, step_idx + 1)
                
                save_predictions(pred_feat_dict, target_feat_dict, path, step_idx + 1)

    # save ensemble climatologies
    torch.save(climatology_surface.cpu(), path + "climatology_surface_ensemble.pt")
    torch.save(climatology_multilevel.cpu(), path + "climatology_multilevel_ensemble.pt")
    torch.save(climatology_diagnostic.cpu(), path + "climatology_diagnostic_ensemble.pt")

    # average across ensemble members
    torch.save(climatology_surface.mean(dim=0).cpu(), path + "climatology_surface.pt")
    torch.save(climatology_multilevel.mean(dim=0).cpu(), path + "climatology_multilevel.pt")
    torch.save(climatology_diagnostic.mean(dim=0).cpu(), path + "climatology_diagnostic.pt")


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
