import os

import lightning as L
import torch
import torch.cuda.nvtx as nvtx

from common.loss import latitude_weighted_rmse
from common.plotting import plot_result, plot_spectrum
from common.utils import assemble_forcing, disassemble_input, assemble_input

_NVTX = os.environ.get("SI_NVTX") == "1"


class _ModelWithScalar:
    """Thin callable that binds a per-step c_scalar onto the DiT forward.

    Schedulers call ``model(x_noised, cond, t, c_grid)`` positionally; wrapping
    lets us route calendar info through without touching scheduler signatures.
    Update ``.c_scalar`` between rollout steps.
    """
    def __init__(self, model):
        self.model = model
        self.c_scalar = None

    def __call__(self, x_noised, cond, t, c_grid=None):
        return self.model(x_noised, cond, t, c_grid=c_grid, c_scalar=self.c_scalar)


class TrainModule(L.LightningModule):
    def __init__(self,
                 config: dict,
                 normalizer= None):
        '''
        TrainModule
        args:
            config (dict): configuration dictionary containing model, training and data configurations
            normalizer (object, optional): normalizer object for scaling input data. Defaults to None.
        '''

        super().__init__()
        self.config=config
        self.dataconfig = config['data']
        self.batch_size = self.dataconfig['batch_size']
        self.surface_variables = self.dataconfig['surface_variables']
        self.multilevel_variables = self.dataconfig['upper_air_variables']
        self.diagnostic_variables = self.dataconfig['diagnostic_variables']
        self.horizontal_resolution = self.dataconfig['horizontal_resolution']
        self.nlat, self.nlon = self.horizontal_resolution
        self.nlevels = len(self.dataconfig['levels'])
        self.plot_val = config['training'].get('plot_val', False)
        self.multistep_rollout = int(self.dataconfig.get('multistep_rollout', 1))
        self.multistep_num_sample_steps = config['training'].get('multistep_num_sample_steps', None)
        self.return_calendar = self.dataconfig.get('return_calendar', False)

        self.modelconfig = config['model']
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.log_dir = config['training']['log_dir']
        self.optimizer_name = config['training']['optimizer']

        self.n = normalizer
        self.constant_boundary_data = self.n.constant_boundary_data # surface geopotential, lsm, in shape c nlat nlon
        # invariant_input will be expanded to match batch size dynamically
        self.invariant_input = self.constant_boundary_data.unsqueeze(0) # 1 c nlat nlon

        self.downsample = None 

        if self.model_name == "SI_DiT":
            from modules.models.DiT import DiT
            from modules.diffusion.dynamic_interpolant import DriftScheduler
            self.model = DiT(**self.modelconfig['SI_DiT']["model"])
            self.scheduler = DriftScheduler(**self.modelconfig["SI_DiT"]['scheduler'])
        elif self.model_name == "SI_X":
            from modules.models.DiT import DiT
            from modules.diffusion.x_interpolant import DynamicInterpolant
            from modules.layers.bilinear import BilinearEncoder

            self.model = DiT(**self.modelconfig['SI_X']["model"])
            self.scheduler = DynamicInterpolant(**self.modelconfig['SI_X']['scheduler'])
            self.downsample = BilinearEncoder()
        elif self.model_name == "FM":
            from modules.models.DiT import DiT
            from modules.diffusion.flow_matching import FlowMatching
            from modules.layers.bilinear import BilinearEncoder

            self.model = DiT(**self.modelconfig['FM']["model"])
            self.scheduler = FlowMatching(**self.modelconfig['FM']['scheduler'])
            self.downsample = BilinearEncoder()
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        if not self.ddp or self.global_rank == 0:
            print(self.model)

        self.save_hyperparameters()

    def preprocess(self, surface_t, upper_air_t, diagnostic_t):
        if self.downsample is not None:
            with torch.no_grad():
                surface_t, upper_air_t, diagnostic_t = self.downsample(surface_t, upper_air_t, diagnostic_t)

        return assemble_input(surface_t, upper_air_t, diagnostic_t) # b c h w

    def forward(self, x, c_grid, return_model_last=False, c_scalar=None):
        if c_scalar is not None:
            model = _ModelWithScalar(self.model)
            model.c_scalar = c_scalar
        else:
            model = self.model

        if return_model_last: # special case; y is the euler step, and y_last is the output of x_pred model
            y, y_last = self.scheduler.sample(model, x, c_grid, return_model_last=return_model_last)
            return y, y_last
        else: # normal sampling
            y = self.scheduler.sample(model, x, c_grid)
            return y
    
    def training_step(self, batch, batch_idx):

        if self.return_calendar:
            surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data, calendar = batch
        else:
            surface_t, upper_air_t, diagnostic_t, surface_t1, upper_air_t1, diagnostic_t1, varying_boundary_data = batch
            calendar = None
        device = surface_t.device

        if _NVTX: nvtx.range_push("preprocess")
        x = self.preprocess(surface_t, upper_air_t, diagnostic_t)
        y = self.preprocess(surface_t1, upper_air_t1, diagnostic_t1)
        if _NVTX: nvtx.range_pop()

        invariant = self.invariant_input.expand(surface_t.shape[0], -1, -1, -1).to(device) # b c nlat nlon

        if _NVTX: nvtx.range_push("forward_loss")
        if self.multistep_rollout > 1:
            # varying_boundary_data: b rollout c h w — assemble forcings per step
            rollout = varying_boundary_data.shape[1]
            c_grids = torch.stack(
                [assemble_forcing(varying_boundary_data[:, step], invariant) for step in range(rollout)],
                dim=1,
            )  # b rollout c h w
            if self.return_calendar:
                # Inline the multistep rollout so we can rebind c_scalar per step.
                model = _ModelWithScalar(self.model)
                x_current = x
                with torch.no_grad():
                    for step in range(rollout - 1):
                        model.c_scalar = calendar[:, step]
                        x_current = self.scheduler.sample(
                            model, x_current, c_grids[:, step], num_steps=self.multistep_num_sample_steps,
                        )
                model.c_scalar = calendar[:, -1]
                loss, spectral_loss = self.scheduler.compute_loss(model, x_current, c_grids[:, -1], y)
            else:
                loss, spectral_loss = self.scheduler.compute_multistep_loss(
                    self.model, x, c_grids, y, num_sample_steps=self.multistep_num_sample_steps,
                )
        else:
            c_grid = assemble_forcing(varying_boundary_data, invariant) # b c h w
            if self.return_calendar:
                model = _ModelWithScalar(self.model)
                model.c_scalar = calendar
                loss, spectral_loss = self.scheduler.compute_loss(model, x, c_grid, y)
            else:
                loss, spectral_loss = self.scheduler.compute_loss(self.model, x, c_grid, y)
        if _NVTX: nvtx.range_pop()

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        self.log("train/spectral_loss", spectral_loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss

    def validation_step(self, batch, batch_idx, evaluate=False): 
        # each batch contains val_nsteps number of snapshots
        loss_dict, pred_feat_dict, target_feat_dict = self.predict(batch)

        if evaluate:
            return pred_feat_dict, target_feat_dict

        self.log_losses(loss_dict)

        if batch_idx == 0:
            if not self.ddp or self.global_rank == 0:
                if self.plot_val:
                    self.plot_predictions(pred_feat_dict, target_feat_dict)
                self.save_predictions(pred_feat_dict, target_feat_dict)
    
    def save_predictions(self, pred_feat_dict, target_feat_dict):
        torch.save(pred_feat_dict, f'{self.log_dir}predictions_epoch_{self.current_epoch}.pt')
        torch.save(target_feat_dict, f'{self.log_dir}targets_epoch_{self.current_epoch}.pt')

    @torch.no_grad()
    def predict(self, batch):

        if self.return_calendar:
            surface_t, upper_air_t, diagnostic_t, \
            targets_surface, targets_upper_air, targets_diagnostic, \
            varying_boundary_data, start_time_tensor, calendar = batch
        else:
            surface_t, upper_air_t, diagnostic_t, \
            targets_surface, targets_upper_air, targets_diagnostic, \
            varying_boundary_data, start_time_tensor = batch
            calendar = None

        # surface_t, upper_air_t, diagnostic_t: b c l h w, input at t=0
        # targets_surface, targets_upper_air, targets_diagnostic: b t c l h w, target trajectories for each variable
        # varying_boundary_data: b t+1 c h w, time-varying forcings and invariants for each target timestep

        b = surface_t.shape[0]
        nt = targets_surface.shape[1]
        nlevel = upper_air_t.shape[2]
        nlat = self.nlat
        nlon = self.nlon
        device = surface_t.device

        if self.downsample is not None:
            nlat = nlat // self.downsample.downsample_factor
            nlon = nlon // self.downsample.downsample_factor

        invariant = self.invariant_input.expand(b, -1, -1, -1).to(device) # b c nlat nlon

        # optimizes memory usage by calculating losses on the fly. Only plot certain timesteps, levels, variables of interest.

        loss_dict = {}
        t_plot = [0, 2, 4, 9] # 1day, 3day, 5day, 10day
        i_plot = 0
        plot_keys = ['2m_temperature', 'geopotential', 'PRATEsfc_24h', 'u_component_of_wind', 'temperature', 'specific_total_water']
        # init plot_dict
        pred_feat_dict = {}
        target_feat_dict = {}

        for surface_feat_name in self.surface_variables:
            loss_dict[surface_feat_name] = torch.zeros((b, nt), device=device) # b t
            if surface_feat_name in plot_keys:
                pred_feat_dict[surface_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device) # b t h w
                target_feat_dict[surface_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device) # b t h w

        for multilevel_feat_name in self.multilevel_variables:
            loss_dict[multilevel_feat_name] = torch.zeros((b, nt, nlevel), device=device) # b t l
            if multilevel_feat_name in plot_keys:
                pred_feat_dict[multilevel_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device) # b t l h w
                target_feat_dict[multilevel_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device) # b t l h w

        for diagnostic_feat_name in self.diagnostic_variables:
            loss_dict[diagnostic_feat_name] = torch.zeros((b, nt), device=device) # b t
            if diagnostic_feat_name in plot_keys:
                pred_feat_dict[diagnostic_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device) # b t h w
                target_feat_dict[diagnostic_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device) # b t h w

        x = self.preprocess(surface_t, upper_air_t, diagnostic_t) # b c h w

        for t in range(nt):
            # assemble forcings
            forcing_input = varying_boundary_data[:, t] # b c h w
            c_grid = assemble_forcing(forcing_input, invariant) # b c h w

            c_scalar_t = calendar[:, t] if calendar is not None else None
            y, y_last = self.forward(x, c_grid, return_model_last=True, c_scalar=c_scalar_t)
            # Pass the ACTUAL channel counts: disassemble_input defaults to
            # nsurface=6, ndiagnostic=15 — silently hardcoded to the Midway AMIP config.
            # Any config with a different diagnostic count (e.g. the Polaris E3SM config's
            # 3) mis-splits the channels and dies in the einops rearrange. This is a no-op
            # for Midway, where len(surface)=6 and len(diagnostic)=15 equal the defaults.
            surface_pred_decoded, multilevel_pred_decoded, diagnostic_pred_decoded = disassemble_input(
                y_last,
                nsurface=len(self.surface_variables),
                ndiagnostic=len(self.diagnostic_variables),
                nlevels=self.nlevels,
            )

            # update state
            x = y

            surface_target_t = targets_surface[:, t] # b c nlat nlon
            multilevel_target_t = targets_upper_air[:, t] # b c nlevel nlat nlon
            diagnostic_target_t = targets_diagnostic[:, t] # b c nlat nlon

            if self.downsample:
                surface_target_t, multilevel_target_t, diagnostic_target_t = self.downsample(surface_target_t, multilevel_target_t, diagnostic_target_t)

            surface_pred_denorm = self.n.surface_inv_transform(surface_pred_decoded)
            surface_true_denorm = self.n.surface_inv_transform(surface_target_t)
            multilevel_pred_denorm = self.n.upper_air_inv_transform(multilevel_pred_decoded)
            multilevel_true_denorm = self.n.upper_air_inv_transform(multilevel_target_t)
            diagnostic_pred_denorm = self.n.diagnostic_inv_transform(diagnostic_pred_decoded)
            diagnostic_true_denorm = self.n.diagnostic_inv_transform(diagnostic_target_t)

            # get losses
            for c, surface_feat_name in enumerate(self.surface_variables):
                loss_dict[surface_feat_name][:, t] = latitude_weighted_rmse(surface_pred_denorm[:, c],
                                                                           surface_true_denorm[:, c],
                                                                           nlon=nlon,
                                                                           nlat=nlat,
                                                                           with_time=False)
                if t in t_plot and surface_feat_name in plot_keys:
                    pred_feat_dict[surface_feat_name][:, i_plot] = surface_pred_denorm[:, c]
                    target_feat_dict[surface_feat_name][:, i_plot] = surface_true_denorm[:, c]

            for c, multilevel_feat_name in enumerate(self.multilevel_variables):
                loss_dict[multilevel_feat_name][:, t] = latitude_weighted_rmse(multilevel_pred_denorm[:, c],
                                                                                 multilevel_true_denorm[:, c],
                                                                                 nlon=nlon,
                                                                                 nlat=nlat,
                                                                                 with_time=False)
                if t in t_plot and multilevel_feat_name in plot_keys:
                    if multilevel_feat_name == 'geopotential':
                        l_plot = -10
                    elif multilevel_feat_name == 'u_component_of_wind':
                        l_plot = -13
                    elif multilevel_feat_name == 'temperature' or multilevel_feat_name == 'specific_total_water':
                        l_plot = -6

                    pred_feat_dict[multilevel_feat_name][:, i_plot] = multilevel_pred_denorm[:, c, l_plot]
                    target_feat_dict[multilevel_feat_name][:, i_plot] = multilevel_true_denorm[:, c, l_plot]

            for c, diagnostic_feat_name in enumerate(self.diagnostic_variables):
                loss_dict[diagnostic_feat_name][:, t] = latitude_weighted_rmse(diagnostic_pred_denorm[:, c],
                                                                              diagnostic_true_denorm[:, c],
                                                                              nlon=nlon,
                                                                              nlat=nlat,
                                                                              with_time=False)
                if t in t_plot and diagnostic_feat_name in plot_keys:
                    pred_feat_dict[diagnostic_feat_name][:, i_plot] = diagnostic_pred_denorm[:, c]
                    target_feat_dict[diagnostic_feat_name][:, i_plot] = diagnostic_true_denorm[:, c]

            if t in t_plot:
                i_plot += 1

        return loss_dict, pred_feat_dict, target_feat_dict
    
    def plot_predictions(self, pred_feat_dict, target_feat_dict):

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b t h w -> t h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu()
        pr_6h_pred = pred_feat_dict['PRATEsfc_24h'][0].cpu()
        pr_6h_target = target_feat_dict['PRATEsfc_24h'][0].cpu()

        z500_pred = pred_feat_dict['geopotential'][0].cpu() # b t h w -> t h w
        z500_target = target_feat_dict['geopotential'][0].cpu()
        pr_6h_pred = pred_feat_dict['PRATEsfc_24h'][0].cpu()
        pr_6h_target = target_feat_dict['PRATEsfc_24h'][0].cpu()
        u250_pred = pred_feat_dict['u_component_of_wind'][0].cpu()
        u250_target = target_feat_dict['u_component_of_wind'][0].cpu()
        t850_pred = pred_feat_dict['temperature'][0].cpu()
        t850_target = target_feat_dict['temperature'][0].cpu()
        q850_pred = pred_feat_dict['specific_total_water'][0].cpu()
        q850_target = target_feat_dict['specific_total_water'][0].cpu()

        plot_result(t2m_pred, # t h w
                    t2m_target,
                    f'{self.log_dir}/t2m_{self.current_epoch}.png')
        plot_result(z500_pred,
                    z500_target,
                    f'{self.log_dir}/z500_{self.current_epoch}.png')
        plot_result(pr_6h_pred,
                    pr_6h_target,
                    f'{self.log_dir}/PRATEsfc_{self.current_epoch}.png')
        plot_result(u250_pred,
                    u250_target,
                    f'{self.log_dir}/u250_{self.current_epoch}.png')
        plot_result(t850_pred,
                    t850_target,
                    f'{self.log_dir}/t850_{self.current_epoch}.png')
        plot_result(q850_pred,
                    q850_target,
                    f'{self.log_dir}/q850_{self.current_epoch}.png')
        
        plot_spectrum(t2m_pred,
                        t2m_target,
                        f'{self.log_dir}/t2m_spectrum_{self.current_epoch}.png')
        plot_spectrum(z500_pred,
                        z500_target,
                        f'{self.log_dir}/z500_spectrum_{self.current_epoch}.png')
        plot_spectrum(pr_6h_pred,
                        pr_6h_target,
                        f'{self.log_dir}/PRATEsfc_spectrum_{self.current_epoch}.png')
        plot_spectrum(u250_pred,
                        u250_target,
                        f'{self.log_dir}/u250_spectrum_{self.current_epoch}.png')
        plot_spectrum(t850_pred,
                        t850_target,
                        f'{self.log_dir}/t850_spectrum_{self.current_epoch}.png')
        plot_spectrum(q850_pred,
                        q850_target,
                        f'{self.log_dir}/q850_spectrum_{self.current_epoch}.png')
        
    def log_losses(self, loss_dict):
        # calculate the mean loss across batch, shape b t for each key, b t l for multilevel keys
        t2m_loss = loss_dict['2m_temperature'].mean(0) # surface temp, mean across batch dim
        pr_6h_loss = loss_dict['PRATEsfc_24h'].mean(0) # 6-hour accumulated PRATEsfc
        z500_loss = loss_dict['geopotential'][..., -10].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., -13].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., -6].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_total_water'][..., -6].mean(0) 
        
        self.log('val/t2m_1', t2m_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 6 hours
        self.log('val/t2m_3', t2m_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 1 day
        self.log('val/t2m_5', t2m_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 3 day
        self.log('val/t2m_10', t2m_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 5 day

        self.log('val/pr_6h_1', pr_6h_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_3', pr_6h_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_5', pr_6h_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_10', pr_6h_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/z500_1', z500_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_3', z500_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_5', z500_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_10', z500_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/u250_1', u250_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_3', u250_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_5', u250_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_10', u250_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/t850_1', t850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_3', t850_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_5', t850_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_10', t850_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/q850_1', q850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_3', q850_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_5', q850_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_10', q850_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
    
    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "muon":
            from muon import MuonWithAuxAdam
            # sa_blocks is a ModuleList of DiTBlock + DiTCrossAttentionBlock,
            # so cross-attention weights/biases are captured here too.
            hidden_weights = [p for p in self.model.sa_blocks.parameters() if p.ndim >= 2]
            hidden_gains_biases = [p for p in self.model.sa_blocks.parameters() if p.ndim < 2]

            # Input embeddings + unpatchify head go to Adam. Optional embedders
            # (c_grid, scalar/calendar, cross-attn context) may be None.
            nonhidden_modules = [
                self.model.patch_embed_main,
                self.model.t_embedder,
                self.model.unpatchify_layer,
            ]
            if self.model.c_grid_embed is not None:
                nonhidden_modules.append(self.model.c_grid_embed)
            if self.model.scalar_embedder is not None:
                nonhidden_modules.append(self.model.scalar_embedder)
            if self.model.ca_embed is not None:
                nonhidden_modules.append(self.model.ca_embed)
            nonhidden_params = [p for m in nonhidden_modules for p in m.parameters()]

            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                    lr=self.lr * 10, weight_decay=0.01),
                dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                    lr=self.lr, betas=(0.9, 0.95), weight_decay=0.01),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]