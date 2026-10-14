import lightning as L
import torch

from common.loss import latitude_weighted_rmse
from common.plotting import plot_result, plot_spectrum
from common.utils import assemble_forcing, assemble_input, disassemble_input


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


class CombinedModule(L.LightningModule):
    """Evaluation-only module combining a low-res forecaster with a downscaler.

    The forecaster (e.g. DiT with a stochastic interpolant) runs at reduced
    resolution. Its output is bilinearly upsampled and fed through the
    downscaler (e.g. UNet with a data-dependent interpolant) to produce a
    full-resolution prediction.

    Weights for the forecaster and downscaler are loaded manually from their
    own Lightning checkpoints. The public API mirrors
    :class:`modules.train_module.TrainModule` so this module can be used as a
    drop-in replacement in evaluation scripts (e.g. ``bias.py``).
    """

    def __init__(self, config: dict, normalizer=None):
        super().__init__()
        self.config = config
        self.dataconfig = config['data']
        self.batch_size = self.dataconfig['batch_size']
        self.surface_variables = self.dataconfig['surface_variables']
        self.multilevel_variables = self.dataconfig['upper_air_variables']
        self.diagnostic_variables = self.dataconfig['diagnostic_variables']
        self.horizontal_resolution = self.dataconfig['horizontal_resolution']
        self.nlat, self.nlon = self.horizontal_resolution
        self.nlevels = len(self.dataconfig['levels'])

        self.modelconfig = config['model']
        self.model_name = self.modelconfig.get('model_name', 'Combined')
        self.log_dir = config['training'].get('log_dir', '')
        self.plot_val = config['training'].get('plot_val', False)
        self.ensemble_size = self.modelconfig.get('ensemble_size', 1)
        self.return_calendar = self.dataconfig.get('return_calendar', False)

        # Which low-res latent feeds the downscaler: Euler-updated state 'y',
        # or the model x-prediction 'y_last'.
        self.downscaler_input = self.modelconfig.get('downscaler_input', 'y_last')
        assert self.downscaler_input in ('y', 'y_last'), \
            f"downscaler_input must be 'y' or 'y_last', got {self.downscaler_input}"

        self.n = normalizer
        self.constant_boundary_data = self.n.constant_boundary_data  # c nlat nlon
        self.invariant_input = self.constant_boundary_data.unsqueeze(0)  # 1 c nlat nlon

        # --- Forecaster (low-res) ---
        forecaster_cfg = self.modelconfig['forecaster']
        fc_name = forecaster_cfg['model_name']

        if fc_name == 'SI_X':
            from modules.models.DiT import DiT
            from modules.diffusion.x_interpolant import DynamicInterpolant
            from modules.layers.bilinear import BilinearEncoder

            self.forecaster_model = DiT(**forecaster_cfg['SI_X']['model'])
            self.forecaster_scheduler = DynamicInterpolant(**forecaster_cfg['SI_X']['scheduler'])
            self.downsample = BilinearEncoder()
        elif fc_name == 'SI_DiT':
            from modules.models.DiT import DiT
            from modules.diffusion.dynamic_interpolant import DriftScheduler

            self.forecaster_model = DiT(**forecaster_cfg['SI_DiT']['model'])
            self.forecaster_scheduler = DriftScheduler(**forecaster_cfg['SI_DiT']['scheduler'])
            self.downsample = None
        elif fc_name == 'FM':
            from modules.models.DiT import DiT
            from modules.diffusion.flow_matching import FlowMatching
            from modules.layers.bilinear import BilinearEncoder

            self.forecaster_model = DiT(**forecaster_cfg['FM']['model'])
            self.forecaster_scheduler = FlowMatching(**forecaster_cfg['FM']['scheduler'])
            self.downsample = BilinearEncoder()
        else:
            raise NotImplementedError(f"Forecaster model {fc_name} not supported")

        # --- Downscaler (low-res -> full-res) ---
        downscaler_cfg = self.modelconfig['downscaler']
        dc_name = downscaler_cfg['model_name']

        if dc_name == 'x_DDC':
            from modules.models.Unet import UNet
            from modules.layers.bilinear import BilinearDecoder
            from modules.diffusion.x_DDC import DataDependentInterpolant

            self.upsample = BilinearDecoder(**downscaler_cfg['x_DDC']['encoder'])
            self.downscaler_model = UNet(**downscaler_cfg['x_DDC']['decoder'])
            self.downscaler_scheduler = DataDependentInterpolant(**downscaler_cfg['x_DDC']['scheduler'])
        else:
            raise NotImplementedError(f"Downscaler model {dc_name} not supported")

        # --- Load pretrained weights ---
        forecaster_ckpt = config['training']['forecaster_checkpoint']
        downscaler_ckpt = config['training']['downscaler_checkpoint']

        self._load_from_ckpt(forecaster_ckpt,
                             [('model.', self.forecaster_model),
                              ('scheduler.', self.forecaster_scheduler)])
        self._load_from_ckpt(downscaler_ckpt,
                             [('model.', self.downscaler_model),
                              ('scheduler.', self.downscaler_scheduler)])

        # Freeze: evaluation-only.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.ddp = False

    @staticmethod
    def _load_from_ckpt(path, bindings):
        """Load state dict from a Lightning checkpoint into multiple submodules by prefix."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt['state_dict']
        for prefix, submodule in bindings:
            filtered = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not filtered:
                print(f"[CombinedModule] No keys with prefix '{prefix}' found in {path}")
                continue
            missing, unexpected = submodule.load_state_dict(filtered, strict=False)
            print(f"[CombinedModule] Loaded {len(filtered) - len(unexpected)}/{len(filtered)} "
                  f"keys for prefix '{prefix}' from {path}")
            if missing:
                print(f"  missing keys: {list(missing)}")
            if unexpected:
                print(f"  unexpected keys: {list(unexpected)}")

    def preprocess(self, surface_t, upper_air_t, diagnostic_t):
        """Downsample full-res inputs to low-res and assemble into forecaster input."""
        if self.downsample is not None:
            with torch.no_grad():
                surface_t, upper_air_t, diagnostic_t = self.downsample(surface_t, upper_air_t, diagnostic_t)
        return assemble_input(surface_t, upper_air_t, diagnostic_t)

    @torch.no_grad()
    def forward(self, x, c_grid, return_model_last=False, c_scalar=None):
        """Run the combined forecaster + downscaler pipeline.

        Args:
            x: low-res assembled forecaster state, shape ``(b, c, h_lr, w_lr)``.
            c_grid: low-res forcing+invariant tensor, shape ``(b, c, h_lr, w_lr)``.
            return_model_last: if True, also return the low-res rollout state.
            c_scalar: optional per-step scalar conditioning (e.g. calendar) routed
                to the forecaster DiT via :class:`_ModelWithScalar`.

        Returns:
            If ``return_model_last=False``:
                ``y_highres``: full-resolution prediction ``(b, c, nlat, nlon)``.
            If ``return_model_last=True``:
                ``(y_lowres, y_highres)`` where ``y_lowres`` is the Euler-updated
                low-res state used to roll the forecaster forward and
                ``y_highres`` is the downscaled full-resolution prediction.
        """
        if c_scalar is not None:
            forecaster_model = _ModelWithScalar(self.forecaster_model)
            forecaster_model.c_scalar = c_scalar
        else:
            forecaster_model = self.forecaster_model

        y_lowres, y_last_lowres = self.forecaster_scheduler.sample(
            forecaster_model, x, c_grid, return_model_last=True)

        z_lowres = y_lowres if self.downscaler_input == 'y' else y_last_lowres

        surf_lr, multi_lr, diag_lr = disassemble_input(
            z_lowres,
            nsurface=len(self.surface_variables),
            ndiagnostic=len(self.diagnostic_variables),
            nlevels=self.nlevels,
        )
        surf_up, multi_up, diag_up = self.upsample(surf_lr, multi_lr, diag_lr)
        z_upsampled = assemble_input(surf_up, multi_up, diag_up)

        y_highres = self.downscaler_scheduler.sample(self.downscaler_model, z_upsampled)

        if return_model_last:
            return y_lowres, y_highres
        return y_highres

    def validation_step(self, batch, batch_idx, evaluate=False):
        """Full-resolution medium-range rollout. Mirrors TrainModule.validation_step."""
        loss_dict, pred_feat_dict, target_feat_dict = self.predict(batch)

        if evaluate:
            return pred_feat_dict, target_feat_dict

        self.log_losses(loss_dict)

        if batch_idx == 0:
            if not self.ddp or self.global_rank == 0:
                if self.plot_val:
                    self.plot_predictions(pred_feat_dict, target_feat_dict)
                self.save_predictions(pred_feat_dict, target_feat_dict)

    @torch.no_grad()
    def predict(self, batch):
        """Autoregressive rollout at full resolution using the combined forecaster + downscaler.

        Rollout state is kept at low resolution (forecaster output ``y``); losses and
        stored predictions are at full resolution (downscaler output ``y_highres``).
        """
        if self.return_calendar:
            surface_t, upper_air_t, diagnostic_t, \
            targets_surface, targets_upper_air, targets_diagnostic, \
            varying_boundary_data, start_time_tensor, calendar = batch
        else:
            surface_t, upper_air_t, diagnostic_t, \
            targets_surface, targets_upper_air, targets_diagnostic, \
            varying_boundary_data, start_time_tensor = batch
            calendar = None

        # surface_t, upper_air_t, diagnostic_t: b c (l) h w, input at t=0 (full res)
        # targets_*: b t c (l) h w, full-res target trajectories
        # varying_boundary_data: b t c h w, full-res forcings for each target step

        b = surface_t.shape[0]
        nt = targets_surface.shape[1]
        nlevel = upper_air_t.shape[2]
        nlat = self.nlat
        nlon = self.nlon
        device = surface_t.device

        e = self.ensemble_size
        be = b * e

        invariant = self.invariant_input.expand(be, -1, -1, -1).to(device)

        loss_dict = {}
        t_plot = [0, 2, 4, 9]  # lead times: 1, 3, 5, 10 days (for 24h timedelta)
        i_plot = 0
        plot_keys = ['2m_temperature', 'geopotential', 'PRATEsfc_24h',
                     'u_component_of_wind', 'temperature', 'specific_total_water']

        pred_feat_dict = {}
        target_feat_dict = {}

        for surface_feat_name in self.surface_variables:
            loss_dict[surface_feat_name] = torch.zeros((b, nt), device=device)
            if surface_feat_name in plot_keys:
                pred_feat_dict[surface_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device)
                target_feat_dict[surface_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device)

        for multilevel_feat_name in self.multilevel_variables:
            loss_dict[multilevel_feat_name] = torch.zeros((b, nt, nlevel), device=device)
            if multilevel_feat_name in plot_keys:
                pred_feat_dict[multilevel_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device)
                target_feat_dict[multilevel_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device)

        for diagnostic_feat_name in self.diagnostic_variables:
            loss_dict[diagnostic_feat_name] = torch.zeros((b, nt), device=device)
            if diagnostic_feat_name in plot_keys:
                pred_feat_dict[diagnostic_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device)
                target_feat_dict[diagnostic_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=device)

        # Low-res forecaster state: bilinearly downsampled and assembled.
        # Ensemble members are folded into the leading batch dim so each one
        # rolls forward independently through the stochastic forecaster.
        surface_t_e = surface_t.repeat_interleave(e, dim=0)
        upper_air_t_e = upper_air_t.repeat_interleave(e, dim=0)
        diagnostic_t_e = diagnostic_t.repeat_interleave(e, dim=0)
        x = self.preprocess(surface_t_e, upper_air_t_e, diagnostic_t_e)

        for t in range(nt):
            forcing_input = varying_boundary_data[:, t].repeat_interleave(e, dim=0)
            c_grid = assemble_forcing(forcing_input, invariant)

            c_scalar_t = calendar[:, t].repeat_interleave(e, dim=0) if calendar is not None else None

            # y_lowres rolls the forecaster forward; y_highres is the full-res prediction.
            # Both have leading dim b*e — individual ensemble members continue independently.
            y_lowres, y_highres = self.forward(x, c_grid, return_model_last=True, c_scalar=c_scalar_t)
            surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(
                y_highres,
                nsurface=len(self.surface_variables),
                ndiagnostic=len(self.diagnostic_variables),
                nlevels=self.nlevels,
            )

            x = y_lowres

            # Targets are already at full resolution; no downsampling.
            surface_target_t = targets_surface[:, t]
            multilevel_target_t = targets_upper_air[:, t]
            diagnostic_target_t = targets_diagnostic[:, t]

            # Denormalize then average over the ensemble dim for losses/plots.
            surface_pred_denorm = self.n.surface_inv_transform(surface_pred)
            surface_pred_denorm = surface_pred_denorm.reshape(b, e, *surface_pred_denorm.shape[1:]).mean(dim=1)
            multilevel_pred_denorm = self.n.upper_air_inv_transform(multilevel_pred)
            multilevel_pred_denorm = multilevel_pred_denorm.reshape(b, e, *multilevel_pred_denorm.shape[1:]).mean(dim=1)
            diagnostic_pred_denorm = self.n.diagnostic_inv_transform(diagnostic_pred)
            diagnostic_pred_denorm = diagnostic_pred_denorm.reshape(b, e, *diagnostic_pred_denorm.shape[1:]).mean(dim=1)

            surface_true_denorm = self.n.surface_inv_transform(surface_target_t)
            multilevel_true_denorm = self.n.upper_air_inv_transform(multilevel_target_t)
            diagnostic_true_denorm = self.n.diagnostic_inv_transform(diagnostic_target_t)

            for c, surface_feat_name in enumerate(self.surface_variables):
                loss_dict[surface_feat_name][:, t] = latitude_weighted_rmse(
                    surface_pred_denorm[:, c], surface_true_denorm[:, c],
                    nlon=nlon, nlat=nlat, with_time=False)
                if t in t_plot and surface_feat_name in plot_keys:
                    pred_feat_dict[surface_feat_name][:, i_plot] = surface_pred_denorm[:, c]
                    target_feat_dict[surface_feat_name][:, i_plot] = surface_true_denorm[:, c]

            for c, multilevel_feat_name in enumerate(self.multilevel_variables):
                loss_dict[multilevel_feat_name][:, t] = latitude_weighted_rmse(
                    multilevel_pred_denorm[:, c], multilevel_true_denorm[:, c],
                    nlon=nlon, nlat=nlat, with_time=False)
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
                loss_dict[diagnostic_feat_name][:, t] = latitude_weighted_rmse(
                    diagnostic_pred_denorm[:, c], diagnostic_true_denorm[:, c],
                    nlon=nlon, nlat=nlat, with_time=False)
                if t in t_plot and diagnostic_feat_name in plot_keys:
                    pred_feat_dict[diagnostic_feat_name][:, i_plot] = diagnostic_pred_denorm[:, c]
                    target_feat_dict[diagnostic_feat_name][:, i_plot] = diagnostic_true_denorm[:, c]

            if t in t_plot:
                i_plot += 1

        return loss_dict, pred_feat_dict, target_feat_dict

    def save_predictions(self, pred_feat_dict, target_feat_dict):
        torch.save(pred_feat_dict, f'{self.log_dir}predictions_epoch_{self.current_epoch}.pt')
        torch.save(target_feat_dict, f'{self.log_dir}targets_epoch_{self.current_epoch}.pt')

    def plot_predictions(self, pred_feat_dict, target_feat_dict):
        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu()
        t2m_target = target_feat_dict['2m_temperature'][0].cpu()
        pr_6h_pred = pred_feat_dict['PRATEsfc_24h'][0].cpu()
        pr_6h_target = target_feat_dict['PRATEsfc_24h'][0].cpu()
        z500_pred = pred_feat_dict['geopotential'][0].cpu()
        z500_target = target_feat_dict['geopotential'][0].cpu()
        u250_pred = pred_feat_dict['u_component_of_wind'][0].cpu()
        u250_target = target_feat_dict['u_component_of_wind'][0].cpu()
        t850_pred = pred_feat_dict['temperature'][0].cpu()
        t850_target = target_feat_dict['temperature'][0].cpu()
        q850_pred = pred_feat_dict['specific_total_water'][0].cpu()
        q850_target = target_feat_dict['specific_total_water'][0].cpu()

        pairs = [
            ('t2m', t2m_pred, t2m_target),
            ('z500', z500_pred, z500_target),
            ('PRATEsfc', pr_6h_pred, pr_6h_target),
            ('u250', u250_pred, u250_target),
            ('t850', t850_pred, t850_target),
            ('q850', q850_pred, q850_target),
        ]
        for name, pred, target in pairs:
            plot_result(pred, target, f'{self.log_dir}/{name}_{self.current_epoch}.png')
            plot_spectrum(pred, target, f'{self.log_dir}/{name}_spectrum_{self.current_epoch}.png')

    def log_losses(self, loss_dict):
        """Log lat-weighted RMSE at 1, 3, 5, 10 day lead times for core variables."""
        t2m_loss = loss_dict['2m_temperature'].mean(0)
        pr_6h_loss = loss_dict['PRATEsfc_24h'].mean(0)
        z500_loss = loss_dict['geopotential'][..., -10].mean(0)
        u250_loss = loss_dict['u_component_of_wind'][..., -13].mean(0)
        t850_loss = loss_dict['temperature'][..., -6].mean(0)
        q850_loss = loss_dict['specific_total_water'][..., -6].mean(0)

        for name, tensor in [('t2m', t2m_loss), ('pr_6h', pr_6h_loss),
                             ('z500', z500_loss), ('u250', u250_loss),
                             ('t850', t850_loss), ('q850', q850_loss)]:
            self.log(f'val/{name}_1', tensor[0].item(),  on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log(f'val/{name}_3', tensor[2].item(),  on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log(f'val/{name}_5', tensor[4].item(),  on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log(f'val/{name}_10', tensor[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError("CombinedModule is evaluation-only")

    def configure_optimizers(self):
        raise NotImplementedError("CombinedModule is evaluation-only")
