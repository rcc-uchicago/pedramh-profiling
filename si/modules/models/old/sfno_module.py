import lightning as L
import torch

from common.loss import latitude_weighted_rmse, WeightedLoss, SpectralBaseLoss
from common.plotting import plot_result, plot_spectrum
from common.utils import assemble_forcing, disassemble_input, assemble_input
from modules.models.SFNO.sfnonet import SphericalFourierNeuralOperatorNet
from data.band_limiter import SphericalBandLimitVector

class SFNOModule(L.LightningModule):
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
        self.horizontal_resolution = self.dataconfig['horizontal_resolution']
        self.nlat, self.nlon = self.horizontal_resolution
        self.nlevels = len(self.dataconfig['levels'])

        self.modelconfig = config['model']
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.weight_decay = self.modelconfig["weight_decay"]
        self.log_dir = config['training']['log_dir']
        self.K = self.modelconfig["K"] # bandlimit hyperparameter

        self.n = normalizer
        self.constant_boundary_data = self.n.constant_boundary_data # surface geopotential, lsm, in shape c nlat nlon
        self.invariant_input = self.constant_boundary_data.unsqueeze(0) # 1 c nlat nlon

        self.model = SphericalFourierNeuralOperatorNet(**self.modelconfig[self.model_name])
        self.band_limiter = SphericalBandLimitVector(nlat = self.nlat, nlon = self.nlon, K = self.K)

        self.criterion = WeightedLoss(latitude_resolution=self.nlat,
                            longitude_resolution=self.nlon,
                            nlevels = self.nlevels,
                            level_weight=self.modelconfig["level_weight"],
                            surface_variable_weight=self.modelconfig["surface_weight"],
                            multi_level_variable_weight=self.modelconfig["multi_level_weight"],
                            use_diagnostic=False)

        self.spectral_loss_weight = self.modelconfig["spectral_loss_weight"]
        self.spectral_criterion = SpectralBaseLoss(img_shape=(self.nlat, self.nlon),
                                                   use_diagnostic=False,
                                                   K=self.K)
    
        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        surface_t, upper_air_t, surface_t1, upper_air_t1, varying_boundary_data = batch
        device = surface_t.device

        surface_t, upper_air_t = self.band_limiter(surface_t, upper_air_t)
        surface_t1, upper_air_t1 = self.band_limiter(surface_t1, upper_air_t1)

        x = assemble_input(surface_t, upper_air_t) # b c h w
        invariant = self.invariant_input.expand(surface_t.shape[0], -1, -1, -1).to(device) # b c nlat nlon
        c_grid = assemble_forcing(varying_boundary_data, invariant) # b c h w

        x = torch.cat((x, c_grid), dim=1) # b c h w

        pred = self.forward(x) # b c h w

        surface_pred, multilevel_pred = disassemble_input(pred, use_diagnostic=False)

        surface_pred, multilevel_pred = self.band_limiter(surface_pred, multilevel_pred)

        pixel_loss = self.criterion(surface_pred, surface_t1, 
                                    multilevel_pred, upper_air_t1)
        
        spectral_loss = self.spectral_loss_weight * self.spectral_criterion(surface_pred, surface_t1, 
                                                                            multilevel_pred, upper_air_t1)

        loss = pixel_loss + spectral_loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        self.log("train/pixel_loss", pixel_loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        self.log("train/spectral_loss", spectral_loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss 

    def validation_step(self, batch, batch_idx, evaluate=False): 
        # each batch contains val_nsteps number of snapshots
        loss_dict, pred_feat_dict, target_feat_dict = self.predict(batch)

        if evaluate:
            return pred_feat_dict, target_feat_dict

        self.log_losses(loss_dict)

        # visualize the prediction for first batch and on one gpu
        if batch_idx == 0:
            if not self.ddp or self.global_rank == 0:
                self.save_predictions(pred_feat_dict, target_feat_dict)
    
    def save_predictions(self, pred_feat_dict, target_feat_dict):
        torch.save(pred_feat_dict, f'{self.log_dir}predictions_epoch_{self.current_epoch}.pt')
        torch.save(target_feat_dict, f'{self.log_dir}targets_epoch_{self.current_epoch}.pt')

    @torch.no_grad()
    def predict(self, batch):

        surface_t, upper_air_t, targets_surface, targets_upper_air, varying_boundary_data, start_time_tensor = batch

        # surface_t, upper_air_t: b c l h w, input at t=0
        # targets_surface, targets_upper_air: b t c l h w, target trajectories for each variable
        # varying_boundary_data: b t+1 c h w, time-varying forcings for each target timestep

        b = surface_t.shape[0]
        nt = targets_surface.shape[1]
        nlevel = self.nlevels
        nlat = self.nlat
        nlon = self.nlon
        device = surface_t.device

        invariant = self.invariant_input.expand(b, -1, -1, -1).to(device) # b c nlat nlon

        # optimizes memory usage by calculating losses on the fly. Only plot certain timesteps, levels, variables of interest.

        loss_dict = {}
        t_plot = [0, 2, 4, 9] # 1day, 3day, 5day, 10day
        i_plot = 0
        plot_keys = ['2m_temperature', 'geopotential', 'u_component_of_wind', 'temperature', 'specific_total_water']
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

        surface_t, upper_air_t = self.band_limiter(surface_t, upper_air_t)
        prognostic_input = assemble_input(surface_t, upper_air_t) # b c h w

        for t in range(nt):
            # assemble forcings
            forcing_input = varying_boundary_data[:, t] # b c h w
            c_grid = assemble_forcing(forcing_input, invariant) # b c h w

            x = torch.cat((prognostic_input, c_grid), dim=1) # b c h w

            y = self.forward(x)

            surface_pred_t, multilevel_pred_t = disassemble_input(y, use_diagnostic=False)
            surface_pred_t, multilevel_pred_t = self.band_limiter(surface_pred_t, multilevel_pred_t)

            # update inputs for next autoregressive step
            prognostic_input = assemble_input(surface_pred_t, multilevel_pred_t) # b c h w

            surface_target_t = targets_surface[:, t] # b c nlat nlon
            multilevel_target_t = targets_upper_air[:, t] # b c nlevel nlat nlon

            surface_target_t, multilevel_target_t = self.band_limiter(surface_target_t, multilevel_target_t)

            surface_pred_denorm = self.n.surface_inv_transform(surface_pred_t)
            surface_true_denorm = self.n.surface_inv_transform(surface_target_t)
            multilevel_pred_denorm = self.n.upper_air_inv_transform(multilevel_pred_t)
            multilevel_true_denorm = self.n.upper_air_inv_transform(multilevel_target_t)

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

            if t in t_plot:
                i_plot += 1

        return loss_dict, pred_feat_dict, target_feat_dict
    
    def plot_predictions(self, pred_feat_dict, target_feat_dict):

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b t h w -> t h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu()

        z500_pred = pred_feat_dict['geopotential'][0].cpu() # b t h w -> t h w
        z500_target = target_feat_dict['geopotential'][0].cpu()
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
        z500_loss = loss_dict['geopotential'][..., -10].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., -13].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., -6].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_total_water'][..., -6].mean(0) 
        
        self.log('val/t2m_1', t2m_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 6 hours
        self.log('val/t2m_3', t2m_loss[2].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 1 day
        self.log('val/t2m_5', t2m_loss[4].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 3 day
        self.log('val/t2m_10', t2m_loss[9].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 5 day

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        return optimizer
   