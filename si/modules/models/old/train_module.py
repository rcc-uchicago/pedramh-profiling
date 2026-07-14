import lightning as L
import torch
from tqdm import tqdm

from common.loss import latitude_weighted_rmse, WeightedLoss
from common.plotting import plot_result, plot_spectrum, plot_bias
from data.amip import SURFACE_VARIABLES, MULTILEVEL_VARIABLES_2, DIAGNOSTIC_VARIABLES
from common.utils import assemble_forcing, disassemble_input, assemble_input, fix_state_dict

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
        self.modelconfig = config['model']
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.log_dir = config['training']['log_dir']
        self.optimizer_name = config['training']['optimizer']

        self.n = normalizer
        self.climatology = None
        self.diffusion=False 
        self.latent=False 

        if self.model_name == "sfno":
            from modules.models.SFNO import SphericalFourierNeuralOperatorNet
            self.model = SphericalFourierNeuralOperatorNet(params={},
                                                           **self.modelconfig["sfno"])
        elif self.model_name == 'SI_Arches':
            from modules.models.Arches_DiT import ArchesDiT
            from modules.diffusion.dynamic_interpolant import DriftScheduler
            self.model = ArchesDiT(**self.modelconfig['SI_Arches']["model"])
            self.scheduler = DriftScheduler(**self.modelconfig["SI_Arches"]['scheduler'])
            self.diffusion=True 
        elif self.model_name == "SI_DiT":
            from modules.models.DiT import DiT
            from modules.diffusion.dynamic_interpolant import DriftScheduler
            self.model = DiT(**self.modelconfig['SI_DiT']["model"])
            self.scheduler = DriftScheduler(**self.modelconfig["SI_DiT"]['scheduler'])
            self.diffusion=True
        elif self.model_name == "SI_Latent_DiT":
            from modules.models.DiT import DiT
            from modules.diffusion.dynamic_interpolant import DriftScheduler

            self.model = DiT(**self.modelconfig['SI_Latent_DiT']["model"])
            self.scheduler = DriftScheduler(**self.modelconfig["SI_Latent_DiT"]['scheduler'])
            self.diffusion=True
            self.latent = True
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if self.latent:
            from modules.models.AE_simple import BilinearDownsample
            #from modules.models.AE_decoder_hfs import DecoderHistory

            self.encoder = BilinearDownsample(**self.modelconfig['SI_Latent_DiT']["encoder"])
            #self.decoder = DecoderHistory(**self.modelconfig['SI_Latent_DiT']["decoder"])
            #self.initialize_decoder()

            self.criterion = WeightedLoss(latitude_resolution=45,longitude_resolution=90)
        else:
            self.criterion = WeightedLoss(latitude_resolution=180, longitude_resolution=360)

        if config['training']['strategy'] == 'ddp' or config['training']['strategy'] == 'ddp_find_unused_parameters_true':
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def initialize_decoder(self):
        checkpoint_path = self.modelconfig['SI_Latent_DiT']["decoder_checkpoint"]
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)['state_dict']
        state_dict = fix_state_dict(state_dict, prefix="decoder.")

        self.decoder.load_state_dict(state_dict)

        # freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()

        print(f"Initialized decoder from checkpoint {checkpoint_path}")

    def forward(self, x, c_grid, c_scalar):
        # x is flattened state, c is scalar conditioning

        if self.diffusion:
            y = self.scheduler.sample(self.model, x, c_grid, c_scalar)
        else: # directly predict
            y = self.model(x, c_grid, c_scalar)
            
        surface_pred, multilevel_pred, diagnostic_pred = disassemble_input(y)
        return surface_pred, multilevel_pred, diagnostic_pred
    
    def compute_loss(self, 
                     surface_pred, surface_target,
                     multilevel_pred, multilevel_target,
                     diagnostic_pred, diagnostic_target):

        return self.criterion(surface_pred, surface_target,
                              multilevel_pred, multilevel_target,
                              diagnostic_pred, diagnostic_target)
    
    
    def training_step(self, batch, batch_idx):

        surface_data = batch['surface'] # b t nlat nlon c
        multilevel_data = batch['multilevel'] # b t nlevel nlat nlon c
        forcing_data = batch['forcing'] # b t nlat nlon c
        invariant_input = batch['invariants'] # b nlat nlon c
        scalar_data = batch['scalars'] # b t 2
        diagnostic_data = batch['diagnostic'] # b t nlat nlon c

        surface_input = surface_data[:, 0] # b nlat nlon c
        multilevel_input = multilevel_data[:, 0] # b nlevel nlat nlon c
        diagnostic_input = diagnostic_data[:, 0] # b nlat nlon c
        forcing_input = forcing_data[:, 0] # b nlat nlon c

        surface_target = surface_data[:, 1] # b nlat nlon c
        multilevel_target = multilevel_data[:, 1] # b nlevel nlat nlon c
        diagnostic_target = diagnostic_data[:, 1] # b nlat nlon c

        x = assemble_input(surface_input, multilevel_input, diagnostic_input) # b c h w
        c_grid = assemble_forcing(forcing_input, invariant_input) # b c h w
        c_scalar = scalar_data[:, 0] # b 2
        y = assemble_input(surface_target, multilevel_target, diagnostic_target) # b c h w

        if self.latent:
            x = self.encoder(x)
            y = self.encoder(y)
            c_grid = self.encoder(c_grid) # destroys some information in the forcing/invariants. Can use a learnable encoder?

        if self.diffusion:
            loss = self.scheduler.compute_loss(self.model, self.criterion,
                                               x, c_grid, c_scalar, y)   
        else:
            surface_pred, multilevel_pred, diagnostic_pred= self.forward(x, c_grid, c_scalar) 

            loss = self.compute_loss(surface_pred, surface_target,
                                    multilevel_pred, multilevel_target,
                                    diagnostic_pred, diagnostic_target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)

        return loss 

    def validation_step(self, batch, batch_idx, dataloader_idx=0): 
        
        if dataloader_idx == 0:
            # each batch contains val_nsteps number of snapshots
            loss_dict, pred_feat_dict, target_feat_dict = self.predict_lowMem(batch)
            self.log_losses(loss_dict)

            # visualize the prediction for first batch and on one gpu
            if batch_idx == 0:
                if not self.ddp or self.global_rank == 0:
                    self.plot_predictions(pred_feat_dict, target_feat_dict, lowMem=True)

        elif dataloader_idx == 1:
            # the first batch (batch_idx=0) contains the initial conditions, forcing, and climatologies
            # each subsequent batch only contains forcing data and HoD, DoY
            return 
            if not self.ddp or self.global_rank == 0: # only run on one GPU
                if batch_idx == 0:
                    if self.climatology is None:
                        self.climatology = batch['climatology_dict'] # can save to persistent RAM, since only ~100 Mb

                    # reset buffers to initial conditions 
                        
                    self.surface_state = batch['surface'][:, 0]
                    self.multilevel_state = batch['multilevel'][:, 0]
                    self.diagnostic_state = batch['diagnostic'][:, 0]
                    self.invariant_state = batch['invariants']

                    self.surface_running_mean = self.n.denormalize_surface(self.surface_state)
                    self.multilevel_running_mean = self.n.denormalize_multilevel(self.multilevel_state)
                    self.diagnostic_running_mean = self.n.denormalize_diagnostic(self.diagnostic_state)

                    self.num = 1

                self.update(batch) # update buffers

                if batch_idx == self.trainer.num_val_batches[1] - 1:
                    bias_dict, pred_climatology_dict = self.compute_biases()
    
    @torch.no_grad()
    def predict(self, batch):
        surface_data = batch['surface'] # b t nlat nlon c
        multilevel_data = batch['multilevel'] # b t nlevel nlat nlon c
        forcing_data = batch['forcing'] # b t nlat nlon c
        invariant_input = batch['invariants'] # b nlat nlon c
        scalar_data = batch['scalars'] # b t 2
        diagnostic_data = batch['diagnostic'] # b t nlat nlon c
                
        surface_input = surface_data[:, 0] # b nlat nlon c
        multilevel_input = multilevel_data[:, 0] # b nlevel nlat nlon c
        diagnostic_input = diagnostic_data[:, 0] # b nlat nlon c

        surface_target = surface_data[:, 1:] # b t nlat nlon c
        multilevel_target = multilevel_data[:, 1:] # b t nlevel nlat nlon c
        diagnostic_target = diagnostic_data[:, 1:] # b t nlat nlon c

        surface_pred_all = torch.zeros_like(surface_target, device=surface_data.device) # b t nlat nlon c
        multilevel_pred_all = torch.zeros_like(multilevel_target, device=multilevel_data.device) # b t nlevel nlat nlon c
        diagnostic_pred_all = torch.zeros_like(diagnostic_target, device=diagnostic_data.device) # b t nlat nlon c

        for t in range(surface_target.shape[1]):
            # assemble forcings
            forcing_input = forcing_data[:, t] # b nlat nlon c
            c_scalar = scalar_data[:, t] # b 2
            c_grid = assemble_forcing(forcing_input, invariant_input) # b c h w
            x = assemble_input(surface_input, multilevel_input, diagnostic_input) # b c h w

            # make prediction
            surface_pred, multilevel_pred, diagnostic_pred = self.forward(x, c_grid, c_scalar)

            # save prediction
            surface_pred_all[:, t] = surface_pred
            multilevel_pred_all[:, t] =  multilevel_pred
            diagnostic_pred_all[:, t] = diagnostic_pred

            # update inputs
            surface_input = surface_pred
            multilevel_input = multilevel_pred
            diagnostic_input = diagnostic_pred

        # denormalize
        surface_pred_all = self.n.denormalize_surface(surface_pred_all)
        multilevel_pred_all = self.n.denormalize_multilevel(multilevel_pred_all)
        diagnostic_pred_all = self.n.denormalize_diagnostic(diagnostic_pred_all)
        surface_target = self.n.denormalize_surface(surface_target)
        multilevel_target = self.n.denormalize_multilevel(multilevel_target)
        diagnostic_target = self.n.denormalize_diagnostic(diagnostic_target)

        pred_feat_dict = {}
        target_feat_dict = {}

        for c, surface_feat_name in enumerate(SURFACE_VARIABLES):
            pred_feat_dict[surface_feat_name] = surface_pred_all[..., c] # b t nlat nlon
            target_feat_dict[surface_feat_name] = surface_target[..., c]

        for c, multilevel_feat_name in enumerate(MULTILEVEL_VARIABLES_2):
            pred_feat_dict[multilevel_feat_name] = multilevel_pred_all[..., c] # b t nlevel nlat nlon 
            target_feat_dict[multilevel_feat_name] = multilevel_target[..., c]

        for c, diagnostic_feat_name in enumerate(DIAGNOSTIC_VARIABLES):
            pred_feat_dict[diagnostic_feat_name] = diagnostic_pred_all[..., c] # b t nlat nlon
            target_feat_dict[diagnostic_feat_name] = diagnostic_target[..., c]

        nlat, nlon = surface_data.shape[2], surface_data.shape[3]
        loss_dict = {k:
                        latitude_weighted_rmse(pred_feat_dict[k], target_feat_dict[k],
                                                nlon=nlon, nlat=nlat,
                                                ) for k in pred_feat_dict.keys()} # b t or b t l for each key
        
        return loss_dict, pred_feat_dict, target_feat_dict
        
    @torch.no_grad()
    def update(self, batch):
        # b = 1 
        # assume these are normalized

        forcing_input = batch['forcing'][:, 0] # b nlat nlon c

        surface_input = self.surface_state # b nlat nlon c
        multilevel_input = self.multilevel_state # b nlevel nlat nlon c
        diagnostic_input = self.diagnostic_state # b nlat nlon c
        invariant_input = self.invariant_state # b nlat nlon c

        x = assemble_input(surface_input, multilevel_input, diagnostic_input) # b c h w
        c_grid = assemble_forcing(forcing_input, invariant_input) # b c h w
        c_scalar = batch['scalars'] # b 2

        surface_pred, multilevel_pred, diagnostic_pred = self.forward(x, c_grid, c_scalar)
        
        self.surface_state = surface_pred
        self.multilevel_state = multilevel_pred
        self.diagnostic_state = diagnostic_pred

        self.surface_running_mean += self.n.denormalize_surface(surface_pred)
        self.multilevel_running_mean += self.n.denormalize_multilevel(multilevel_pred)
        self.diagnostic_running_mean += self.n.denormalize_diagnostic(diagnostic_pred)

        self.num += 1

    @torch.no_grad()
    def compute_biases(self):
        surface_climatology = self.surface_running_mean / self.num # b nlat nlon c
        multilevel_climatology = self.multilevel_running_mean / self.num # b nlevel nlat nlon c
        diagnostic_climatology = self.diagnostic_running_mean / self.num # b nlat nlon c

        pred_feat_dict = {}
        for c, surface_feat_name in enumerate(SURFACE_VARIABLES):
            pred_feat_dict[surface_feat_name] = surface_climatology[..., c] # b nlat nlon 

        for c, multilevel_feat_name in enumerate(MULTILEVEL_VARIABLES_2):
            pred_feat_dict[multilevel_feat_name] = multilevel_climatology[..., c] # b nlevel nlat nlon

        for c, diagnostic_feat_name in enumerate(DIAGNOSTIC_VARIABLES):
            pred_feat_dict[diagnostic_feat_name] = diagnostic_climatology[..., c] # b nlat nlon 

        key_list = ['2m_temperature', 'PRATEsfc', 'geopotential', 'temperature', 'u_component_of_wind', 'specific_humidity']
        result_dict = {}

        for var_name in key_list:
            true_climatology = self.climatology[var_name].unsqueeze(0) # 1 nlat nlon or 1 nlevel nlat nlon
            pred_climatology = pred_feat_dict[var_name] # 1 nlat nlon or 1 nlevel nlat nlon
            
            l = -1 
            if var_name == "geopotential":
                l = 10
            elif var_name == "u_component_of_wind":
                l = 13
            elif var_name == "temperature" or var_name == "specific_humidity":
                l = 6
            
            if l != -1:
                pred_climatology = pred_climatology[:, l, ...]
                true_climatology = true_climatology[:, l, ...]

            loss = latitude_weighted_rmse(pred_climatology, 
                                        true_climatology,
                                        nlon=360,
                                        nlat=180,
                                        with_time=False)
            
            result_dict[var_name] = loss.item()
            plot_bias(pred_climatology[0].cpu(), true_climatology[0].cpu(), save_path=f"{self.log_dir}/{var_name}_bias_{self.current_epoch}.png")

        self.log('bias/t2m', result_dict['2m_temperature'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/pr_6h', result_dict['PRATEsfc'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/z500', result_dict['geopotential'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/u250', result_dict['u_component_of_wind'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/t850', result_dict['temperature'], on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('bias/q850', result_dict['specific_humidity'], on_step=False, on_epoch=True, sync_dist=self.ddp)

        return result_dict, pred_feat_dict
    
    @torch.no_grad()
    def predict_lowMem(self, batch):
        surface_data = batch['surface'] # b t nlat nlon c
        multilevel_data = batch['multilevel'] # b t nlevel nlat nlon c
        forcing_data = batch['forcing'] # b t nlat nlon c
        invariant_input = batch['invariants'] # b nlat nlon c
        scalar_data = batch['scalars'] # b t 2
        diagnostic_data = batch['diagnostic'] # b t nlat nlon c

        b = surface_data.shape[0]
        nt = surface_data.shape[1]
        nlevel = multilevel_data.shape[2]
                
        surface_input = surface_data[:, 0] # b nlat nlon c
        multilevel_input = multilevel_data[:, 0] # b nlevel nlat nlon c
        diagnostic_input = diagnostic_data[:, 0] # b nlat nlon c
        nlat = surface_input.shape[1]
        nlon = surface_input.shape[2]

        if self.latent:
            nlat = nlat // 4
            nlon = nlon // 4

        surface_target = surface_data[:, 1:] # b t nlat nlon c
        multilevel_target = multilevel_data[:, 1:] # b t nlevel nlat nlon c
        diagnostic_target = diagnostic_data[:, 1:] # b t nlat nlon c

        # optimizes memory usage by calculating losses on the fly. Only plot certain timesteps, levels, variables of interest.

        loss_dict = {}
        t_plot = [0, 3, 11, 19, 39] # 6hour, 1day, 3day, 5day, 10day
        i_plot = 0
        plot_keys = ['2m_temperature', 'geopotential', 'PRATEsfc', 'u_component_of_wind', 'temperature', 'specific_humidity']
        # init plot_dict
        pred_feat_dict = {}
        target_feat_dict = {}

        for surface_feat_name in SURFACE_VARIABLES:
            loss_dict[surface_feat_name] = torch.zeros((b, nt), device=surface_data.device) # b t
            if surface_feat_name in plot_keys:
                pred_feat_dict[surface_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=surface_data.device) # b t h w
                target_feat_dict[surface_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=surface_data.device) # b t h w

        for multilevel_feat_name in MULTILEVEL_VARIABLES_2:
            loss_dict[multilevel_feat_name] = torch.zeros((b, nt, nlevel), device=multilevel_data.device) # b t l
            if multilevel_feat_name in plot_keys:
                pred_feat_dict[multilevel_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=multilevel_data.device) # b t l h w
                target_feat_dict[multilevel_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=multilevel_data.device) # b t l h w

        for diagnostic_feat_name in DIAGNOSTIC_VARIABLES:
            loss_dict[diagnostic_feat_name] = torch.zeros((b, nt), device=diagnostic_data.device) # b t
            if diagnostic_feat_name in plot_keys:
                pred_feat_dict[diagnostic_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=diagnostic_data.device) # b t h w
                target_feat_dict[diagnostic_feat_name] = torch.zeros((b, len(t_plot), nlat, nlon), device=diagnostic_data.device) # b t h w

        for t in range(surface_target.shape[1]):
            # assemble forcings
            forcing_input = forcing_data[:, t] # b nlat nlon c
            c_scalar = scalar_data[:, t] # b 2

            x = assemble_input(surface_input, multilevel_input, diagnostic_input) # b c h w
            c_grid = assemble_forcing(forcing_input, invariant_input) # b c h w

            if self.latent:
                x = self.encoder(x) if t == 0 else x
                c_grid = self.encoder(c_grid)
            
            # make prediction
            surface_pred, multilevel_pred, diagnostic_pred = self.forward(x, c_grid, c_scalar)

            surface_target_t = surface_target[:, t] # b nlat nlon c
            multilevel_target_t = multilevel_target[:, t] # b nlevel nlat nlon c
            diagnostic_target_t = diagnostic_target[:, t] # b nlat nlon c

            if self.latent:
                #surface_pred_decoded, multilevel_pred_decoded, diagnostic_pred_decoded = \
                #    self.decoder(surface_history, multilevel_history, diagnostic_history,
                #                 surface_pred, multilevel_pred, diagnostic_pred)
                surface_pred_decoded = surface_pred
                multilevel_pred_decoded = multilevel_pred
                diagnostic_pred_decoded = diagnostic_pred
                
                target_t = assemble_input(surface_target_t, multilevel_target_t, diagnostic_target_t)
                target_t = self.encoder(target_t)
                surface_target_t, multilevel_target_t, diagnostic_target_t = disassemble_input(target_t)

            else:
                surface_pred_decoded = surface_pred
                multilevel_pred_decoded = multilevel_pred
                diagnostic_pred_decoded = diagnostic_pred

            surface_pred_denorm = self.n.denormalize_surface(surface_pred_decoded)
            surface_true_denorm = self.n.denormalize_surface(surface_target_t)  
            multilevel_pred_denorm = self.n.denormalize_multilevel(multilevel_pred_decoded)
            multilevel_true_denorm = self.n.denormalize_multilevel(multilevel_target_t)
            diagnostic_pred_denorm = self.n.denormalize_diagnostic(diagnostic_pred_decoded)
            diagnostic_true_denorm = self.n.denormalize_diagnostic(diagnostic_target_t)

            # get losses
            for c, surface_feat_name in enumerate(SURFACE_VARIABLES):
                loss_dict[surface_feat_name][:, t] = latitude_weighted_rmse(surface_pred_denorm[..., c], 
                                                                           surface_true_denorm[..., c],
                                                                           nlon=nlon,
                                                                           nlat=nlat,
                                                                           with_time=False)
                if t in t_plot and surface_feat_name in plot_keys:
                    pred_feat_dict[surface_feat_name][:, i_plot, ...] = surface_pred_denorm[..., c]
                    target_feat_dict[surface_feat_name][:, i_plot, ...] = surface_true_denorm[..., c]
                    
            for c, multilevel_feat_name in enumerate(MULTILEVEL_VARIABLES_2):
                loss_dict[multilevel_feat_name][:, t, :] = latitude_weighted_rmse(multilevel_pred_denorm[..., c], 
                                                                                 multilevel_true_denorm[..., c],
                                                                                 nlon=nlon,
                                                                                 nlat=nlat,
                                                                                 with_time=False)
                if t in t_plot and multilevel_feat_name in plot_keys:
                    if multilevel_feat_name == 'geopotential':
                        l_plot = 10
                    elif multilevel_feat_name == 'u_component_of_wind':
                        l_plot = 13
                    elif multilevel_feat_name == 'temperature' or multilevel_feat_name == 'specific_humidity':
                        l_plot = 6

                    pred_feat_dict[multilevel_feat_name][:, i_plot, ...] = multilevel_pred_denorm[:, l_plot, ..., c]
                    target_feat_dict[multilevel_feat_name][:, i_plot, ...] = multilevel_true_denorm[:, l_plot, ..., c]

            for c, diagnostic_feat_name in enumerate(DIAGNOSTIC_VARIABLES):
                loss_dict[diagnostic_feat_name][:, t] = latitude_weighted_rmse(diagnostic_pred_denorm[..., c], 
                                                                              diagnostic_true_denorm[..., c],
                                                                              nlon=nlon,
                                                                              nlat=nlat,
                                                                              with_time=False)
                if t in t_plot and diagnostic_feat_name in plot_keys:
                    pred_feat_dict[diagnostic_feat_name][:, i_plot, ...] = diagnostic_pred_denorm[..., c]
                    target_feat_dict[diagnostic_feat_name][:, i_plot, ...] = diagnostic_true_denorm[..., c]
            
            # update inputs
            surface_input = surface_pred
            multilevel_input = multilevel_pred
            diagnostic_input = diagnostic_pred

            if t in t_plot:
                i_plot += 1

        return loss_dict, pred_feat_dict, target_feat_dict
    
    def plot_predictions(self, pred_feat_dict, target_feat_dict, lowMem=False):

        t2m_pred = pred_feat_dict['2m_temperature'][0].cpu() #b t h w -> t h w 
        t2m_target = target_feat_dict['2m_temperature'][0].cpu()
        pr_6h_pred = pred_feat_dict['PRATEsfc'][0].cpu()
        pr_6h_target = target_feat_dict['PRATEsfc'][0].cpu()

        if lowMem:
            z500_pred = pred_feat_dict['geopotential'][0].cpu() # b t h w -> t h w
            z500_target = target_feat_dict['geopotential'][0].cpu()
            pr_6h_pred = pred_feat_dict['PRATEsfc'][0].cpu()
            pr_6h_target = target_feat_dict['PRATEsfc'][0].cpu()
            u250_pred = pred_feat_dict['u_component_of_wind'][0].cpu()
            u250_target = target_feat_dict['u_component_of_wind'][0].cpu()
            t850_pred = pred_feat_dict['temperature'][0].cpu()
            t850_target = target_feat_dict['temperature'][0].cpu()
            q850_pred = pred_feat_dict['specific_humidity'][0].cpu()
            q850_target = target_feat_dict['specific_humidity'][0].cpu()
        else:
            z500_pred = pred_feat_dict['geopotential'][0, :, 10, ...].cpu() # b t l h w -> t h w
            z500_target = target_feat_dict['geopotential'][0, :, 10, ...].cpu()
            u250_pred = pred_feat_dict['u_component_of_wind'][0, :, 13, ...].cpu()
            u250_target = target_feat_dict['u_component_of_wind'][0, :, 13, ...].cpu()
            t850_pred = pred_feat_dict['temperature'][0, :, 6, ...].cpu()
            t850_target = target_feat_dict['temperature'][0, :, 6, ...].cpu()
            q850_pred = pred_feat_dict['specific_humidity'][0, :, 6, ...].cpu()
            q850_target = target_feat_dict['specific_humidity'][0, :, 6, ...].cpu()

        #print(t2m_pred.shape)
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
        pr_6h_loss = loss_dict['PRATEsfc'].mean(0) # 6-hour accumulated PRATEsfc
        z500_loss = loss_dict['geopotential'][..., 10].mean(0) # geopotential at level=10
        u250_loss = loss_dict['u_component_of_wind'][..., 13].mean(0) # u wind at level=13
        t850_loss = loss_dict['temperature'][..., 6].mean(0) # temp at level=6
        q850_loss = loss_dict['specific_humidity'][..., 6].mean(0) # specific humidity at level=6
        
        self.log('val/t2m_6', t2m_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 6 hours
        self.log('val/t2m_24', t2m_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 1 day
        self.log('val/t2m_72', t2m_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 3 day
        self.log('val/t2m_120', t2m_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 5 day
        self.log('val/t2m_240', t2m_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp) # 10 day

        self.log('val/pr_6h_6', pr_6h_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_24', pr_6h_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_72', pr_6h_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_120', pr_6h_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/pr_6h_240', pr_6h_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/z500_6', z500_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_24', z500_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_72', z500_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_120', z500_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/z500_240', z500_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/u250_6', u250_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_24', u250_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_72', u250_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_120', u250_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/u250_240', u250_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/t850_6', t850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_24', t850_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_72', t850_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_120', t850_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/t850_240', t850_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)

        self.log('val/q850_6', q850_loss[0].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_24', q850_loss[3].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_72', q850_loss[11].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_120', q850_loss[19].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
        self.log('val/q850_240', q850_loss[39].item(), on_step=False, on_epoch=True, sync_dist=self.ddp)
    
    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "shampoo":
            from distributed_shampoo import (
                AdamPreconditionerConfig,
                DDPDistributedConfig,
                DistributedShampoo,
            )

            optimizer = DistributedShampoo(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                epsilon=1e-12,
                weight_decay=1e-05,
                max_preconditioner_dim=8192,
                precondition_frequency=100,
                use_decoupled_weight_decay=True,
                grafting_config=AdamPreconditionerConfig(
                    beta2=0.999,
                    epsilon=1e-12,
                ),
                distributed_config=DDPDistributedConfig(
                    communication_dtype=torch.float32,
                    num_trainers_per_group=8,
                    communicate_params=False,
                ),
            )
        elif self.optimizer_name == "soap":
            from distributed_shampoo import (
                DistributedShampoo,
                DefaultSOAPConfig,
            )

            optimizer = DistributedShampoo(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                epsilon=1e-12,
                weight_decay=1e-06,
                max_preconditioner_dim=8192,
                precondition_frequency=100,
                preconditioner_config=DefaultSOAPConfig,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        return [optimizer], [scheduler]
    