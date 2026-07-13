import torch
import numpy as np
from torch.utils.data import Dataset
import h5pickle as h5f
import pickle 
from data.normalizer import Normalizer

SURFACE_VARIABLES = ["skin_temperature",
                     "surface_pressure",
                     "2m_temperature",
                     "2m_specific_humidity", 
                     "10m_u_component_of_wind", 
                     "10m_v_component_of_wind"]

MULTILEVEL_VARIABLES = ["temperature",
                  "u_component_of_wind",
                  "v_component_of_wind",
                  "geopotential",
                  "specific_humidity",
                  "specific_cloud_liquid_water_content",
                  "specific_cloud_ice_water_content",
                  "fraction_of_cloud_cover",
                  "vertical_velocity"]

MULTILEVEL_VARIABLES_2 = ["temperature",
                  "u_component_of_wind",
                  "v_component_of_wind",
                  "geopotential",
                  "specific_humidity"]

FORCING_VARIABLES = ["DSWRFtoa", 
                     "sea_surface_temperature", # has nans
                     "sea_ice_cover"]  # has nans

INVARIANT_VARIABLES = ["geopotential_at_surface", 
                       "land_sea_mask"]

DIAGNOSTIC_VARIABLES = ["USWRFtoa",
                        "ULWRFtoa", 
                       "USWRFsfc",
                       "ULWRFsfc",
                       "DSWRFsfc",
                       "DLWRFsfc",
                       "PRATEsfc",
                       "LHTFLsfc",
                       "SHTFLsfc"]

class AMIPData(Dataset):
    def __init__(self,
                 data_path,
                 norm_stats_path,
                 split="train",
                 normalize=True,
                 nsteps=1,   # how many steps to load
                 horizon=-1,
                 downsample_levels=1,
                 clouds=False
                 ):

        self.data_path = data_path 
        self.nsteps = nsteps
        self.norm_stats_path = norm_stats_path
        self.normalize = normalize
        self.split = split 
        self.downsample_levels = downsample_levels
        self.clouds = clouds    
        print(self.clouds)

        self.file = h5f.File(self.data_path, 'r') # has keys of 'split'
        self.data = self.file[split] # has keys of 'surface', 'multilevel', 'forcing', 'forcing_invariant', 'diagnostic', lat', 'lon', 'hour', 'day'
        self.n = Normalizer(norm_stats_path,
                            downsample_levels=downsample_levels,
                            clouds=clouds)

        # load refs
        self.surface = self.data['surface'] # t nlat nlon nsurface_channels
        self.multilevel = self.data['multilevel'] # t nlat nlon nlevels nmulti_channels
        self.diagnostic = self.data['diagnostic'] # t nlat nlon ndiagnostic_channels
        self.forcing = self.data['forcing'] # t nlat nlon nforcing_channels

        # directly load into memory
        self.invariants = torch.tensor(np.array(self.data['invariant'][:]), dtype=torch.float32) # nlat nlon n_invariant
        self.hour = torch.from_numpy(self.data['hour'][:]) # t
        self.day = torch.from_numpy(self.data['day'][:]) # t
        self.scalars = torch.concat([self.day.unsqueeze(-1), self.hour.unsqueeze(-1)], dim=-1) # t 2

        if self.normalize:
            self.invariants = self.n.normalize_invariant(self.invariants)

        if horizon == -1: # use full dataset
            self.horizon = len(self.hour) # t
        else:
            self.horizon = horizon

        self.num_samples = self.horizon - nsteps + 1
        print(f"Loaded {self.horizon} snapshots for {split} split, from {data_path}")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        surface = torch.tensor(np.array(self.surface[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon nsurface_channels
        multilevel = torch.tensor(np.array(self.multilevel[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlevels nlat nlon nmulti_channels
        diagnostic = torch.tensor(np.array(self.diagnostic[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon ndiagnostic_channels
        forcing = torch.tensor(np.array(self.forcing[idx:idx+self.nsteps]), dtype=torch.float32) # nsteps nlat nlon nforcing_channels
        scalars = self.scalars[idx:idx+self.nsteps] # nsteps 2

        if self.downsample_levels > 1:
            multilevel = multilevel[:, ::self.downsample_levels]

        if not self.clouds:
            multi_idx = torch.tensor([0, 1, 2, 3, 4])
            multilevel = multilevel[..., multi_idx] # remove cloud variables

        if self.normalize:
            surface = self.n.normalize_surface(surface)
            multilevel = self.n.normalize_multilevel(multilevel)
            diagnostic = self.n.normalize_diagnostic(diagnostic)
            forcing = self.n.normalize_forcing(forcing)

        return_dict = {"surface": surface,
                       "multilevel": multilevel,
                       "diagnostic": diagnostic,
                       "forcing": forcing,
                       "invariants": self.invariants,
                       "scalars": scalars
                       }
        
        return return_dict
    
class ClimatologyData(Dataset):
    def __init__(self,
                data_path,
                norm_stats_path,
                climatology_path,
                horizon=7308,
                start_time = 16072,
                normalize=True,
                ):
        
        # loads one initial frame and {horizon} timesteps of forcing data. Also returns true biases

        self.split = "train"
        self.data_path = data_path 
        self.climatology_path = climatology_path
        self.norm_stats_path = norm_stats_path
        self.normalize = normalize 
        self.horizon = horizon
        self.start_time = start_time # 0 is Jan 1st, 1979. 16072 is Jan 1st, 1990

        self.file = h5f.File(self.data_path, 'r') 
        self.data = self.file[self.split] 

        self.n = Normalizer(norm_stats_path)

        # can load initial conditions into memory
        self.surface = torch.tensor(np.array(self.data['surface'][start_time]), dtype=torch.float32) # nlat nlon nsurface_channels
        self.multilevel = torch.tensor(np.array(self.data['multilevel'][start_time]), dtype=torch.float32) # nlat nlon nlevels nmulti_channels
        self.invariants = torch.tensor(np.array(self.data['invariant'][:]), dtype=torch.float32) # nlat nlon n_invariant
        self.diagnostic = torch.tensor(np.array(self.data['diagnostic'][start_time]), dtype=torch.float32) # nlat nlon ndiagnostic_channels
        self.hour = torch.from_numpy(self.data['hour'][start_time:start_time + horizon]) # horizon
        self.day = torch.from_numpy(self.data['day'][start_time:start_time + horizon]) # horizon
        self.scalars = torch.concat([self.day.unsqueeze(-1), self.hour.unsqueeze(-1)], dim=-1) # horizon 2

        # Each year of forcing variables is around 1 GB, so don't load it into memoery
        self.forcing = self.data['forcing']

        if self.normalize:
            self.invariants = self.n.normalize_invariant(self.invariants)
            self.surface = self.n.normalize_surface(self.surface)
            self.multilevel = self.n.normalize_multilevel(self.multilevel)
            self.diagnostic = self.n.normalize_diagnostic(self.diagnostic)

        with open(climatology_path, 'rb') as file:
            self.climatology_dict = pickle.load(file) # unnormalized
            self.climatology_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.climatology_dict.items()}

        print(f"Loaded {horizon} time stamps for climatology")

    def __len__(self):
        return self.horizon

    def __getitem__(self, idx):
        forcing = torch.tensor(np.array(self.forcing[self.start_time + idx]), dtype=torch.float32) # nlat nlon nforcing_channels
        scalars = self.scalars[idx] # 2

        if self.normalize:
            forcing = self.n.normalize_forcing(forcing)

        if idx == 0:
            # return initial conditions
            return_dict = {"surface": self.surface,
                        "multilevel": self.multilevel,
                        "diagnostic": self.diagnostic,
                        "forcing": forcing,
                        "invariants": self.invariants,
                        "scalars": scalars,
                        "climatology_dict": self.climatology_dict
                        }
        else:
            return_dict = {"forcing": forcing,
                        "scalars": scalars}
    
        return return_dict

