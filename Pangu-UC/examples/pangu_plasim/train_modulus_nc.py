import argparse
import os
from datetime import datetime
import cftime
import numpy as np


from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
import pandas as pd
import modulus

# vscode Relative path
import sys
sys.path.append("../../")
sys.path.append("/glade/work/awikner/Pangu-UC")

from weatherlearn.models import PanguPlasimModulus
from data_loader_nc import DatasetFromFolder

parser = argparse.ArgumentParser(description="Train Pangu_Plasim Models")
parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--boundary_dir", type=str, default='boundary_vars', help="Path to boundary data directory")
parser.add_argument("--train_year_start", type=int, required=True, help="Start year for the training data")
parser.add_argument("--train_year_end", type=int, required=True, help="End year for the training data")
parser.add_argument("--val_year_start", type=int, required=True, help="Start year for the validation data")
parser.add_argument("--val_year_end", type=int, required=True, help="End year for the validation data")
parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per batch")
parser.add_argument("--surface_variables", nargs="*", required=True, help="List of surface variables to include")
parser.add_argument("--upper_air_variables", nargs="*", required=True, help="List of upper air variables to include")
parser.add_argument("--constant_boundary_variables", nargs="*", required=True, help="List of constant boundary variables to include")
parser.add_argument("--varying_boundary_variables", nargs="*", required=True, help="List of varying boundary variables to include")
parser.add_argument("--surface_mean", type=str, default="surface_mean.nc", help="Name of surface mean file in datadir")
parser.add_argument("--surface_std", type=str, default="surface_std.nc", help="Name of surface std file in datadir")
parser.add_argument("--upper_air_mean", type=str, default="upper_air_mean.nc", help="Name of upper_air mean file in datadir")
parser.add_argument("--upper_air_std", type=str, default="upper_air_std.nc", help="Name of upper air std file in datadir")
parser.add_argument("--calendar", type=str, default = 'proleptic_gregorian', help="Type of calendar for data (cftime)")
parser.add_argument("--timedelta_hours", type=int, default=6, help="Prediction lead time in hours")

if __name__ == "__main__":
    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs
    DATA_DIR = opt.data_dir
    BOUNDARY_DIR = opt.boundary_dir
    YEAR_START_TRAIN = opt.train_year_start
    YEAR_END_TRAIN = opt.train_year_end
    YEAR_START_VAL = opt.val_year_start
    YEAR_END_VAL = opt.val_year_end
    SURFACE_VARIABLES = opt.surface_variables
    UPPER_AIR_VARIABLES = opt.upper_air_variables
    CONSTANT_BOUNDARY_VARIABLES = opt.constant_boundary_variables
    VARYING_BOUNDARY_VARIABLES = opt.varying_boundary_variables
    SURFACE_MEAN = opt.surface_mean
    SURFACE_STD  = opt.surface_std
    UPPER_AIR_MEAN = opt.upper_air_mean
    UPPER_AIR_STD  = opt.upper_air_std
    CALENDAR = opt.calendar
    TIMEDELTA_HOURS = opt.timedelta_hours
    BATCH_SIZE=opt.batch_size

    train_set = DatasetFromFolder(DATA_DIR, YEAR_START_TRAIN, YEAR_END_TRAIN, "train", SURFACE_VARIABLES, UPPER_AIR_VARIABLES,
                                  CONSTANT_BOUNDARY_VARIABLES, VARYING_BOUNDARY_VARIABLES,BOUNDARY_DIR, SURFACE_MEAN, SURFACE_STD, 
                                  UPPER_AIR_MEAN,UPPER_AIR_STD, CALENDAR, TIMEDELTA_HOURS)
    val_set = DatasetFromFolder(DATA_DIR, YEAR_START_VAL, YEAR_END_VAL, "valid", SURFACE_VARIABLES, UPPER_AIR_VARIABLES,
                                CONSTANT_BOUNDARY_VARIABLES, VARYING_BOUNDARY_VARIABLES, BOUNDARY_DIR, SURFACE_MEAN, SURFACE_STD, 
                                UPPER_AIR_MEAN,UPPER_AIR_STD, CALENDAR, TIMEDELTA_HOURS)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    constant_boundary_data = train_set.constant_boundary_data.unsqueeze(0) * torch.ones(BATCH_SIZE, 1, 1, 1)
    if torch.cuda.is_available():
        constant_boundary_data = constant_boundary_data.cuda()

    lat, lon = train_set.get_lat_lon()

    PanguPlasim = PanguPlasimModulus(horizontal_resolution = (65, 128),
                              num_levels = 10, num_atmo_vars = len(UPPER_AIR_VARIABLES),
                              num_surface_vars = len(SURFACE_VARIABLES),
                              num_boundary_vars = len(CONSTANT_BOUNDARY_VARIABLES) + len(VARYING_BOUNDARY_VARIABLES),
                              patch_size = (2,2,2))
    print("# parameters: ", sum(param.numel() for param in PanguPlasim.parameters()))

    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    if torch.cuda.is_available():
        PanguPlasim.cuda()
        surface_criterion.cuda()
        upper_air_criterion.cuda()

    surface_invTrans = train_set.surface_inv_transform
    upper_air_invTrans = train_set.upper_air_inv_transform

    optimizer = torch.optim.Adam(PanguPlasim.parameters(), lr=5e-4, weight_decay=3e-6)

    results = {'loss': [], 'surface_mse': [], 'upper_air_mse': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}

        PanguPlasim.train()
        for input_surface, input_upper_air, target_surface, target_upper_air, varying_boundary_data in train_bar:
            batch_size = input_surface.size(0)
            if torch.cuda.is_available():
                input_surface = input_surface.cuda()
                input_upper_air = input_upper_air.cuda()
                target_surface = target_surface.cuda()
                target_upper_air = target_upper_air.cuda()
                varying_boundary_data = varying_boundary_data.cuda()

            output_surface, output_upper_air = PanguPlasim(input_surface, constant_boundary_data, varying_boundary_data, input_upper_air)

            optimizer.zero_grad()
            surface_loss = surface_criterion(output_surface, target_surface)
            upper_air_loss = upper_air_criterion(output_upper_air, target_upper_air)
            loss = upper_air_loss + surface_loss * 0.25
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            running_results["batch_sizes"] += batch_size

            train_bar.set_description(desc="[%d/%d] Loss: %.4f" % (epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]))

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {"batch_sizes": 0, "surface_mse": 0, "upper_air_mse": 0}
            for val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, val_varying_boundary_data, times in val_bar:
                batch_size = val_input_surface.size(0)
                if torch.cuda.is_available():
                    val_input_surface = val_input_surface.cuda()
                    val_input_upper_air = val_input_upper_air.cuda()
                    val_target_surface = val_target_surface.cuda()
                    val_target_upper_air = val_target_upper_air.cuda()
                    val_varying_boundary_data = val_varying_boundary_data.cuda()

                val_output_surface, val_output_upper_air = PanguPlasim(val_input_surface, constant_boundary_data, val_varying_boundary_data, val_input_upper_air)

                val_output_surface = val_output_surface.squeeze(0)
                val_output_upper_air = val_output_upper_air.squeeze(0)

                val_target_surface = val_target_surface.squeeze(0)
                val_target_upper_air = val_target_upper_air.squeeze(0)

                valing_results["batch_sizes"] += batch_size

                surface_mse = ((val_output_surface - val_target_surface) ** 2).data.mean().cpu().item()
                upper_air_mse = ((val_output_upper_air - val_target_upper_air) ** 2).data.mean().cpu().item()

                valing_results["surface_mse"] += surface_mse * batch_size
                valing_results["upper_air_mse"] += upper_air_mse * batch_size

                val_bar.set_description(desc="[validating] Surface MSE: %.4f Upper Air MSE: %.4f" % (valing_results["surface_mse"] / valing_results["batch_sizes"], valing_results["upper_air_mse"] / valing_results["batch_sizes"]))

        os.makedirs("epochs", exist_ok=True)
        torch.save(PanguPlasim.state_dict(), "epochs/pangu_plasim_epoch_%d.pth" % (epoch))

        results["loss"].append(running_results["loss"] / running_results["batch_sizes"])
        results["surface_mse"].append(valing_results["surface_mse"] / valing_results["batch_sizes"])
        results["upper_air_mse"].append(valing_results["upper_air_mse"] / valing_results["batch_sizes"])

        data_frame = pd.DataFrame(
            data=results,
            index=range(1, epoch + 1)
        )
        save_root = "train_logs"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        data_frame.to_csv(os.path.join(save_root, "logs.csv"), index_label="Epoch")

