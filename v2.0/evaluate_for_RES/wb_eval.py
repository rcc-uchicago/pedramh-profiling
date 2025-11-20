from utils.YParams import YParams
import xarray as xr
import numpy as np
import inspect
from weatherbench2 import config
import weatherbench2.metrics as metrics_module
from weatherbench2 import thresholds
from weatherbench2.evaluation import evaluate_with_beam
from weatherbench2.regions import SliceRegion
import dask
import json
import argparse

def make_latitude_increasing(dataset: xr.Dataset) -> xr.Dataset:
    """Make sure latitude values are increasing. Flip dataset if necessary."""
    lat = dataset.latitude.values
    if (np.diff(lat) < 0).all():
        reverse_lat = lat[::-1]
        dataset = dataset.sel(latitude=reverse_lat)
    return dataset

def main(params):
    # Load climatology file
    climatology = xr.open_zarr(params.paths["climatology_path"])

    # Flip latitude ordering in climatology to ensure ACC calculation works
    climatology = make_latitude_increasing(climatology)

    # Load the JSON
    with open(params.region_file, "r") as f:
        region_json = json.load(f)

    # Create a full dictionary of all regions
    all_regions = {}

    for name, coords in region_json.items():
        xvals = coords.get("xvals", [])
        yvals = coords.get("yvals", [])
        
        # Use slice(None) to indicate "take all values"
        lon_slice = slice(*xvals) if xvals else slice(None)
        lat_slice = slice(*yvals) if yvals else slice(None)
        
        all_regions[name] = SliceRegion(lat_slice=lat_slice, lon_slice=lon_slice)

    # Select desired regions
    regions_dict = {name: all_regions[name] for name in params.regions if name in all_regions}
    print("Configured region(s).")


    # Automatically gather all classes from metrics_module
    metric_classes = {
        name: cls for name, cls in inspect.getmembers(metrics_module, inspect.isclass)
        if cls.__module__ == metrics_module.__name__
    }
    # Instantiate metrics
    selected_metrics = {}
    for metric_name in params.metrics:
        cls = metric_classes[metric_name]

        # Check if 'climatology' is a parameter of the class __init__
        init_params = inspect.signature(cls.__init__).parameters
        if "climatology" in init_params:
            selected_metrics[metric_name] = cls(climatology=climatology)

        ### TO-DO: make this work for the selected threshold(s) automatically
        elif "thresholds" in init_params:
            climatology_mean_std = xr.open_zarr("/glade/derecho/scratch/aasche/PLASIM/data/ensemble_test/paper/climatology_mean_std.zarr")
            climatology_mean_std = make_latitude_increasing(climatology_mean_std)
            threshold_object = thresholds.GaussianQuantileThreshold(
                climatology=climatology_mean_std,
                quantile=params.quantile
            )
            selected_metrics[metric_name] = cls(thresholds=[threshold_object])
        else:
            selected_metrics[metric_name] = cls()

    print("Configured metrics.")
    print(selected_metrics)


    # Configurations for wb2
    paths = config.Paths(
        forecast=params.paths["forecast_path"],
        obs=params.paths["obs_path"],
        output_dir=params.paths["output_dir"], # Directory to save evaluation results
    )

    selection = config.Selection(
        variables=params.variables,
        levels=params.levels,
        time_slice=slice(params.time_start, params.time_end),
    )

    data_config = config.Data(selection=selection, paths=paths)

    eval_configs = {
    params.output_name: config.Eval(
        metrics=selected_metrics,
        regions=regions_dict,
        evaluate_persistence=params.persistence
    )
    }

    print("Starting evalutation procedure...")
    with dask.config.set(array__slicing__split_large_chunks=True):
        input_chunks = {
                "latitude": 32,
                "longitude": 32,
            }
        num_threads = 16
        evaluate_with_beam(data_config, eval_configs, input_chunks=input_chunks, runner="DirectRunner", num_threads=num_threads, skipna=True)
        print("Evalutation complete; file should be saved.")

if __name__ == "__main__":
    print("===> Script started!")
    # Parse arguments and load YAML config
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="/glade/u/home/aasche/PanguWeather/v2.0/config/ENSEMBLE_EVAL_DERECHO.yaml", type=str)
    parser.add_argument("--config", default="base", type=str)

    ####### for UCAR
    parser.add_argument("--local-rank", type=int)
    #######

    args = parser.parse_args()

    params = YParams(args.yaml_config, args.config, print_params=True)
    main(params)