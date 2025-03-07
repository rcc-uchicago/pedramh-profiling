import os, subprocess

jobs_per_training = 1 # Total number of 12 hour trainings runs expected to reach the final epoch
debug = True

start_run_num = 517
num_runs = 2
run_nums = [f'{run_num:04}' for run_num in range(start_run_num, start_run_num+num_runs)]
#run_nums = ['0515', '0516'] # Number for run that will be logged to wandb

my_scratch = '/pscratch/sd/a/awikner' # CHANGE THIS TO YOUR SCRATCH DIRECTORY

replace_fields_dicts = [
                  {'name': f'Pangu-PLASIM-{run_nums[0]}',
                  'data_dir': os.path.join(my_scratch, 'PLASIM/data/h5/sigma_data'),
                  'use_sigma_levels': True,
                  'surface_mean': 'data_12-111_sigma_mean.nc',
                  'surface_std': 'data_12-111_sigma_std.nc',
                  'surface_ff_std': 'data_12-111_sigma_std.nc', 
                  'upper_air_mean': 'data_12-111_sigma_mean.nc', 
                  'upper_air_std': 'data_12-111_sigma_std.nc',
                  'upper_air_ff_std': 'data_12-111_sigma_std.nc', 
                  'boundary_mean': 'data_12-111_sigma_mean.nc', 
                  'boundary_std': 'data_12-111_sigma_std.nc',
                  'diagnostic_mean': 'data_12-111_sigma_mean.nc',
                  'diagnostic_std': 'data_12-111_sigma_std.nc',
                  'levels': '[20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000]',
                  'checkpointing': '2',
                  'batch_size': 64,
                  'num_inferences': 128,
                  'timedelta_hours': 6,
                  'diagnostic_variables': '["pr_6h"]',
                  'epsilon_factor': 0.01,
                  'loss': 'l1',
                  'diagnostic_acc': False,
                  'diagnostic_gif': True,
                  'diagnostic_gif_var_dict': '{"zg": [50000], "ua": [0.03830000013113022, 0.21085000783205032], "ta":[0.8233500719070435], "tas": []}',
                  'diagnostic_spectra': False,
                  'forecast_lead_times': '[1, 12, 20, 40, 60]',
                  'lev': 'lev', 
                  'climatology_file': 'mean_daymean_climatology_sigma.nc'},
                  {'name': f'Pangu-PLASIM-{run_nums[1]}',
                  'data_dir': os.path.join(my_scratch, 'PLASIM/data/h5/plev_data'),
                  'use_sigma_levels': False,
                  'surface_mean': 'data_12-111_mean.nc',
                  'surface_std': 'data_12-111_std.nc',
                  'surface_ff_std': 'data_12-111_std.nc',
                  'upper_air_mean': 'data_12-111_mean.nc',
                  'upper_air_std': 'data_12-111_std.nc',
                  'upper_air_ff_std': 'data_12-111_std.nc',
                  'boundary_mean': 'data_12-111_mean.nc',
                  'boundary_std': 'data_12-111_std.nc',
                  'diagnostic_mean': 'data_12-111_mean.nc',
                  'diagnostic_std': 'data_12-111_std.nc',
                  'climatology_file': 'mean_daily_climatology_time_pl.nc',
                  'num_levels': 13,
                  'levels': '[5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000]',
                  'checkpointing': '2',
                  'batch_size': 64,
                  'num_inferences': 128,
                  'long_validation': True,
                  'diagnostic_variables': '["pr_6h"]',
                  'epsilon_factor': 0.01,
                  'loss': 'l1',
                  'diagnostic_acc': True,
                  'diagnostic_gif': True,
                  'diagnostic_gif_var_dict': '{"zg": [50000], "ua": [5000, 25000], "ta":[85000], "tas": []}',
                  'diagnostic_spectra': True,
                  'forecast_lead_times': '[1, 12, 20, 40, 60]'}]

#run_nums = [run_nums[1]]
#replace_fields_dicts = [replace_fields_dicts[1]]
                  

runtime = '48:00:00'
base_config = '/pscratch/sd/a/awikner/PanguWeather/v2.0/config/PANGU_PLASIM_H5_PERLMUTTER.yaml'
submit_script = 'perlmutter_training.sh'
    
for run_num, replace_fields in zip(run_nums, replace_fields_dicts):
    config = os.path.join(os.getcwd(), 'config', os.path.basename(base_config).split('.')[0] + f'_{run_num}.yaml')
    os.makedirs(os.path.dirname(config), exist_ok=True)

    with open(base_config, "r") as src, open(config, "w") as dest:
        for line in src:
            for field, value in replace_fields.items():
                if f'  {field}:' in line:
                    line = f'  {field}: {value} \n'
            dest.write(line)

    if debug:
        submit_cmd = ['./'+submit_script, run_num, config, str(int(debug))]
        print(' '.join(submit_cmd))
        #subprocess.run(submit_cmd)
    else:
        submit_cmd = ['sbatch', '-t', runtime, submit_script, run_num, config, str(int(debug))]
        print(submit_cmd)
        jobID = str(int(subprocess.check_output(submit_cmd).decode().strip().split()[-1]))
        for i in range(1, jobs_per_training):
            submit_cmd = ['sbatch', '-t', runtime, '-d', f'after:{jobID}', submit_script, run_num, config, str(int(debug)), jobID]
            print(submit_cmd)
            jobID      = str(int(subprocess.check_output(submit_cmd).decode().strip().split()[-1]))