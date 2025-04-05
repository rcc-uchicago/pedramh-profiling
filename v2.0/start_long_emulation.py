import os, subprocess, cftime
import numpy as np
from datetime import timedelta
from utils.YParams import YParams

def submit_long_emulation(init_nc_filepaths, init_datetime, final_datetime,
                          run_iter, dependency_jobID=None, runtime = '12:00:00',
                          use_6h_24h_model = False):
    init_datetime_str = init_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    final_datetime_str = final_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    init_nc_filestr = '\,'.join(init_nc_filepaths)
    cmd = ['qsub', '-v', f'INIT_DATETIME="{init_datetime_str}",FINAL_DATETIME="{final_datetime_str}",INIT_NC_FILEPATHS={init_nc_filestr},RUN_ITER={run_iter}']
    if dependency_jobID:
        cmd += ['-W', f'depend=afterok:{dependency_jobID}']
    if use_6h_24h_model:
        cmd += ['-l', f'walltime={runtime}', 'derecho_long_inference.sh']
    else:
        cmd += ['-l', f'walltime={runtime}', 'derecho_long_inference_6h.sh']
    print(' '.join(cmd))
    output = subprocess.check_output(cmd).decode()
    print(output)
    jobID = output.split('.')[0]
    return jobID
    

start_year = 1
end_year = 2151
jobs_per_sim = 2
gpus_per_run = 4
total_sims = 48
runtime_per_job = '12:00:00'

reinit_offset = 24
yaml_config   = '/glade/work/awikner/PanguWeather-long/v2.0/config/PANGU_PLASIM_H5_DERECHO_0515_longtest_2.yaml'
init_nc_dir = '/glade/derecho/scratch/awikner/PLASIM/data/panguplasim_no_soil_moisture_1/start_data/'
params = YParams(os.path.abspath(yaml_config), "PLASIM")

year_splits = [round(year) for year in np.linspace(start_year, end_year, jobs_per_sim+1)]

init_datetimes = [cftime.DatetimeProlepticGregorian(year, 1, 1, hour=0, has_year_zero = params.has_year_zero)\
    for year in year_splits[:-1]]

for i in range(1, len(init_datetimes)):
    init_datetimes[i] -= timedelta(hours = reinit_offset)
    
final_datetimes = [cftime.DatetimeProlepticGregorian(year, 1, 1, hour=0, has_year_zero = params.has_year_zero)\
    for year in year_splits[1:]]

members_per_sim = gpus_per_run * params.num_ensemble_members
num_sim_jobs = total_sims // members_per_sim

for sim_run in range(num_sim_jobs):
    init_nc_filepaths = [os.path.join(init_nc_dir, f'start_{i}.nc') for i in \
        range(sim_run*members_per_sim, (sim_run+1)*members_per_sim)]
    print(f'Submitting sims {sim_run*members_per_sim}-{(sim_run+1)*members_per_sim}, years {year_splits[0]}-{year_splits[1]}')
    jobID = submit_long_emulation(init_nc_filepaths, init_datetimes[0], final_datetimes[0],
                                  sim_run+1, runtime = runtime_per_job)
    for j in range(1, jobs_per_sim):
        init_nc_filepaths = [os.path.join(params.output_dir, f'{params.save_basename}_member{i:03}_y{year_splits[j]-1:04}.nc') for i in \
            range(sim_run*members_per_sim, (sim_run+1)*members_per_sim)]
        print(f'Submitting sims {sim_run*members_per_sim}-{(sim_run+1)*members_per_sim}, years {year_splits[j]}-{year_splits[j+1]}')
        jobID = submit_long_emulation(init_nc_filepaths, init_datetimes[j], final_datetimes[j],
                                    sim_run+1, dependency_jobID=jobID, runtime = runtime_per_job)
    