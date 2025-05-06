import os, subprocess, cftime
import numpy as np
from datetime import timedelta
from utils.YParams import YParams

def submit_long_emulation(init_nc_filepaths, init_datetime, final_datetime,
                          run_iter, dependency_jobID=None, runtime = '12:00:00',
                          use_6h_24h_model = False, num_gpus = 4, output_dir=None):
    init_datetime_str = init_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    final_datetime_str = final_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    init_nc_filestr = '\,'.join(init_nc_filepaths)
    num_cpus = num_gpus * 16
    cmd = ['qsub', '-l' f'select=1:ncpus={num_cpus}:ngpus={num_gpus}', '-v', f'INIT_DATETIME="{init_datetime_str}",FINAL_DATETIME="{final_datetime_str}",INIT_NC_FILEPATHS={init_nc_filestr},RUN_ITER={run_iter}']
    if output_dir:
        cmd[-1] += f',OUTPUT_DIR="{output_dir}"'
    else:
        cmd[-1] += ',OUTPUT_DIR=""'
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
end_year = 111
jobs_per_sim = 1
gpus_per_run = 4
#sim_start = 0
#sim_end = 44
#gpus_per_run = 1
sim_start = 0
sim_end = 4
runtime_per_job = '3:00:00'
use_6h_24h_model = True
#dependency_jobIDs = '9101934:9101936:9101938:9101939:9101940:9101941:9101942:9101943:9101944:9101945:9101946'

reinit_offset = 24
yaml_config   = '/glade/work/awikner/PanguWeather-long/v2.0/config/PANGU_PLASIM_H5_DERECHO_0515_longtest_3.yaml'
init_nc_dir = '/glade/derecho/scratch/awikner/PLASIM/data/panguplasim_no_soil_moisture_1/start_data/'
output_dir = '/glade/derecho/scratch/awikner/PLASIM/data/panguplasim_no_soil_moisture_6h_24h/'
params = YParams(os.path.abspath(yaml_config), "PLASIM")

year_splits = [round(year) for year in np.linspace(start_year, end_year, jobs_per_sim+1)]

init_datetimes = [cftime.DatetimeProlepticGregorian(year, 1, 1, hour=0, has_year_zero = params.has_year_zero)\
    for year in year_splits[:-1]]

if start_year == 1:
    datetime_shift_start = 1
else:
    datetime_shift_start = 0
for i in range(datetime_shift_start, len(init_datetimes)):
    init_datetimes[i] -= timedelta(hours = reinit_offset)
    
final_datetimes = [cftime.DatetimeProlepticGregorian(year, 1, 1, hour=0, has_year_zero = params.has_year_zero)\
    for year in year_splits[1:]]

members_per_sim = gpus_per_run * params.num_ensemble_members
sim_job_start = sim_start // members_per_sim
sim_job_end = sim_end // members_per_sim

for sim_run in range(sim_job_start, sim_job_end):
    if start_year == 1:
        init_nc_filepaths = [os.path.join(init_nc_dir, f'start_{i}.nc') for i in \
            range(sim_run*members_per_sim, (sim_run+1)*members_per_sim)]
    else:
        init_nc_filepaths = [os.path.join(params.output_dir, f'{params.save_basename}_member{i:03}_y{year_splits[0]-1:04}.nc') for i in \
            range(sim_run*members_per_sim, (sim_run+1)*members_per_sim)]
    print(f'Submitting sims {sim_run*members_per_sim}-{(sim_run+1)*members_per_sim}, years {year_splits[0]}-{year_splits[1]}')
    jobID = submit_long_emulation(init_nc_filepaths, init_datetimes[0], final_datetimes[0],
                                  sim_run+1, runtime = runtime_per_job, num_gpus = gpus_per_run,
                                  output_dir = output_dir, use_6h_24h_model=use_6h_24h_model)#, dependency_jobID=dependency_jobIDs)
    #jobID = dependency_jobIDs + f':{jobID}'
    for j in range(1, jobs_per_sim):
        init_nc_filepaths = [os.path.join(params.output_dir, f'{params.save_basename}_member{i:03}_y{year_splits[j]-1:04}.nc') for i in \
            range(sim_run*members_per_sim, (sim_run+1)*members_per_sim)]
        print(f'Submitting sims {sim_run*members_per_sim}-{(sim_run+1)*members_per_sim}, years {year_splits[j]}-{year_splits[j+1]}')
        jobID = submit_long_emulation(init_nc_filepaths, init_datetimes[j], final_datetimes[j],
                                    sim_run+1, dependency_jobID=jobID, runtime = runtime_per_job, num_gpus = gpus_per_run,
                                    output_dir = output_dir, use_6h_24h_model=use_6h_24h_model)
        #jobID = jobID + f':{jobID_out}'
    