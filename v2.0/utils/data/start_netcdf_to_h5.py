import subprocess, os
import numpy as np

total_runs = 32
year_range = list(range(7, 132))
idxs = [round(elem) for elem in np.linspace(0, len(year_range)-1, total_runs + 1)]
data_dir = '/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52'
save_dir = '/scratch/09979/awikner/PLASIM/data/2100_year_sims_rerun/sim52/h5'
for i, (start_idx, end_idx) in enumerate(zip(idxs[:-1], idxs[1:])):
    
    cmd = ['python', 'netcdf-to-h5-new.py', f'--run_idx={i}', f'--start_year={year_range[start_idx]}',
           f'--end_year={year_range[end_idx]}', f'--root_dir={data_dir}', f'--save_dir={save_dir}' , '&']
    subprocess.run(' '.join(cmd), shell=True, stderr=subprocess.DEVNULL)
    print(f'Began processing years {year_range[start_idx]} to {year_range[end_idx]}')
    