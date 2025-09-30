#!/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/python -u
#SBATCH -p spr
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=112
#SBATCH -A TG-ATM170020
#SBATCH -J climatology
#SBATCH -o /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.out
#SBATCH -e /work2/09979/awikner/stampede3/PLASIM/log_files/%x-%j.err
#SBATCH --mail-user=awikner@uchicago.edu
#SBATCH --mail-type=ALL
import sys, glob, os, subprocess, json
from multiprocessing import Pool
from tqdm import tqdm
from natsort import natsorted
import numpy as np

def get_latlon_str(region_xvals, region_yvals, latlon_eps = 1e-2):
    selstr = ''
    if len(region_xvals) == 0:
        selstr += '0,360,'
    else:
        selstr += f'{region_xvals[0] - latlon_eps:.2f},{region_xvals[1] + latlon_eps:.2f},'
    if len(region_yvals) == 0:
        selstr += '-90,90'
    else:
        selstr += f'{region_yvals[0] - latlon_eps:.2f},{region_yvals[1] + latlon_eps:.2f}'
    return selstr

def get_sel_region(region_xvals, region_yvals):
    sel_region = []
    if all([not isinstance(i, list) for i in region_xvals]) and all([not isinstance(i, list) for i in region_xvals]):
        sel_region.append(get_latlon_str(region_xvals, region_yvals))
    else:
        for region_xvals_i, region_yvals_i in zip(region_xvals, region_yvals):
            sel_region.append(get_latlon_str(region_xvals_i, region_yvals_i))
    return sel_region

def compute_sum_climatology(files, overwrite = False, compute_mean = True, compute_std = False):
    cdo = '/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/cdo'
    cdo_cmd = [cdo, '-O', '-P', '112']
    savedir = os.path.dirname(files[0])
    file_cmd = [['-del29feb', file] for file in files]
    file_cmd = [item for sublist in file_cmd for item in sublist]
    if compute_mean:
        mean_clim_file = os.path.join(savedir, 'mean_daysum_climatology.nc')
        mean_clim_cmd = cdo_cmd + ['-ydaymean', '-daysum', '-mergetime'] + file_cmd + [mean_clim_file]
        if not overwrite and os.path.isfile(mean_clim_file):
            print(f'Found existing mean sum climatology for {savedir}, skipping...')
        else:
            subprocess.run(mean_clim_cmd)
            print(f'Processed mean sum climatology for {savedir}')
    if compute_std:
        std_clim_file = os.path.join(savedir, 'std_daysum_climatology.nc')
        mean_clim_cmd = cdo_cmd + ['-ydaystd', '-daysum', '-mergetime'] + file_cmd + [std_clim_file]
        if not overwrite and os.path.isfile(std_clim_file):
            print(f'Found existing std sum climatology for {savedir}, skipping...')
        else:
            subprocess.run(mean_clim_cmd)
            print(f'Processed std sum climatology for {savedir}')

def compute_mean_climatology(files, plev_str, select_plev, id_str = '', overwrite = False,
                             compute_mean = True, compute_std = False, mean_days = None):
    cdo = '/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/cdo'
    cdo_cmd = [cdo, '-O', '-P', '112']
    savedir = os.path.dirname(files[0])
    if select_plev:
        file_cmd = []
        for file in files:
            file_cmd.extend([f'-sellevel,{plev_str}', file])
    else:
        file_cmd = files
    if compute_mean:
        mean_clim_file = os.path.join(savedir, f'mean_daymean{id_str}_climatology.nc')
        if not overwrite and os.path.exists(mean_clim_file):
            print(f'Found existing mean climatology for {savedir}, skipping...')
        else:
            if id_str in ['_begin', '_end']:
                mean_clim_cmd = cdo_cmd + ['-ydrunmean,15', '-del29feb', '-mergetime'] + file_cmd + [mean_clim_file]
            else:
                if mean_days:
                    mean_clim_cmd = cdo_cmd + [f'-ydrunmean,{mean_days}', '-del29feb', '-mergetime'] + file_cmd + [mean_clim_file]
                else:
                    mean_clim_cmd = cdo_cmd + ['-ydaymean', '-del29feb', '-mergetime'] + file_cmd + [mean_clim_file]
            subprocess.run(mean_clim_cmd)
            print(f'Processed mean climatology for {savedir}')
    if compute_std:
        std_clim_file = os.path.join(savedir, f'std_daymean{id_str}_climatology.nc')
        if not overwrite and os.path.exists(std_clim_file):
            print(f'Found existing std climatology for {savedir}, skipping...')
        else:
            if id_str in ['_begin', '_end']:
                mean_clim_cmd = cdo_cmd + ['-ydrunstd,15', '-daymean', '-del29feb', '-mergetime'] + file_cmd + [std_clim_file]
            else:
                if mean_days:
                    mean_clim_cmd = cdo_cmd + ['-ydaystd',  f'-runmean,{mean_days}', '-daymean', '-del29feb', '-mergetime'] + file_cmd + [std_clim_file]
                else:
                    mean_clim_cmd = cdo_cmd + ['-ydaystd', '-daymean', '-del29feb', '-mergetime'] + file_cmd + [std_clim_file]
            subprocess.run(mean_clim_cmd)
            print(f'Processed std climatology for {savedir}')

def compute_max_climatology(files, plev_str, variable, sel_region, region, overwrite = False,
                            compute_mean = True, compute_std = False):
    cdo = '/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/cdo'
    cdo_cmd = [cdo, '-O', '-P', '112']
    savedir = os.path.dirname(files[0])
    sel_file_cmd = []
    for file in files:
        if len(sel_region) > 1:
            sel_file_cmd += ['-merge']
        for sel_region_i in sel_region:
            if plev_str:
                sel_file_cmd += [f'-sellonlatbox,{sel_region_i}', f'-sellevel,{plev_str}', f'-selname,{variable}', file]
            else:
                sel_file_cmd += [f'-sellonlatbox,{sel_region_i}', f'-selname,{variable}', file]
    if compute_mean:
        mean_clim_file = os.path.join(savedir, f'mean_fldmax_daymax_{region}_climatology.nc')
        if not overwrite and os.path.isfile(mean_clim_file):
            print(f'Found mean max climatology for {savedir}, skipping...')
        else:
            mean_clim_cmd = cdo_cmd + ['-ydaymean', '-fldmax', '-daymax', '-del29feb', '-mergetime'] + sel_file_cmd + [mean_clim_file]
            subprocess.run(mean_clim_cmd)
            print(f'Processed mean max climatology for {savedir}')
    if compute_std:
        std_clim_file = os.path.join(savedir, f'std_fldmax_daymax_{region}_climatology.nc')
        if not overwrite and os.path.isfile(std_clim_file):
            print(f'Found std max climatology for {savedir}, skipping...')
        else:
            mean_clim_cmd = cdo_cmd + ['-ydaystd', '-fldmax', '-daymax', '-del29feb', '-mergetime'] + sel_file_cmd + [std_clim_file]
            subprocess.run(mean_clim_cmd)
            print(f'Processed std max climatology for {savedir}')
        
def compute_min_climatology(files, plev_str, variable, sel_region, region, overwrite = False,
                            compute_mean = True, compute_std = False):
    cdo = '/work2/09979/awikner/stampede3/miniconda3/envs/emul4climate_nompi/bin/cdo'
    cdo_cmd = [cdo, '-O', '-P', '112']
    savedir = os.path.dirname(files[0])
    sel_file_cmd = []
    for file in files:
        if len(sel_region) > 1:
            sel_file_cmd += ['-merge']
        for sel_region_i in sel_region:
            if plev_str:
                sel_file_cmd += [f'-sellonlatbox,{sel_region_i}', f'-sellevel,{plev_str}', f'-selname,{variable}', file]
            else:
                sel_file_cmd += [f'-sellonlatbox,{sel_region_i}', f'-selname,{variable}', file]
    if compute_mean:
        mean_clim_file = os.path.join(savedir, f'mean_fldmin_daymin_{region}_climatology.nc')
        if not overwrite and os.path.isfile(mean_clim_file):
            print(f'Found mean max climatology for {savedir}, skipping...')
        else:
            mean_clim_cmd = cdo_cmd + ['-ydaymean', '-fldmin', '-daymin', '-del29feb', '-mergetime'] + sel_file_cmd + [mean_clim_file]
            subprocess.run(mean_clim_cmd)
            print(f'Processed mean max climatology for {savedir}')
    if compute_std:
        std_clim_file = os.path.join(savedir, f'std_fldmin_daymin_{region}_climatology.nc')
        if not overwrite and os.path.isfile(std_clim_file):
            print(f'Found std max climatology for {savedir}, skipping...')
        else:
            mean_clim_cmd = cdo_cmd + ['-ydaystd', '-fldmin', '-daymin', '-del29feb', '-mergetime'] + sel_file_cmd + [std_clim_file]
            subprocess.run(mean_clim_cmd)
            print(f'Processed std max climatology for {savedir}')

    
if __name__ == "__main__":
    args = sys.argv[1:]
    data_dir_base = args[0]
    print(f'Data dir base: {data_dir_base}')
    variables = args[1].split(',')
    print(f'Variables: {args[1]}')
    plev_units = args[2]
    print(f'Plev units: {plev_units}')
    if len(args) > 3:
        get_data_dir = True
        sims = [int(i) for i in args[3].split(',')]
        print(f"Sims: {','.join([str(sim) for sim in sims])}")
        print(f"Sims: {','.join([str(sim) for sim in sims])}")
    else:
        get_data_dir = False
        sims = [0]
    if len(args) > 4:
        mean_days_list = [int(i) for i in args[4].split(',')]
        id_strs = [f'_{mean_days}day' for mean_days in mean_days_list]
    else:
        mean_days_list = [None]
        id_strs = ['']
    region_file = '/work2/09979/awikner/stampede3/Emul4Climate/regions.json'
    with open(region_file, 'r') as f:
        region_dir = json.load(f)
    if plev_units == 'Pa':
        plev_str = '20000,25000,50000,85000,100000'
    elif plev_units == 'hPa':
        plev_str = '200,250,500,850,1000'
    else:
        raise ValueError('plev units not recognized')
    compute_mean = True
    compute_std = False
    compute_daymean = True
    compute_max_min = False
    spin_up = 5
    atm_vars = ['ta', 'ua', 'va', 'hus', 'zg']
    for sim in sims:
        if get_data_dir:
            data_dir = os.path.join(data_dir_base, f'sim{sim}')
        else:
            data_dir = data_dir_base
        for variable in variables:
            all_files = natsorted(glob.glob(os.path.join(data_dir, variable, '*_gaussian.nc'), recursive=True))[spin_up:]
            if variable == 'pr_6h':
                compute_sum_climatology(all_files, overwrite = True, compute_mean=compute_mean,
                                        compute_std = compute_std)
            elif compute_daymean:
                for mean_days, id_str in zip(mean_days_list, id_strs):
                    compute_mean_climatology(all_files, plev_str, variable in atm_vars, overwrite = True, 
                                            compute_mean=compute_mean, compute_std = compute_std,
                                            mean_days = mean_days, id_str=id_str)
            #elif variable in atm_vars:
            #   file_years = [int(os.path.basename(file).split('_')[0]) for file in all_files]
            #   file_sizes = [os.path.getsize(file) for file in all_files]
            #   mean_file_size = np.mean(np.array(file_sizes))
            #   file_years_small = [year for year, size in zip(file_years, file_sizes) if size <= mean_file_size]
            #   files = [file for year, file in zip(file_years, all_files) if year in file_years_small]
            #   files_begin = [file for year, file in zip(file_years, all_files) if year < file_years_small[0]]
            #   files_end = [file for year, file in zip(file_years, all_files) if year > file_years_small[-1]]
            #   compute_mean_climatology(files, None, False, overwrite = True)
            #   compute_mean_climatology(files_begin, None, False, '_begin', overwrite = True)
            #   compute_mean_climatology(files_end, None, False, '_end', overwrite = True)
        if 'tas' in variables and compute_max_min:
            regions_tas = ['France', 'PNW', 'Chicago']
            files = natsorted(glob.glob(os.path.join(data_dir, 'tas', '*_gaussian.nc'), recursive=True))[spin_up:]
            for region in regions_tas:
                sel_region = get_sel_region(region_dir[region]['xvals'], region_dir[region]['yvals'])
                #compute_max_climatology(files, None, 'tas', sel_region, region, overwrite = True, 
                #                        compute_mean=compute_mean, compute_std = compute_std)
                compute_min_climatology(files, None, 'tas', sel_region, region, overwrite = True, 
                                        compute_mean=compute_mean, compute_std = compute_std)
        if 'zg' in variables and compute_max_min:
            regions_zg = ['France_5x5', 'PNW_5x5', 'Chicago_5x5']
            files = natsorted(glob.glob(os.path.join(data_dir, 'zg', '*_gaussian.nc'), recursive=True))[spin_up:]
            for region in regions_zg:
                sel_region = get_sel_region(region_dir[region]['xvals'], region_dir[region]['yvals'])
                compute_max_climatology(files, plev_str, 'zg', sel_region, region, overwrite = True, 
                                        compute_mean=compute_mean, compute_std = compute_std)
