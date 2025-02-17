import os, subprocess

num_gpus = 8 # Either 4 or 8
num_training_runs = 3 # Total number of 12 hour trainings runs expected to reach the final epoch

if num_gpus == 4:
    base_config = '/net/scratch2/awikner/PanguWeather/v2.0/config/PANGU_PLASIM_H5_DSI_4.yaml'
    submit_script = 'dsi_training.sh'
elif num_gpus == 8:
    base_config = '/net/scratch2/awikner/PanguWeather/v2.0/config/PANGU_PLASIM_H5_DSI.yaml'
    submit_script = 'dsi_training_8.sh'
    
run_num = '0502' # Number for run that will be logged to wandb
config = os.path.join(os.getcwd(), 'config', os.path.basename(base_config).split('.')[0] + f'_{run_num}.yaml')
os.makedirs(os.path.dirname(config), exist_ok=True)

my_scratch = '/net/scratch2/awikner' # CHANGE THIS TO YOUR SCRATCH DIRECTORY

replace_fields = {'name': f'Pangu-PLASIM-{run_num}',
                  'data_dir': os.path.join(my_scratch, 'PLASIM/data/sigma_data'),
                  'land_variables': "['mrso']"}

with open(base_config, "r") as src, open(config, "w") as dest:
    for line in src:
        for field, value in replace_fields.items():
            if f'{field}:' in line:
                line = f'  {field}: {value} \n'
        dest.write(line)

submit_cmd = ['sbatch', submit_script, run_num, config]
print(submit_cmd)
jobID = str(int(subprocess.check_output(submit_cmd).decode().strip().split()[-1]))
for i in range(1, num_training_runs):
    submit_cmd = ['sbatch', '-d', f'after:{jobID}', submit_script, run_num, config, jobID]
    print(submit_cmd)
    jobID      = str(int(subprocess.check_output(submit_cmd).decode().strip().split()[-1]))