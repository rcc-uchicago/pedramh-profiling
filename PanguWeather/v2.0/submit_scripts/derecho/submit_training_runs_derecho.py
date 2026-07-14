import os, subprocess, sys

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    num_gpus = 4 # Either 4 or 8
    num_training_runs = 2 # Total number of 12 hour trainings runs expected to reach the final epoch
    config = "SFNO"

    if config == "SFNO":
        base_config = '/glade/work/marchakitus/PLASIM/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_'
        submit_script = 'derecho_training_sfno.sh'
    else:
        base_config = '/glade/work/marchakitus/PLASIM/PanguWeather/v2.0/config/PANGU_PLASIM_H5_DERECHO_'
        submit_script = 'derecho_training.sh'
        
    run_nums = args[0].split(',') # Number for run that will be logged to wandb
    configs = [base_config + run_num + '.yaml' for run_num in run_nums]

    for config, run_num in zip(configs, run_nums):
        submit_cmd = ['qsub', '-v', f'RUN_NUM={run_num},YAML_CONFIG={config},DEBUG=0', submit_script]
        print(submit_cmd)
        output = subprocess.check_output(submit_cmd).decode()
        print(output)
        jobID = output.split('.')[0]
        for i in range(1, num_training_runs):
            submit_cmd = ['qsub', '-W', f'depend=afterany:{jobID}', '-v', f'RUN_NUM={run_num},YAML_CONFIG={config},DEBUG=0,JOBID={jobID}', submit_script]
            print(submit_cmd)
            output = subprocess.check_output(submit_cmd).decode()
            print(output)
            jobID = output.split('.')[0]