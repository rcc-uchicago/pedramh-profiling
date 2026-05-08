import os, subprocess, sys

if __name__ == "__main__":
    args = sys.argv[1:]
    num_gpus = 4
    num_training_runs = 3
    config_type = "SFNO"

    if config_type == "SFNO":
        base_config = '/glade/work/marchakitus/PLASIM/PanguWeather/v2.0/config/SFNO_PLASIM_H5_DERECHO_'
        submit_script = 'derecho_training_sfno.sh'
    else:
        base_config = '/glade/work/marchakitus/PLASIM/PanguWeather/v2.0/config/PANGU_PLASIM_H5_DERECHO_'
        submit_script = 'derecho_training.sh'

    run_nums = args[0].split(',')
    configs = [base_config + run_num + '.yaml' for run_num in run_nums]

    for yaml_config, run_num in zip(configs, run_nums):
        env = os.environ.copy()
        env.update({
            "RUN_NUM": run_num,
            "YAML_CONFIG": yaml_config,
            "DEBUG": "1",
            "NUM_GPUS": str(num_gpus),
            "NUM_TRAINING_RUNS": str(num_training_runs),
        })

        submit_cmd = ["bash", submit_script]
        print("Submitting:", submit_cmd, env)

        output = subprocess.check_output(submit_cmd, env=env).decode()
        print(output)
