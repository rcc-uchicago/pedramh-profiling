#!/usr/bin/env python3
"""
Script to begin a curriculum learning fine-tuning run.

This script:
1. Creates a new config file from an existing one with curriculum learning settings
2. Copies the checkpoint from the starting epoch
3. Submits a training run

Usage:
    python start_curriculum_learning.py \
        --config config/PANGU_PLASIM_H5_DERECHO_0514.yaml \
        --events_json events/Chicago_heatwaves_test.json \
        --curriculum_fraction 0.5 \
        --start_epoch 50 \
        --num_epochs 20 \
        --learning_rate 0.00005 \
        [--submit_script submit_scripts/derecho/derecho_training.sh]
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cftime
import yaml
from ruamel.yaml import YAML as RuamelYAML


def extract_run_num_from_config(config_path):
    """Extract the 4-digit run_num from config filename."""
    filename = os.path.basename(config_path)
    # Look for pattern like _0514.yaml or _0514_ at the end
    match = re.search(r'_(\d{4})\.yaml$', filename)
    if match:
        return match.group(1)
    # Try to find any 4-digit number in the filename
    match = re.search(r'(\d{4})', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract run_num from config filename: {config_path}")


def generate_new_run_num(config_dir, extra_exclude=None):
    """Generate a new unique 4-digit run_num based on existing configs.

    Args:
        config_dir: Directory containing YAML config files.
        extra_exclude: Optional set of integer run_nums to exclude in addition
            to those found on disk.  Pass a shared mutable set across calls to
            reserve numbers in-session without requiring files to be written
            (e.g. during a dry run or a grid search).
    """
    # Get all config files in the directory and finetune/ subdirectory
    config_files = list(Path(config_dir).glob('*.yaml'))
    finetune_subdir = Path(config_dir) / 'finetune'
    if finetune_subdir.is_dir():
        config_files.extend(finetune_subdir.glob('*.yaml'))

    # Extract all existing run_nums
    existing_nums = set()
    for config_file in config_files:
        try:
            run_num = extract_run_num_from_config(str(config_file))
            existing_nums.add(int(run_num))
        except ValueError:
            continue

    if extra_exclude:
        existing_nums.update(extra_exclude)

    # Find the next available number starting from current date
    # Use last two digits of year + month + day as base, or start from 1000
    now = datetime.now()
    base_num = int(f"{now.year % 100:02d}{now.month:02d}{now.day:02d}")
    # Clamp to valid 4-digit range before searching; dates in 2100+ produce
    # base_num > 9999 which would bypass the wrap-around inside the while loop.
    if base_num > 9999:
        base_num = 1000

    # If base is already taken, increment until we find an available one
    new_num = base_num
    while new_num in existing_nums:
        new_num += 1
        # Wrap around if we exceed 9999
        if new_num > 9999:
            new_num = 1000

    return f"{new_num:04d}"


def load_json_data(json_path):
    """Load the JSON file containing data directories and date ranges."""
    with open(json_path, 'r') as f:
        return json.load(f)


def datetime_class_from_calendar(calendar):
    """Get the appropriate cftime datetime class for a calendar type."""
    datetime_class_dict = {
        'standard': cftime.DatetimeGregorian,
        'Gregorian:': cftime.DatetimeGregorian,
        'noleap': cftime.DatetimeNoLeap,
        '365_day': cftime.DatetimeNoLeap,
        'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
        'all_leap': cftime.DatetimeAllLeap,
        '366_day': cftime.DatetimeAllLeap,
        '360_day': cftime.Datetime360Day,
        'julian': cftime.DatetimeJulian
    }
    return datetime_class_dict.get(calendar, cftime.DatetimeProlepticGregorian)


def select_events(events_json_data, num_events):
    """
    Select the first num_events+1 entries from events_json_data.
    
    Args:
        events_json_data: Dictionary mapping directories to [start_date, end_date] lists
        num_events: Number of events to select (0 means all events)
    
    Returns:
        Dictionary with selected events
    """
    if num_events == 0:
        return events_json_data
    
    total_events = len(events_json_data)
    if num_events >= total_events:
        return events_json_data
    
    # Select first num_events+1 entries (since num_events means num_events+1 total)
    selected_data = {}
    for i, (data_dir, date_range) in enumerate(events_json_data.items()):
        if i <= num_events:  # 0-indexed, so <= means first num_events+1 entries
            selected_data[data_dir] = date_range
        else:
            break
    
    return selected_data


def adjust_dates_with_offset(events_json_data, start_day_offset, calendar, has_year_zero=False):
    """
    Adjust start dates in events_json_data by adding start_day_offset days.
    The first entry is NOT modified.
    
    Args:
        events_json_data: Dictionary mapping directories to [start_date, end_date] lists
        start_day_offset: Number of days to add to start dates
        calendar: Calendar type (e.g., 'proleptic_gregorian')
        has_year_zero: Whether the calendar has year zero
    
    Returns:
        Dictionary with adjusted start dates (first entry unchanged)
    """
    if start_day_offset == 0:
        return events_json_data
    
    adjusted_data = {}
    datetime_class = datetime_class_from_calendar(calendar)
    
    for i, (data_dir, date_range) in enumerate(events_json_data.items()):
        start_date_str, end_date_str = date_range
        
        # Skip the first entry (index 0) - don't modify its start date
        if i == 0:
            adjusted_data[data_dir] = [start_date_str, end_date_str]
            continue
        
        # Parse the start date - try both formats
        try:
            # Try format with underscore: "0011-01-01_00:00:00"
            start_date = cftime.datetime.strptime(
                start_date_str, 
                "%Y-%m-%d_%H:%M:%S",
                has_year_zero=has_year_zero,
                calendar=calendar
            )
        except ValueError:
            try:
                # Try format with space: "0011-01-01 00:00:00"
                start_date = cftime.datetime.strptime(
                    start_date_str,
                    "%Y-%m-%d %H:%M:%S",
                    has_year_zero=has_year_zero,
                    calendar=calendar
                )
            except ValueError:
                raise ValueError(f"Could not parse date string: {start_date_str}")
        
        # Add the offset in days using the datetime_class approach
        # Create a new datetime using the datetime_class and add timedelta
        adjusted_start_date = datetime_class(
            start_date.year,
            start_date.month,
            start_date.day,
            hour=start_date.hour,
            has_year_zero=has_year_zero
        ) + timedelta(days=start_day_offset)
        
        # Convert back to string format (preserve original format)
        if '_' in start_date_str:
            adjusted_start_str = adjusted_start_date.strftime("%Y-%m-%d_%H:%M:%S")
        else:
            adjusted_start_str = adjusted_start_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Keep end date unchanged
        adjusted_data[data_dir] = [adjusted_start_str, end_date_str]
    
    return adjusted_data


def create_curriculum_config(
    original_config_path,
    new_config_path,
    events_json_data,
    curriculum_fraction,
    learning_rate,
    start_epoch,
    num_epochs,
    old_run_num,
    new_run_num,
    finetune=False,
    config_section='PLASIM',
    ensemble_params=None,
    curriculum_bulk_size=None,
    exp_dir=None,
    load_exp_dir=None,
    use_ema=None,
    ema_decay=None,
    train_date_range=None,
    ensemble_validation_frequency=None,
    balanced_learning=None,
):
    """Create a new config file with curriculum learning settings."""

    # Use ruamel.yaml round-trip loader to preserve YAML anchors and merge keys
    # (<<: *BASE etc.) so the output config stays concise and inheritance works.
    yaml_rt = RuamelYAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.default_flow_style = False
    yaml_rt.width = 4096  # prevent line wrapping of long values

    with open(original_config_path, 'r') as f:
        config_data = yaml_rt.load(f)

    if config_section not in config_data:
        raise ValueError(f"{config_section} section not found in config file: {original_config_path}")

    section_config = config_data[config_section]

    # Collect the set of keys explicitly written by this script so they are
    # never filtered out during the deduplication step below.
    explicitly_set_keys = set()

    if finetune:
        section_config['finetune_run_num'] = new_run_num
        section_config['lr'] = learning_rate
        section_config['finetune_num_epochs'] = num_epochs
        explicitly_set_keys.update({'finetune_run_num', 'lr', 'finetune_num_epochs'})
    else:
        # Update name in base_config anchor if present
        if 'base_config' in config_data and config_data['base_config'] and \
                'name' in config_data['base_config']:
            config_data['base_config']['name'] = f'Pangu-{config_section}-{new_run_num}'
        section_config['name'] = f'Pangu-{config_section}-{new_run_num}'
        section_config['lr'] = learning_rate
        section_config['max_epochs'] = start_epoch + num_epochs
        explicitly_set_keys.update({'name', 'lr', 'max_epochs'})

    # ------------------------------------------------------------------ #
    # Build train_data_sets with the full bulk training data as the      #
    # first entry, followed by the event-specific entries.               #
    # The bulk entry uses data_dir from the base config and date strings #
    # derived from train_year_start, train_year_end, data_timedelta_hours.
    # ------------------------------------------------------------------ #
    with open(original_config_path, 'r') as _f:
        _resolved = yaml.safe_load(_f)
    _sec = _resolved.get(config_section, {})

    _data_dir = _sec.get('data_dir')
    _train_year_start = _sec.get('train_year_start')
    _train_year_end = _sec.get('train_year_end')
    _timedelta_hours = _sec.get('data_timedelta_hours', 6)

    if _data_dir is None:
        raise ValueError(
            f"'data_dir' not found in '{config_section}' section of "
            f"{original_config_path}"
        )
    if _train_year_start is None:
        raise ValueError(
            f"'train_year_start' not found in '{config_section}' section of "
            f"{original_config_path}"
        )
    if _train_year_end is None:
        raise ValueError(
            f"'train_year_end' not found in '{config_section}' section of "
            f"{original_config_path}"
        )

    _bulk_start = f"{int(_train_year_start):04d}-01-01 00:00:00"
    _last_hour = (24 - int(_timedelta_hours)) % 24
    _bulk_end = f"{int(_train_year_end) - 1:04d}-12-31 {_last_hour:02d}:00:00"

    train_data_sets = {_data_dir: [_bulk_start, _bulk_end]}
    train_data_sets.update(events_json_data)

    section_config['scheduler'] = None
    section_config['train_data_sets'] = train_data_sets
    section_config['curriculum_learning'] = True
    section_config['curriculum_learning_fraction'] = curriculum_fraction
    section_config['start_epoch'] = start_epoch
    explicitly_set_keys.update({
        'scheduler', 'train_data_sets', 'curriculum_learning',
        'curriculum_learning_fraction', 'start_epoch',
    })

    # Write ensemble validation parameters if provided.
    if ensemble_params:
        for key, value in ensemble_params.items():
            section_config[key] = value
            explicitly_set_keys.add(key)

    # Write curriculum_bulk_size if provided (used when curriculum_fraction >= 1.0).
    if curriculum_bulk_size is not None:
        section_config['curriculum_bulk_size'] = curriculum_bulk_size
        explicitly_set_keys.add('curriculum_bulk_size')

    # Override exp_dir so train.py saves to the correct absolute path instead
    # of inheriting the default 'results' value from the base config anchor.
    if exp_dir is not None:
        section_config['exp_dir'] = exp_dir
        explicitly_set_keys.add('exp_dir')

    if load_exp_dir is not None:
        section_config['load_exp_dir'] = load_exp_dir
        explicitly_set_keys.add('load_exp_dir')

    if use_ema is not None:
        section_config['use_ema'] = use_ema
        explicitly_set_keys.add('use_ema')

    if ema_decay is not None:
        section_config['ema_decay'] = ema_decay
        explicitly_set_keys.add('ema_decay')

    if train_date_range is not None:
        section_config['train_date_range'] = train_date_range
        explicitly_set_keys.add('train_date_range')

    if ensemble_validation_frequency is not None:
        section_config['ensemble_validation_frequency'] = ensemble_validation_frequency
        explicitly_set_keys.add('ensemble_validation_frequency')

    if balanced_learning:
        section_config['balanced_learning'] = True
        explicitly_set_keys.add('balanced_learning')

    # ------------------------------------------------------------------ #
    # Deduplication: remove entries from section_config that are already  #
    # provided by a lower-priority section (base_config) with the same    #
    # value.  This keeps the output file concise.                         #
    #                                                                      #
    # With anchor-based configs (<<: *BASE) ruamel.yaml only writes the   #
    # explicitly set keys in the section, so the merge key provides the   #
    # base values automatically. For flat configs (no anchors) we compare #
    # against base_config and remove identical entries.                   #
    # ------------------------------------------------------------------ #
    base_config = config_data.get('base_config') or {}

    # Keys that come from YAML merge references inside section_config
    # (these live in section_config.merge, not in the explicit dict, so
    # deleting them via __delitem__ would raise KeyError).
    merged_keys: set = set()
    merge_refs = getattr(section_config, 'merge', None)
    if merge_refs:
        for _priority, merge_map in merge_refs:
            merged_keys.update(merge_map)

    for key in list(section_config.keys()):
        if key in explicitly_set_keys or key in merged_keys:
            continue
        if key in base_config and section_config[key] == base_config[key]:
            del section_config[key]

    with open(new_config_path, 'w') as f:
        yaml_rt.dump(config_data, f)


def copy_checkpoint(old_run_num, new_run_num, start_epoch, exp_dir='results', config_section='PLASIM'):
    """Copy the checkpoint from the starting epoch to the new run directory."""
    old_checkpoint_dir = Path(exp_dir) / config_section / old_run_num / 'checkpoints'
    new_checkpoint_dir = Path(exp_dir) / config_section / new_run_num / 'checkpoints'

    # Filename convention used by train.py: ckpt_epoch_{N}.tar
    checkpoint_filename = f'ckpt_epoch_{start_epoch}.tar'
    old_checkpoint = old_checkpoint_dir / checkpoint_filename

    if not old_checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {old_checkpoint}\n"
            f"Available checkpoints: {list(old_checkpoint_dir.glob('ckpt_epoch_*.tar'))}"
        )

    # Create new checkpoint directory
    new_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy the epoch checkpoint
    new_checkpoint = new_checkpoint_dir / checkpoint_filename
    shutil.copy2(old_checkpoint, new_checkpoint)
    print(f"Copied checkpoint: {old_checkpoint} -> {new_checkpoint}")


def generate_interactive_command(submit_script_path, run_num, config_path, config_section='PLASIM', use_legacy_model=False):
    """
    Generate the command to run the training job interactively on a compute node.
    
    Args:
        submit_script_path: Path to the submission script
        run_num: Run number for the job
        config_path: Path to the YAML config file
        use_legacy_model: Whether to use the legacy model (adds --use_legacy_model flag)
    
    Returns:
        str: The command to run interactively
    """
    with open(submit_script_path, 'r') as f:
        script_content = f.read()
    
    # Extract working directory
    work_dir = None
    for line in script_content.split('\n'):
        if line.strip().startswith('cd '):
            work_dir = line.strip().split('cd ')[1].strip().rstrip(';').strip('"').strip("'")
            break
    
    # Extract conda environment
    conda_env = None
    for line in script_content.split('\n'):
        if 'conda activate' in line:
            conda_env = line.strip().split('conda activate')[1].strip().rstrip(';').strip()
            break
    
    # Extract DEBUG flag from script (check for DEBUG=0 or DEBUG=1 in environment variables or qsub -v)
    debug = False
    debug_match = re.search(r'DEBUG=(\d+)', script_content)
    if debug_match:
        debug = int(debug_match.group(1)) == 1
    # Also check for $DEBUG variable usage in the script
    if '$DEBUG' in script_content or '${DEBUG}' in script_content:
        # Check if there's a conditional that uses DEBUG
        for line in script_content.split('\n'):
            if 'if' in line.lower() and 'debug' in line.lower() and ('== 1' in line or '==1' in line):
                debug = True
                break
    
    # Determine if PBS or SLURM and extract training command
    is_pbs = '#PBS' in script_content or 'qsub' in script_content.lower()
    is_slurm = '#SBATCH' in script_content or 'sbatch' in script_content.lower()
    
    # Extract the training command pattern
    training_cmd = None
    num_gpus = 4  # default
    
    # Select command based on debug flag
    if debug:
        # Debug mode: use single-process python command
        if is_pbs:
            training_cmd = f'python train.py --config={config_section} --yaml_config={config_path} --run_num={run_num}'
        else:
            training_cmd = f'python train.py --yaml_config={config_path} --run_num={run_num}'
    else:
        # Normal mode: use distributed training (torchrun)
        # Try to extract number of GPUs from PBS directives or script
        if is_pbs:
            # Check PBS directives for ngpus
            ngpus_match = re.search(r'#PBS.*ngpus=(\d+)', script_content)
            if ngpus_match:
                num_gpus = int(ngpus_match.group(1))
            # PBS scripts use environment variables
            # Look for torchrun, torch.distributed.launch, or python train.py commands
            for line in script_content.split('\n'):
                if 'torchrun' in line and 'train.py' in line:
                    # Extract nproc_per_node if present (look for numeric value)
                    nproc_match = re.search(r'--nproc_per_node=(\d+)', line)
                    if nproc_match:
                        num_gpus = int(nproc_match.group(1))
                    # Build command with actual values
                    training_cmd = f'torchrun --nproc_per_node={num_gpus} train.py --config={config_section} --yaml_config={config_path} --run_num={run_num}'
                    break
                elif 'torch.distributed.launch' in line and 'train.py' in line:
                    # Extract nproc_per_node if present (look for numeric value)
                    nproc_match = re.search(r'--nproc_per_node=(\d+)', line)
                    if nproc_match:
                        num_gpus = int(nproc_match.group(1))
                    # Convert torch.distributed.launch to torchrun
                    training_cmd = f'torchrun --nproc_per_node={num_gpus} train.py --config={config_section} --yaml_config={config_path} --run_num={run_num}'
                    break
        elif is_slurm:
            # Check SLURM directives for gpus-per-node
            gpus_match = re.search(r'#SBATCH.*--gpus-per-node=(\d+)', script_content)
            if gpus_match:
                num_gpus = int(gpus_match.group(1))
            # SLURM scripts use positional arguments
            for line in script_content.split('\n'):
                if 'torchrun' in line and 'train.py' in line:
                    # Extract nproc_per_node if present (look for numeric value)
                    nproc_match = re.search(r'--nproc_per_node=(\d+)', line)
                    if nproc_match:
                        num_gpus = int(nproc_match.group(1))
                    # Build command with actual values
                    training_cmd = f'torchrun --nproc_per_node={num_gpus} train.py --yaml_config={config_path} --run_num={run_num}'
                    break
                elif 'torch.distributed.launch' in line and 'train.py' in line:
                    # Extract nproc_per_node if present (look for numeric value)
                    nproc_match = re.search(r'--nproc_per_node=(\d+)', line)
                    if nproc_match:
                        num_gpus = int(nproc_match.group(1))
                    # Convert torch.distributed.launch to torchrun
                    training_cmd = f'torchrun --nproc_per_node={num_gpus} train.py --yaml_config={config_path} --run_num={run_num}'
                    break
        
        # If we couldn't find a specific command, use a generic one
        if training_cmd is None:
            if is_pbs:
                training_cmd = f'torchrun --nproc_per_node={num_gpus} train.py --config={config_section} --yaml_config={config_path} --run_num={run_num}'
            else:
                training_cmd = f'torchrun --nproc_per_node={num_gpus} train.py --yaml_config={config_path} --run_num={run_num}'
    
    # Add --fresh_start since JOBID won't be set in interactive mode
    if '--fresh_start' not in training_cmd:
        training_cmd += ' --fresh_start'
    
    # Add --use_legacy_model if requested
    if use_legacy_model and '--use_legacy_model' not in training_cmd:
        training_cmd += ' --use_legacy_model'
    
    # Build the full interactive command
    commands = []
    
    if work_dir:
        commands.append(f'cd {work_dir}')
    
    if conda_env:
        commands.append(f'conda activate {conda_env}')
    
    # Add environment variables that might be needed
    commands.append('export HDF5_USE_FILE_LOCKING=FALSE')
    commands.append('export WANDB_MODE=offline')
    
    commands.append(training_cmd)
    
    return ' && '.join(commands)


def submit_training_job(submit_script_path, run_num, config_path, config_section='PLASIM', scheduler_args=None, use_legacy_model=False):
    """
    Submit the training job using the submission script.
    
    Args:
        submit_script_path: Path to the submission script
        run_num: Run number for the job
        config_path: Path to the YAML config file
        scheduler_args: Optional string of additional arguments to pass to the scheduler
                       (e.g., "-l select=2:ncpus=32:ngpus=8" for PBS or "--time=24:00:00" for SLURM)
        use_legacy_model: Whether to use the legacy model (passes USE_LEGACY_MODEL environment variable)
    """
    # Get absolute paths
    submit_script_path = os.path.abspath(submit_script_path)
    config_path = os.path.abspath(config_path)
    
    # Change to the directory containing the submit script
    submit_script_dir = os.path.dirname(submit_script_path)
    submit_script_name = os.path.basename(submit_script_path)
    
    # Submit using qsub (for PBS/Torque) or sbatch (for SLURM)
    # Check which system by looking at the submit script
    with open(submit_script_path, 'r') as f:
        script_content = f.read()
    
    # Capture working directory before any chdir so it can be used for log
    # paths and exported to the job environment.
    original_dir = os.getcwd()
    log_dir = os.path.join(original_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if '#PBS' in script_content or 'qsub' in script_content.lower():
        # PBS/Torque system
        script_full_path = os.path.join(submit_script_dir, submit_script_name) if submit_script_dir else submit_script_name
        os.environ['RUN_NUM'] = run_num
        os.environ['YAML_CONFIG'] = config_path
        os.environ['DEBUG'] = '0'
        os.environ['JOBID'] = run_num
        os.environ['WORKDIR'] = original_dir
        os.environ['CONFIG'] = config_section
        if use_legacy_model:
            os.environ['USE_LEGACY_MODEL'] = '1'
        cmd = [
            'qsub',
            '-V',
            # Override the script-header -e/-o with absolute paths so logs
            # always land in v2.0/logs/ regardless of submission directory.
            '-e', log_dir,
            '-o', log_dir,
        ]
        # Add additional scheduler arguments if provided
        if scheduler_args:
            # Split the scheduler_args string into individual arguments
            cmd.extend(shlex.split(scheduler_args))
        cmd.append(script_full_path)
    elif '#SBATCH' in script_content or 'sbatch' in script_content.lower():
        # SLURM system
        script_full_path = os.path.join(submit_script_dir, submit_script_name) if submit_script_dir else submit_script_name
        cmd = ['sbatch']
        # Add environment variable export if use_legacy_model is set
        if use_legacy_model:
            cmd.extend(['--export', 'ALL,USE_LEGACY_MODEL=1'])
        # Add additional scheduler arguments if provided
        if scheduler_args:
            # Split the scheduler_args string into individual arguments
            cmd.extend(shlex.split(scheduler_args))
        cmd.extend([script_full_path, run_num, config_path])
    else:
        raise ValueError(f"Could not determine job scheduler from submit script: {submit_script_path}")

    print(f"Submitting training job with command: {' '.join(cmd)}")
    print(f"Working directory: {submit_script_dir}")
    try:
        os.chdir(submit_script_dir)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        print(f"Job submitted successfully:\n{output}")
        return output
    except subprocess.CalledProcessError as e:
        qsub_output = e.output.decode() if e.output else "(no output)"
        raise RuntimeError(
            f"qsub failed with exit code {e.returncode}:\n{qsub_output}"
        ) from None
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Start a curriculum learning fine-tuning run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the original .yaml config file'
    )
    
    parser.add_argument(
        '--config_section',
        type=str,
        default='PLASIM',
        help='Name of the config section to read from the YAML file and pass to train.py (default: PLASIM)'
    )
    
    parser.add_argument(
        '--events_json',
        type=str,
        required=True,
        help='Path to the .json file containing data directories and date ranges'
    )
    
    parser.add_argument(
        '--curriculum_fraction',
        type=float,
        required=True,
        help='Curriculum learning fraction (between 0 and 1)'
    )
    
    parser.add_argument(
        '--start_epoch',
        type=int,
        required=True,
        help='Starting epoch for fine-tuning'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        required=True,
        help='Number of epochs to train'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        required=True,
        help='Learning rate for fine-tuning'
    )
    
    parser.add_argument(
        '--submit_script',
        type=str,
        default='submit_scripts/derecho/derecho_training.sh',
        help='Path to the training submission script (default: submit_scripts/derecho/derecho_training.sh)'
    )
    
    parser.add_argument(
        '--new_run_num',
        type=str,
        default=None,
        help='Optional: specify a new run_num instead of auto-generating'
    )
    
    parser.add_argument(
        '--no_submit',
        action='store_true',
        help='Do not submit the job, just create the config and copy checkpoint'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run mode: show what would be done without making any changes or submitting jobs'
    )
    
    parser.add_argument(
        '--start_day_offset',
        type=int,
        default=0,
        help='Number of days to add to each start date in the events JSON file (default: 0). The first entry is not modified.'
    )
    
    parser.add_argument(
        '--num_events',
        type=int,
        default=0,
        help='Number of events to select from the JSON file. Selects first num_events+1 entries. Default 0 means all events.'
    )
    
    parser.add_argument(
        '--scheduler_args',
        type=str,
        default=None,
        help='Additional arguments to pass to the job scheduler (qsub or sbatch). '
             'For PBS: e.g., "-l select=2:ncpus=32:ngpus=8 -q gpu" '
             'For SLURM: e.g., "--time=24:00:00 --gres=gpu:4"'
    )
    
    parser.add_argument(
        '--finetune',
        action='store_true',
        help='Finetune mode: keep original run_num, add finetune_run_num field, '
             'set finetune_lr and finetune_num_epochs instead of modifying lr and max_epochs, '
             'and create directory without copying checkpoint'
    )
    
    parser.add_argument(
        '--use_legacy_model',
        action='store_true',
        help='Use the legacy model architecture (passes --use_legacy_model to train.py)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not 0 <= args.curriculum_fraction <= 1:
        raise ValueError(f"curriculum_fraction must be between 0 and 1, got {args.curriculum_fraction}")
    
    if args.start_epoch < 0:
        raise ValueError(f"start_epoch must be >= 0, got {args.start_epoch}")
    
    if args.num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {args.num_epochs}")
    
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {args.learning_rate}")
    
    # Get absolute paths
    original_config_path = os.path.abspath(args.config)
    events_json_path = os.path.abspath(args.events_json)
    
    if not os.path.exists(original_config_path):
        raise FileNotFoundError(f"Config file not found: {original_config_path}")
    
    if not os.path.exists(events_json_path):
        raise FileNotFoundError(f"Events JSON file not found: {events_json_path}")
    
    # Extract old run_num
    old_run_num = extract_run_num_from_config(original_config_path)
    print(f"Extracted old run_num: {old_run_num}")
    
    # Generate new run_num
    if args.new_run_num:
        new_run_num = args.new_run_num
        if len(new_run_num) != 4 or not new_run_num.isdigit():
            raise ValueError(f"new_run_num must be a 4-digit string, got {new_run_num}")
    else:
        config_dir = os.path.dirname(original_config_path)
        new_run_num = generate_new_run_num(config_dir)
    print(f"Generated new run_num: {new_run_num}")
    
    # Load events JSON data
    events_json_data = load_json_data(events_json_path)
    total_events = len(events_json_data)
    print(f"Loaded events JSON with {total_events} data directories")
    
    # Select events if num_events is specified
    if args.num_events > 0:
        events_json_data = select_events(events_json_data, args.num_events)
        print(f"Selected first {len(events_json_data)} events (num_events={args.num_events})")
    
    # Read calendar and has_year_zero from config file
    calendar = 'proleptic_gregorian'  # default
    has_year_zero = False  # default
    try:
        with open(original_config_path, 'r') as f:
            config_data_temp = yaml.safe_load(f)
            if args.config_section in config_data_temp and isinstance(config_data_temp[args.config_section], dict):
                section_config = config_data_temp[args.config_section]
                if 'calendar' in section_config:
                    calendar = section_config['calendar']
                if 'has_year_zero' in section_config:
                    has_year_zero = section_config['has_year_zero']
    except Exception as e:
        print(f"Warning: Could not read calendar from config, using defaults: {e}")
    
    print(f"Using calendar: {calendar}, has_year_zero: {has_year_zero}")
    
    # Adjust dates with offset if provided (first entry is not modified)
    if args.start_day_offset != 0:
        print(f"Adjusting start dates by {args.start_day_offset} days (first entry unchanged)...")
        events_json_data = adjust_dates_with_offset(
            events_json_data, 
            args.start_day_offset, 
            calendar, 
            has_year_zero
        )
        print(f"Adjusted {len(events_json_data) - 1} date ranges (first entry kept unchanged)")
    
    # Create new config file path
    config_dir = os.path.dirname(original_config_path)
    config_basename = os.path.basename(original_config_path)
    if args.finetune:
        # In finetune mode, create a new config file with finetune_run_num in the filename
        # Extract base name without extension
        name, ext = os.path.splitext(config_basename)
        # Create new filename with finetune_run_num appended
        new_config_basename = f"{name}_finetune_{new_run_num}{ext}"
        new_config_path = os.path.join(config_dir, new_config_basename)
    else:
        # Replace the old run_num with new one in filename
        new_config_basename = re.sub(r'_\d{4}(\.yaml)$', f'_{new_run_num}\\1', config_basename)
        if new_config_basename == config_basename:
            # If no replacement happened, append the run_num before .yaml
            name, ext = os.path.splitext(config_basename)
            new_config_basename = f"{name}_{new_run_num}{ext}"
        new_config_path = os.path.join(config_dir, new_config_basename)
    
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No changes will be made")
        print("="*60 + "\n")
    
    print(f"Creating new config file: {new_config_path}")
    
    # Create the new config file
    if args.dry_run:
        print("  [DRY RUN] Would create config file with the following modifications:")
        if args.finetune:
            print(f"    - finetune_run_num: {new_run_num}")
            print(f"    - scheduler: None")
            print(f"    - lr: {args.learning_rate}")
            print(f"    - finetune_num_epochs: {args.num_epochs}")
            print(f"    - train_data_sets: {len(events_json_data)} directories from JSON")
            print(f"    - curriculum_learning: True")
            print(f"    - curriculum_learning_fraction: {args.curriculum_fraction}")
            print(f"    - start_epoch: {args.start_epoch}")
        else:
            print(f"    - name: Pangu-{args.config_section}-{new_run_num}")
            print(f"    - scheduler: None")
            print(f"    - lr: {args.learning_rate}")
            print(f"    - max_epochs: {args.start_epoch + args.num_epochs}")
            print(f"    - train_data_sets: {len(events_json_data)} directories from JSON")
            print(f"    - curriculum_learning: True")
            print(f"    - curriculum_learning_fraction: {args.curriculum_fraction}")
            print(f"    - start_epoch: {args.start_epoch}")
    else:
        create_curriculum_config(
            original_config_path,
            new_config_path,
            events_json_data,
            args.curriculum_fraction,
            args.learning_rate,
            args.start_epoch,
            args.num_epochs,
            old_run_num,
            new_run_num,
            finetune=args.finetune,
            config_section=args.config_section
        )
        print(f"Created new config file: {new_config_path}")
    
    # Copy checkpoint
    # Read exp_dir from config if available, otherwise use default
    exp_dir = 'results'
    try:
        with open(original_config_path, 'r') as f:
            config_data_temp = yaml.safe_load(f)
            if args.config_section in config_data_temp and isinstance(config_data_temp[args.config_section], dict):
                if 'exp_dir' in config_data_temp[args.config_section]:
                    exp_dir = config_data_temp[args.config_section]['exp_dir']
    except Exception:
        pass  # Use defaults if we can't read it
    
    old_checkpoint_dir = Path(exp_dir) / args.config_section / old_run_num / 'checkpoints'
    new_checkpoint_dir = Path(exp_dir) / args.config_section / new_run_num / 'checkpoints'
    checkpoint_filename = f'ckpt_epoch_{args.start_epoch}.tar'
    old_checkpoint = old_checkpoint_dir / checkpoint_filename

    if args.finetune:
        # In finetune mode train.py loads from the original run directory and
        # creates its own output directories, so nothing needs to be done here.
        if args.dry_run:
            print(f"\n  [DRY RUN] Finetune mode: train.py will load from {old_checkpoint_dir}")
            print(f"  [DRY RUN] Would NOT copy checkpoint or create directories (train.py handles this)")
        else:
            print("\nFinetune mode: train.py will create output directories and load the "
                  f"checkpoint from the original run directory ({old_checkpoint_dir}).")
    else:
        # Normal mode: copy checkpoint so the new run can resume from start_epoch
        print(f"\nCopying checkpoint from epoch {args.start_epoch}...")
        if args.dry_run:
            if old_checkpoint.exists():
                print(f"  [DRY RUN] Would copy checkpoint:")
                print(f"    From: {old_checkpoint}")
                print(f"    To:   {new_checkpoint_dir / checkpoint_filename}")
            else:
                print(f"  [DRY RUN] WARNING: Checkpoint not found at {old_checkpoint}")
                available = list(old_checkpoint_dir.glob('ckpt_epoch_*.tar')) if old_checkpoint_dir.exists() else []
                if available:
                    print(f"    Available checkpoints: {available}")
                else:
                    print(f"    Directory does not exist: {old_checkpoint_dir}")
        else:
            copy_checkpoint(old_run_num, new_run_num, args.start_epoch, exp_dir=exp_dir, config_section=args.config_section)
            print("Checkpoint copied successfully")
    
    # Submit training job
    # In finetune mode, use original run_num for job submission (since config keeps original run_num)
    # In normal mode, use new_run_num
    job_run_num = old_run_num if args.finetune else new_run_num
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would submit training job with:")
        print(f"  Submit script: {args.submit_script}")
        print(f"  Run num: {job_run_num}")
        print(f"  Config: {new_config_path}")
        if args.use_legacy_model:
            print(f"  Use legacy model: True")
    elif not args.no_submit:
        submit_script_path = os.path.abspath(args.submit_script)
        if not os.path.exists(submit_script_path):
            raise FileNotFoundError(f"Submit script not found: {submit_script_path}")
        
        submit_training_job(submit_script_path, job_run_num, new_config_path, config_section=args.config_section, scheduler_args=args.scheduler_args, use_legacy_model=args.use_legacy_model)
    else:
        print("Skipping job submission (--no_submit flag set)")
        submit_script_path = os.path.abspath(args.submit_script)
        if not os.path.exists(submit_script_path):
            print(f"Warning: Submit script not found: {submit_script_path}")
            print("Cannot generate interactive command.")
        else:
            interactive_cmd = generate_interactive_command(submit_script_path, job_run_num, new_config_path, config_section=args.config_section, use_legacy_model=args.use_legacy_model)
            print(f"\nTo run interactively on a compute node, use:")
            print(f"  {interactive_cmd}")
    
    print("\n" + "="*60)
    if args.dry_run:
        print("DRY RUN SUMMARY - No changes were made")
    else:
        print("Curriculum learning setup complete!")
    print(f"  Old run_num: {old_run_num}")
    if args.finetune:
        print(f"  Finetune mode: finetune_run_num = {new_run_num}")
        print(f"  Config file: {new_config_path} (new file created)")
        print(f"  Original config: {original_config_path} (preserved)")
        print(f"  Finetune num epochs: {args.num_epochs}")
        print(f"  Learning rate: {args.learning_rate}")
    else:
        print(f"  New run_num: {new_run_num}")
        print(f"  Config file: {new_config_path}")
        print(f"  Max epochs: {args.start_epoch + args.num_epochs}")
        print(f"  Learning rate: {args.learning_rate}")
    print(f"  Starting epoch: {args.start_epoch}")
    print(f"  Curriculum fraction: {args.curriculum_fraction}")
    if args.start_day_offset != 0:
        print(f"  Start day offset: {args.start_day_offset} days (first entry unchanged)")
    if args.num_events > 0:
        print(f"  Number of events selected: {args.num_events + 1} (num_events={args.num_events})")
    if args.dry_run:
        print("\nTo actually perform these actions, run without --dry_run flag")
    print("="*60)


if __name__ == '__main__':
    main()

