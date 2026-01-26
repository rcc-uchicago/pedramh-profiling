#!/usr/bin/env python3
"""
Script to run start_curriculum_learning.py for a mesh/grid of parameter values.

This script runs start_curriculum_learning.py for all combinations of:
- num_events
- curriculum_learning_fraction
- learning_rate

Usage:
    python run_curriculum_learning_grid.py \
        --config config/PANGU_PLASIM_H5_DERECHO_0514.yaml \
        --events_json events/Chicago_heatwaves_test.json \
        --start_epoch 50 \
        --num_epochs 20 \
        --num_events 0,1,2 \
        --curriculum_fractions 0.3,0.5,0.7 \
        --learning_rates 0.00005,0.0001,0.0002
"""

import argparse
import itertools
import os
import subprocess
import sys


def parse_list_arg(arg_str, arg_type):
    """Parse a comma-separated list of values."""
    if not arg_str:
        return []
    return [arg_type(x.strip()) for x in arg_str.split(',') if x.strip()]


def run_curriculum_learning(
    script_path,
    config,
    events_json,
    curriculum_fraction,
    learning_rate,
    start_epoch,
    num_epochs,
    num_events=0,
    start_day_offset=0,
    submit_script=None,
    new_run_num=None,
    no_submit=False,
    dry_run=False,
    scheduler_args=None
):
    """Run start_curriculum_learning.py with specified parameters."""
    cmd = [
        sys.executable,
        script_path,
        '--config', config,
        '--events_json', events_json,
        '--curriculum_fraction', str(curriculum_fraction),
        '--learning_rate', str(learning_rate),
        '--start_epoch', str(start_epoch),
        '--num_epochs', str(num_epochs),
        '--num_events', str(num_events),
        '--start_day_offset', str(start_day_offset),
    ]
    
    if submit_script:
        cmd.extend(['--submit_script', submit_script])
    
    if new_run_num:
        cmd.extend(['--new_run_num', new_run_num])
    
    if no_submit:
        cmd.append('--no_submit')
    
    if dry_run:
        cmd.append('--dry_run')
    
    if scheduler_args:
        cmd.extend(['--scheduler_args', scheduler_args])
    
    print(f"\n{'='*80}")
    print(f"Running with parameters:")
    print(f"  num_events: {num_events}")
    print(f"  curriculum_fraction: {curriculum_fraction}")
    print(f"  learning_rate: {learning_rate}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running command: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        return False, error_msg


def main():
    parser = argparse.ArgumentParser(
        description='Run start_curriculum_learning.py for a grid of parameter values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments (same as start_curriculum_learning.py)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the original .yaml config file'
    )
    
    parser.add_argument(
        '--events_json',
        type=str,
        required=True,
        help='Path to the .json file containing data directories and date ranges'
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
    
    # Grid search parameters (comma-separated lists)
    parser.add_argument(
        '--num_events',
        type=str,
        required=True,
        help='Comma-separated list of num_events values (e.g., "0,1,2")'
    )
    
    parser.add_argument(
        '--curriculum_fractions',
        type=str,
        required=True,
        help='Comma-separated list of curriculum_fraction values (e.g., "0.3,0.5,0.7")'
    )
    
    parser.add_argument(
        '--learning_rates',
        type=str,
        required=True,
        help='Comma-separated list of learning_rate values (e.g., "0.00005,0.0001,0.0002")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--start_day_offset',
        type=int,
        default=0,
        help='Number of days to add to each start date (default: 0)'
    )
    
    parser.add_argument(
        '--submit_script',
        type=str,
        default='submit_scripts/derecho/derecho_training.sh',
        help='Path to the training submission script'
    )
    
    parser.add_argument(
        '--no_submit',
        action='store_true',
        help='Do not submit jobs, just create configs and copy checkpoints'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run mode: show what would be done without making changes'
    )
    
    parser.add_argument(
        '--script_path',
        type=str,
        default=None,
        help='Path to start_curriculum_learning.py script (default: same directory as this script)'
    )
    
    parser.add_argument(
        '--stop_on_error',
        action='store_true',
        help='Stop execution if any run fails (default: continue with remaining combinations)'
    )
    
    parser.add_argument(
        '--scheduler_args',
        type=str,
        default=None,
        help='Additional arguments to pass to the job scheduler (qsub or sbatch). '
             'For PBS: e.g., "-l select=2:ncpus=32:ngpus=8 -q gpu" '
             'For SLURM: e.g., "--time=24:00:00 --gres=gpu:4"'
    )
    
    args = parser.parse_args()
    
    # Get the path to start_curriculum_learning.py
    if args.script_path:
        script_path = os.path.abspath(args.script_path)
    else:
        # Default: look for it in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'start_curriculum_learning.py')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"start_curriculum_learning.py not found at: {script_path}")
    
    # Parse the grid parameters
    num_events_list = parse_list_arg(args.num_events, int)
    curriculum_fractions_list = parse_list_arg(args.curriculum_fractions, float)
    learning_rates_list = parse_list_arg(args.learning_rates, float)
    
    if not num_events_list:
        raise ValueError("--num_events must contain at least one value")
    if not curriculum_fractions_list:
        raise ValueError("--curriculum_fractions must contain at least one value")
    if not learning_rates_list:
        raise ValueError("--learning_rates must contain at least one value")
    
    # Validate curriculum fractions
    for cf in curriculum_fractions_list:
        if not 0 <= cf <= 1:
            raise ValueError(f"curriculum_fraction must be between 0 and 1, got {cf}")
    
    # Get absolute paths
    config_path = os.path.abspath(args.config)
    events_json_path = os.path.abspath(args.events_json)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(events_json_path):
        raise FileNotFoundError(f"Events JSON file not found: {events_json_path}")
    
    # Generate all combinations
    combinations = list(itertools.product(
        num_events_list,
        curriculum_fractions_list,
        learning_rates_list
    ))
    
    total_combinations = len(combinations)
    print(f"\n{'='*80}")
    print(f"GRID SEARCH PARAMETERS")
    print(f"{'='*80}")
    print(f"num_events: {num_events_list}")
    print(f"curriculum_fractions: {curriculum_fractions_list}")
    print(f"learning_rates: {learning_rates_list}")
    print(f"\nTotal combinations: {total_combinations}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("DRY RUN MODE - No actual runs will be executed\n")
        for i, (num_events, curriculum_fraction, learning_rate) in enumerate(combinations, 1):
            print(f"Combination {i}/{total_combinations}:")
            print(f"  num_events: {num_events}")
            print(f"  curriculum_fraction: {curriculum_fraction}")
            print(f"  learning_rate: {learning_rate}")
        print(f"\nTotal: {total_combinations} combinations would be run")
        return
    
    # Run each combination
    successful = 0
    failed = 0
    failed_combinations = []
    
    for i, (num_events, curriculum_fraction, learning_rate) in enumerate(combinations, 1):
        print(f"\n[{i}/{total_combinations}] Processing combination...")
        
        success, error = run_curriculum_learning(
            script_path=script_path,
            config=config_path,
            events_json=events_json_path,
            curriculum_fraction=curriculum_fraction,
            learning_rate=learning_rate,
            start_epoch=args.start_epoch,
            num_epochs=args.num_epochs,
            num_events=num_events,
            start_day_offset=args.start_day_offset,
            submit_script=args.submit_script if not args.no_submit else None,
            no_submit=args.no_submit,
            dry_run=False,  # Already handled above
            scheduler_args=args.scheduler_args
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_combinations.append({
                'num_events': num_events,
                'curriculum_fraction': curriculum_fraction,
                'learning_rate': learning_rate,
                'error': error
            })
            
            if args.stop_on_error:
                print(f"\nStopping due to error (--stop_on_error flag set)")
                break
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"GRID SEARCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_combinations:
        print(f"\nFailed combinations:")
        for combo in failed_combinations:
            print(f"  num_events={combo['num_events']}, "
                  f"curriculum_fraction={combo['curriculum_fraction']}, "
                  f"learning_rate={combo['learning_rate']}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

