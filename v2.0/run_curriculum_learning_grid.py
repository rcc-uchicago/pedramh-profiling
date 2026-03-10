#!/usr/bin/env python3
"""
Run start_curriculum_learning.py helpers for a grid of parameter values,
configured entirely via a YAML file.

The events JSON must be in the format produced by create_event_json.py:
    {
        "event_type_1": {
            "/path/to/particle_N/file.nc": ["YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS"],
            ...
        },
        "event_type_2": { ... }
    }

For each particle nc file the h5 data directory used by train.py is derived as:
    os.path.join(os.path.dirname(nc_file), 'h5')

Example YAML config (grid_config.yaml):

    # Required
    config: config/PANGU_PLASIM_H5_DERECHO_0514.yaml
    events_json: events/ensemble_validation_test_derecho.json
    event_types: [typical, extreme]
    start_epoch: 50
    num_epochs: 20

    # Grid axes
    #
    # num_events accepts four forms:
    #   (a) single int       → one value, broadcast to all types, for every fraction
    #   (b) list of ints     → grid axis; each int broadcast to all event types;
    #                          iterates all (num_events × fractions × rates) combos
    #   (c) list of lists    → paired mode; outer list length == len(curriculum_fractions);
    #                          num_events[i] gives per-type counts for fractions[i];
    #                          iterates (learning_rates) only
    #   (d) list of lists-of-lists → paired mode with multiple sets per fraction;
    #                          outer list length == len(curriculum_fractions);
    #                          num_events[i] is a list of per-type count lists;
    #                          iterates each set × learning_rates for fraction[i]
    #
    num_events: [2, 4]       # (b) two grid-axis entries, each broadcast to all types
    # num_events: 4          # (a) single value for all fractions
    # num_events:            # (c) one per-type list per fraction
    #   - [2, 1]
    #   - [4, 2]
    # num_events:            # (d) multiple per-type lists per fraction
    #   - [[2, 1], [4, 2]]
    #   - [[8, 4], [16, 8]]
    curriculum_fractions: [0.3, 0.5]
    learning_rates: [0.00005, 0.0001]

    # Selection criteria: "first", "last", or "random"
    # Can be a single string or a list with one entry per event type
    selection_criteria: first

    # Optional (shown with their defaults)
    config_section: PLASIM
    start_day_offset: 0
    submit_script: submit_scripts/derecho/derecho_training.sh
    no_submit: false
    dry_run: false
    stop_on_error: false
    scheduler_args: null
    finetune: false
    use_legacy_model: false
    random_seed: null        # integer seed for reproducible random selection
    exp_dir: null            # if null, read from base config or default to 'results'
    load_exp_dir: null       # if null, read from base config or default to 'results'
    event_data_len: null     # required when any curriculum_fraction >= 1.0;
                             # curriculum_bulk_size = sum(num_events) * event_data_len
    balanced_learning: false # if true, only fraction=0.5 runs are generated and
                             # balanced_learning: true is written to the new config

Usage:
    python run_curriculum_learning_grid.py grid_config.yaml
"""

import itertools
import json
import os
import random
import re
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import yaml

# Import helpers directly from start_curriculum_learning so we avoid subprocess
# overhead and temp-file creation.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from start_curriculum_learning import (
    adjust_dates_with_offset,
    copy_checkpoint,
    create_curriculum_config,
    extract_run_num_from_config,
    generate_interactive_command,
    generate_new_run_num,
    submit_training_job,
)

# ------------------------------------------------------------------ #
# Config loading                                                       #
# ------------------------------------------------------------------ #

_REQUIRED_KEYS = {
    'config', 'events_json', 'event_types',
    'start_epoch', 'num_epochs',
    'num_events', 'curriculum_fractions', 'learning_rates',
}

_DEFAULTS = {
    'config_section': 'PLASIM',
    'selection_criteria': 'first',
    'particle_indices': None,
    'start_day_offset': 0,
    'submit_script': 'submit_scripts/derecho/derecho_training.sh',
    'no_submit': False,
    'dry_run': False,
    'stop_on_error': False,
    'scheduler_args': None,
    'finetune': False,
    'use_legacy_model': False,
    'random_seed': None,
    'exp_dir': None,
    'load_exp_dir': None,
    'ensemble_validation_params': None,
    'ensemble_validation_frequency': None,
    'event_data_len': None,
    'use_ema': False,
    'ema_decay': None,
    'train_date_ranges': None,
    'balanced_learning': None,
}


def load_grid_config(yaml_path: str) -> dict:
    """Load and validate the YAML grid config, applying defaults for optional keys."""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(f"Grid config is missing required keys: {sorted(missing)}")

    for key, default in _DEFAULTS.items():
        cfg.setdefault(key, default)

    return cfg


# ------------------------------------------------------------------ #
# Events JSON processing                                               #
# ------------------------------------------------------------------ #

def particle_nc_to_h5_dir(nc_file_path: str) -> str:
    """Return the h5 data directory corresponding to a particle nc file path.

    Convention: the h5 directory lives at <particle_dir>/h5/
    where <particle_dir> is the parent directory of the nc file.
    """
    return os.path.join(os.path.dirname(nc_file_path), 'h5')


def select_events_for_type(
    events_dict: Dict[str, list],
    n: int,
    criteria: str,
    rng: Optional[random.Random] = None,
    particle_indices: Optional[List[int]] = None,
) -> Dict[str, list]:
    """Select *n* events from a single event-type dict.

    Args:
        events_dict: {nc_path: [start_dt, end_dt]}
        n: Number of events to select. 0 means select all.
        criteria: "first", "last", or "random"
        rng: Random instance used when criteria == "random"
        particle_indices: Optional list of integer indices used to reorder (and
            optionally filter) the particles before applying the selection
            criteria.  Each value is an index into the natural ordering of
            ``events_dict``.  Out-of-range indices are silently skipped.
            If ``None`` the natural ordering is kept.

    Returns:
        Sub-dict with the selected entries (insertion order preserved).
    """
    items = list(events_dict.items())

    if particle_indices is not None:
        items = [items[i] for i in particle_indices if 0 <= i < len(items)]

    if n == 0 or n >= len(items):
        return dict(items)

    if criteria == 'first':
        selected = items[:n]
    elif criteria == 'last':
        selected = items[-n:]
    elif criteria == 'random':
        if rng is None:
            rng = random.Random()
        selected = rng.sample(items, n)
    else:
        raise ValueError(
            f"selection_criteria must be 'first', 'last', or 'random', got '{criteria}'"
        )

    return dict(selected)


def build_flat_events_dict(
    events_json: Dict[str, Dict[str, list]],
    event_types: List[str],
    counts_per_type: List[int],
    criteria_per_type: List[str],
    rng: Optional[random.Random],
    particle_indices_per_type: Optional[List[Optional[List[int]]]] = None,
) -> Dict[str, list]:
    """Build the flat {h5_dir: [start_dt, end_dt]} dict expected by train.py.

    For each event type the selected particle nc paths are mapped to their
    h5 subdirectories and merged into a single dict.

    Args:
        particle_indices_per_type: Optional list (one entry per event type) of
            integer index lists used to reorder particles before selection.
            Pass ``None`` for the whole argument or ``None`` for an individual
            entry to use the natural ordering for that type.
    """
    flat: Dict[str, list] = {}
    for i, (event_type, n, criteria) in enumerate(
        zip(event_types, counts_per_type, criteria_per_type)
    ):
        events = events_json.get(event_type, {})
        pidxs = (
            particle_indices_per_type[i]
            if particle_indices_per_type is not None
            else None
        )
        selected = select_events_for_type(events, n, criteria, rng, pidxs)
        for nc_path, date_range in selected.items():
            h5_dir = particle_nc_to_h5_dir(nc_path)
            flat[h5_dir] = date_range
    return flat


def normalize_num_events(
    raw,
    n_types: int,
    n_fractions: int,
) -> Tuple[list, bool]:
    """Normalise the *num_events* config value.

    Four accepted forms:

    1. **Single integer** (``num_events: 4``):
       Broadcast to all event types; used for every curriculum fraction.
       Equivalent to a one-element list of ints (mode 2).

    2. **List of integers** (``num_events: [4, 8, 16]``):
       Grid axis.  Each integer is broadcast to all event types and every
       combination with ``curriculum_fractions`` and ``learning_rates`` is run.

    3. **List of per-type count lists** (``num_events: [[4, 2], [8, 4]]``):
       *Paired* mode — one per-type count list per curriculum fraction.
       The outer list **must** have the same length as ``curriculum_fractions``.
       ``num_events[i]`` provides per-type counts for ``curriculum_fractions[i]``.
       The grid iterates over ``learning_rates`` only.

    4. **List of lists-of-lists** (``num_events: [[[4,2],[6,3]], [[8,4],[12,6]]]``):
       *Paired* mode with multiple num_events sets per curriculum fraction.
       The outer list **must** have the same length as ``curriculum_fractions``.
       ``num_events[i]`` is a list of per-type count lists; the grid iterates
       over each of those lists *and* over ``learning_rates``.

    Returns:
        ``(normalised_list, paired_mode)``

        * If ``paired_mode`` is ``False`` (modes 1/2): a ``List[List[int]]``
          where each inner list is one per-type count entry on the grid axis.
        * If ``paired_mode`` is ``True`` (modes 3/4): a ``List[List[List[int]]]``
          where ``normalised_list[i]`` is the list of per-type count lists for
          ``curriculum_fractions[i]`` (length 1 for mode 3, ≥1 for mode 4).
    """
    # Mode 1: bare integer
    if isinstance(raw, int):
        return [[raw] * n_types], False

    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            f"num_events must be an int, a list of ints, or a list of lists; "
            f"got {type(raw).__name__}"
        )

    raw = list(raw)
    if not raw:
        raise ValueError("num_events must not be empty")

    all_int  = all(isinstance(e, int) for e in raw)
    all_list = all(isinstance(e, (list, tuple)) for e in raw)

    if not (all_int or all_list):
        raise TypeError(
            "num_events must be a list of ints OR a list of lists — "
            "mixing scalars and lists is not allowed"
        )

    if all_int:
        # Mode 2: list of ints – grid axis, each broadcast to all event types
        return [[v] * n_types for v in raw], False

    # Modes 3/4: list of lists – paired with curriculum_fractions
    if len(raw) != n_fractions:
        raise ValueError(
            f"When num_events is a list of lists it must have the same length "
            f"as curriculum_fractions ({n_fractions}), got {len(raw)}"
        )
    result = []
    for i, entry in enumerate(raw):
        entry = list(entry)
        if not entry:
            raise ValueError(f"num_events[{i}] must not be empty")

        if all(isinstance(e, int) for e in entry):
            # Mode 3: single per-type count list for this fraction
            if len(entry) != n_types:
                raise ValueError(
                    f"num_events[{i}] has {len(entry)} elements but there are "
                    f"{n_types} event types — must match"
                )
            result.append([[int(v) for v in entry]])

        elif all(isinstance(e, (list, tuple)) for e in entry):
            # Mode 4: multiple per-type count lists for this fraction
            ne_lists = []
            for j, sub in enumerate(entry):
                sub = list(sub)
                if len(sub) != n_types:
                    raise ValueError(
                        f"num_events[{i}][{j}] has {len(sub)} elements but "
                        f"there are {n_types} event types — must match"
                    )
                ne_lists.append([int(v) for v in sub])
            result.append(ne_lists)

        else:
            raise TypeError(
                f"num_events[{i}] must be a list of ints or a list of lists "
                f"of ints — mixing is not allowed"
            )
    return result, True


# ------------------------------------------------------------------ #
# Per-combination runner                                               #
# ------------------------------------------------------------------ #

def run_one_combination(
    cfg: dict,
    flat_events: Dict[str, list],
    curriculum_fraction: float,
    learning_rate: float,
    old_run_num: str,
    calendar: str,
    has_year_zero: bool,
    ensemble_params: Optional[dict] = None,
    curriculum_bulk_size: Optional[int] = None,
    reserved_run_nums: Optional[set] = None,
    use_ema: bool = False,
    ema_decay: Optional[float] = None,
    train_date_ranges: Optional[list] = None,
    ensemble_validation_frequency: Optional[int] = None,
    balanced_learning: Optional[bool] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Run the curriculum learning setup for one grid combination.

    Mirrors the logic in start_curriculum_learning.main() but calls helpers
    directly rather than going through a subprocess.

    Returns:
        (success, new_run_num, error_message)
    """
    config_section = cfg['config_section']
    original_config_path = os.path.abspath(cfg['config'])
    config_dir = os.path.dirname(original_config_path)
    config_basename = os.path.basename(original_config_path)
    finetune = cfg['finetune']
    dry_run = cfg['dry_run']
    no_submit = cfg['no_submit']
    exp_dir = cfg['exp_dir'] or 'results'
    load_exp_dir = cfg['load_exp_dir'] or 'results'


    try:
        # Adjust dates with offset
        events_data = adjust_dates_with_offset(
            flat_events, cfg['start_day_offset'], calendar, has_year_zero
        )

        # Generate a unique run number, excluding any already reserved in this
        # session (covers dry-run and rapid real-mode execution where the
        # previous config file may not yet be visible on disk).
        new_run_num = generate_new_run_num(config_dir, extra_exclude=reserved_run_nums)
        if reserved_run_nums is not None:
            reserved_run_nums.add(int(new_run_num))

        # Build new config path (same logic as start_curriculum_learning)
        if finetune:
            name, ext = os.path.splitext(config_basename)
            new_config_basename = f"{name}_finetune_{new_run_num}{ext}"
        else:
            new_config_basename = re.sub(
                r'_\d{4}(\.yaml)$', f'_{new_run_num}\\1', config_basename
            )
            if new_config_basename == config_basename:
                name, ext = os.path.splitext(config_basename)
                new_config_basename = f"{name}_{new_run_num}{ext}"
        new_config_path = os.path.join(config_dir, new_config_basename)

        print(f"  new_run_num:          {new_run_num}")
        print(f"  new_config_path:      {new_config_path}")
        print(f"  events (h5 dirs):     {len(events_data)}")

        if dry_run:
            print(f"  [DRY RUN] Would create config and {'skip checkpoint (finetune)' if finetune else 'copy checkpoint'}")
        else:
            # Create config
            create_curriculum_config(
                original_config_path,
                new_config_path,
                events_data,
                curriculum_fraction,
                learning_rate,
                cfg['start_epoch'],
                cfg['num_epochs'],
                old_run_num,
                new_run_num,
                finetune=finetune,
                config_section=config_section,
                ensemble_params=ensemble_params,
                curriculum_bulk_size=curriculum_bulk_size,
                exp_dir=exp_dir,
                load_exp_dir=load_exp_dir,
                use_ema=use_ema,
                ema_decay=ema_decay,
                train_date_range=train_date_ranges,
                ensemble_validation_frequency=ensemble_validation_frequency,
                balanced_learning=balanced_learning,
            )

            # Copy / skip checkpoint
            if finetune:
                print(f"  Finetune mode: train.py will load checkpoint from original run directory")
            else:
                copy_checkpoint(
                    old_run_num, new_run_num, cfg['start_epoch'],
                    exp_dir=load_exp_dir, config_section=config_section
                )

        # Submit or print interactive command
        job_run_num = old_run_num if finetune else new_run_num

        if dry_run:
            print(f"  [DRY RUN] Would submit job: run_num={job_run_num}, config={new_config_path}")
        elif not no_submit:
            submit_script_path = os.path.abspath(cfg['submit_script'])
            if not os.path.exists(submit_script_path):
                raise FileNotFoundError(f"Submit script not found: {submit_script_path}")
            submit_training_job(
                submit_script_path, job_run_num, new_config_path,
                config_section=config_section,
                scheduler_args=cfg['scheduler_args'],
                use_legacy_model=cfg['use_legacy_model'],
            )
        else:
            submit_script_path = os.path.abspath(cfg['submit_script'])
            if os.path.exists(submit_script_path):
                interactive_cmd = generate_interactive_command(
                    submit_script_path, job_run_num, new_config_path,
                    config_section=config_section,
                    use_legacy_model=cfg['use_legacy_model'],
                )
                print(f"  To run interactively:\n    {interactive_cmd}")
            else:
                print(f"  Skipping submission (--no_submit). Submit script not found: {submit_script_path}")

        return True, new_run_num, None

    except Exception:
        error_msg = traceback.format_exc()
        print(f"  ERROR:\n{error_msg}", file=sys.stderr)
        return False, None, error_msg


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run curriculum learning for a grid of parameter values (YAML-configured)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to the YAML grid config file',
    )
    args = parser.parse_args()

    # ---- load config ------------------------------------------------
    cfg = load_grid_config(os.path.abspath(args.config))

    original_config_path = os.path.abspath(cfg['config'])
    if not os.path.exists(original_config_path):
        raise FileNotFoundError(f"Base config not found: {original_config_path}")

    events_json_path = os.path.abspath(cfg['events_json'])
    if not os.path.exists(events_json_path):
        raise FileNotFoundError(f"Events JSON not found: {events_json_path}")

    event_types: List[str] = list(cfg['event_types'])
    n_types = len(event_types)

    # ---- load events JSON -------------------------------------------
    with open(events_json_path, 'r') as f:
        events_json: Dict[str, Dict[str, list]] = json.load(f)

    for et in event_types:
        if et not in events_json:
            raise ValueError(
                f"Event type '{et}' not found in events JSON. "
                f"Available types: {list(events_json.keys())}"
            )

    # ---- normalise num_events and selection_criteria ----------------
    num_events_normalized, paired_mode = normalize_num_events(
        cfg['num_events'], n_types, len(cfg['curriculum_fractions'])
    )

    sc_raw = cfg['selection_criteria']
    if isinstance(sc_raw, str):
        criteria_per_type = [sc_raw] * n_types
    else:
        criteria_per_type = list(sc_raw)
        if len(criteria_per_type) != n_types:
            raise ValueError(
                f"selection_criteria has {len(criteria_per_type)} entries "
                f"but there are {n_types} event types"
            )
    for c in criteria_per_type:
        if c not in ('first', 'last', 'random'):
            raise ValueError(f"selection_criteria must be 'first', 'last', or 'random', got '{c}'")

    pi_raw = cfg['particle_indices']
    if pi_raw is None:
        particle_indices_per_type: Optional[List[Optional[List[int]]]] = None
    elif isinstance(pi_raw[0], (int, type(None))):
        # Single list of ints (or None entries) → broadcast to all event types
        particle_indices_per_type = [list(pi_raw) if pi_raw is not None else None] * n_types
    else:
        # List of per-type lists
        particle_indices_per_type = [
            list(entry) if entry is not None else None for entry in pi_raw
        ]
        if len(particle_indices_per_type) != n_types:
            raise ValueError(
                f"particle_indices has {len(particle_indices_per_type)} entries "
                f"but there are {n_types} event types"
            )

    curriculum_fractions: List[float] = list(cfg['curriculum_fractions'])
    learning_rates: List[float] = list(cfg['learning_rates'])

    for cf in curriculum_fractions:
        if cf < 0:
            raise ValueError(f"curriculum_fraction must be >= 0, got {cf}")

    # ---- balanced_learning: filter fractions to 0.5 only -----------
    balanced_learning = bool(cfg['balanced_learning'])
    if balanced_learning:
        if paired_mode:
            paired = [(cf, ne_lists) for cf, ne_lists in
                      zip(curriculum_fractions, num_events_normalized) if cf == 0.5]
            if not paired:
                raise ValueError(
                    "balanced_learning is True but curriculum_fractions contains no value of 0.5"
                )
            curriculum_fractions, num_events_normalized = map(list, zip(*paired))
        else:
            curriculum_fractions = [cf for cf in curriculum_fractions if cf == 0.5]
            if not curriculum_fractions:
                raise ValueError(
                    "balanced_learning is True but curriculum_fractions contains no value of 0.5"
                )

    # ---- read calendar from base config -----------------------------
    calendar = 'proleptic_gregorian'
    has_year_zero = False
    try:
        with open(original_config_path, 'r') as f:
            base_cfg_tmp = yaml.safe_load(f)
        section = base_cfg_tmp.get(cfg['config_section'], {})
        if isinstance(section, dict):
            calendar = section.get('calendar', calendar)
            has_year_zero = section.get('has_year_zero', has_year_zero)
    except Exception as exc:
        print(f"Warning: Could not read calendar from base config, using defaults: {exc}")

    # ---- extract old run_num ----------------------------------------
    old_run_num = extract_run_num_from_config(original_config_path)

    # ---- read exp_dir from base config if not set in grid config ----
    if cfg['exp_dir'] is None:
        try:
            with open(original_config_path, 'r') as f:
                base_cfg_tmp = yaml.safe_load(f)
            section = base_cfg_tmp.get(cfg['config_section'], {})
            if isinstance(section, dict) and 'exp_dir' in section:
                cfg['exp_dir'] = section['exp_dir']
        except Exception:
            pass
        if cfg['exp_dir'] is None:
            cfg['exp_dir'] = 'results'

    # ---- read exp_dir to load_exp_dir from base config if not set in grid config ----
    if cfg['load_exp_dir'] is None:
        try:
            with open(original_config_path, 'r') as f:
                base_cfg_tmp = yaml.safe_load(f)
            section = base_cfg_tmp.get(cfg['config_section'], {})
            if isinstance(section, dict) and 'exp_dir' in section:
                cfg['load_exp_dir'] = section['exp_dir']
        except Exception:
            pass
        if cfg['load_exp_dir'] is None:
            cfg['load_exp_dir'] = 'results'

    # ---- RNG --------------------------------------------------------
    rng = random.Random(cfg['random_seed'])

    # ---- EMA decay grid values --------------------------------------
    # When use_ema is True and ema_decay is set, normalise it to a list so
    # it becomes its own grid axis (like learning_rates).  When use_ema is
    # False or ema_decay is not set, carry a single None through every
    # combination so the tuple arity stays consistent.
    if cfg['use_ema'] and cfg['ema_decay'] is not None:
        raw_ema = cfg['ema_decay']
        ema_decay_values: List[Optional[float]] = (
            [float(v) for v in raw_ema] if isinstance(raw_ema, list) else [float(raw_ema)]
        )
    else:
        ema_decay_values = [None]

    # ---- build combinations -----------------------------------------
    if paired_mode:
        # num_events_normalized[i] is a List[List[int]] of per-type count lists
        # paired with curriculum_fractions[i].  Grid over each list,
        # learning_rates, and ema_decay_values.
        combinations = [
            (ne, curriculum_fractions[i], lr, ed)
            for i, ne_lists in enumerate(num_events_normalized)
            for ne in ne_lists
            for lr in learning_rates
            for ed in ema_decay_values
        ]
        num_events_mode = 'paired with curriculum_fractions'
    else:
        # Grid over all combinations of num_events × curriculum_fractions ×
        # learning_rates × ema_decay_values.
        combinations = list(itertools.product(
            num_events_normalized, curriculum_fractions, learning_rates, ema_decay_values
        ))
        num_events_mode = 'grid axis'
    total = len(combinations)

    # Validate that event_data_len is provided when any fraction >= 1.0.
    bulk_fractions = [cf for cf in curriculum_fractions if cf >= 1.0]
    if bulk_fractions and cfg['event_data_len'] is None:
        raise ValueError(
            f"curriculum_fractions contains values >= 1.0 ({bulk_fractions}) which "
            f"require bulk-size mode, but 'event_data_len' is not set in the grid config."
        )

    print(f"\n{'='*80}")
    print("GRID SEARCH PARAMETERS")
    print(f"{'='*80}")
    print(f"  base config:          {original_config_path}")
    print(f"  events JSON:          {events_json_path}")
    print(f"  event_types:          {event_types}")
    print(f"  selection_criteria:   {criteria_per_type}")
    if particle_indices_per_type is not None:
        print(f"  particle_indices:     {dict(zip(event_types, particle_indices_per_type))}")
    print(f"  num_events ({num_events_mode}):")
    if paired_mode:
        for cf, ne_lists in zip(curriculum_fractions, num_events_normalized):
            for ne in ne_lists:
                print(f"    fraction={cf}: {dict(zip(event_types, ne))}")
    else:
        for ne in num_events_normalized:
            print(f"    {dict(zip(event_types, ne))}")
    print(f"  curriculum_fractions: {curriculum_fractions}")
    if cfg['event_data_len'] is not None:
        print(f"  event_data_len:       {cfg['event_data_len']}  "
              f"(curriculum_bulk_size = sum(num_events) × event_data_len when fraction ≥ 1.0)")
    print(f"  learning_rates:       {learning_rates}")
    if cfg['use_ema'] and ema_decay_values != [None]:
        print(f"  ema_decay_values:     {ema_decay_values}")
    print(f"  calendar:             {calendar}  (has_year_zero={has_year_zero})")
    print(f"  start_epoch:          {cfg['start_epoch']}")
    print(f"  num_epochs:           {cfg['num_epochs']}")
    print(f"  finetune:             {cfg['finetune']}")
    if balanced_learning:
        print(f"  balanced_learning:    True  (only fraction=0.5 runs)")
    print(f"\n  Total combinations:   {total}")
    print(f"{'='*80}\n")

    if cfg['dry_run']:
        print("DRY RUN MODE — no files will be created or jobs submitted\n")

    # ---- run each combination ---------------------------------------
    successful = 0
    failed = 0
    failed_combos = []
    # Shared set of run_nums reserved in this session.  run_one_combination
    # adds each generated number here immediately so subsequent combinations
    # receive a distinct number even in dry-run mode (where no files are
    # written to disk between iterations).
    reserved_run_nums: set = set()

    for i, (counts_per_type, curriculum_fraction, learning_rate, ema_decay) in enumerate(combinations, 1):
        print(f"\n[{i}/{total}] Combination:")
        print(f"  num_events (per type): {dict(zip(event_types, counts_per_type))}")
        print(f"  curriculum_fraction:   {curriculum_fraction}")
        print(f"  learning_rate:         {learning_rate}")
        if ema_decay is not None:
            print(f"  ema_decay:             {ema_decay}")

        # When curriculum_fraction >= 1.0 switch to bulk-size mode:
        #   - select ALL available events (counts → 0)
        #   - set curriculum_bulk_size = sum(counts_per_type) × event_data_len
        if curriculum_fraction >= 1.0:
            curriculum_bulk_size = sum(counts_per_type) * cfg['event_data_len']
            event_counts = [0] * n_types
            print(f"  [bulk mode] curriculum_bulk_size: {curriculum_bulk_size} "
                  f"(all events selected)")
        else:
            curriculum_bulk_size = None
            event_counts = counts_per_type

        flat_events = build_flat_events_dict(
            events_json, event_types, event_counts, criteria_per_type, rng,
            particle_indices_per_type,
        )

        success, new_run_num, error = run_one_combination(
            cfg=cfg,
            flat_events=flat_events,
            curriculum_fraction=curriculum_fraction,
            learning_rate=learning_rate,
            old_run_num=old_run_num,
            calendar=calendar,
            has_year_zero=has_year_zero,
            ensemble_params=cfg['ensemble_validation_params'],
            curriculum_bulk_size=curriculum_bulk_size,
            reserved_run_nums=reserved_run_nums,
            use_ema=cfg['use_ema'],
            ema_decay=ema_decay,
            train_date_ranges=cfg['train_date_ranges'],
            ensemble_validation_frequency=cfg['ensemble_validation_frequency'],
            balanced_learning=balanced_learning if balanced_learning else None,
        )

        if success:
            successful += 1
            print(f"  [OK] new_run_num={new_run_num}")
        else:
            failed += 1
            failed_combos.append({
                'counts_per_type': counts_per_type,
                'curriculum_fraction': curriculum_fraction,
                'learning_rate': learning_rate,
                'ema_decay': ema_decay,
                'error': error,
            })
            if cfg['stop_on_error']:
                print("\nStopping due to error (stop_on_error: true)")
                break

    # ---- summary ----------------------------------------------------
    print(f"\n{'='*80}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*80}")
    print(f"  Total combinations: {total}")
    print(f"  Successful:         {successful}")
    print(f"  Failed:             {failed}")
    if failed_combos:
        print("\n  Failed combinations:")
        for c in failed_combos:
            counts = dict(zip(event_types, c['counts_per_type']))
            ema_str = f", ema_decay={c['ema_decay']}" if c.get('ema_decay') is not None else ""
            print(f"    num_events={counts}, "
                  f"curriculum_fraction={c['curriculum_fraction']}, "
                  f"learning_rate={c['learning_rate']}{ema_str}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
