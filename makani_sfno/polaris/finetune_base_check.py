"""Pre-submit gate for a fine-tune arm that starts from a CHANNEL-SUBSET base (ports F, G).

polaris_makani_f_finetune_handoff.md §2.3: the arm launcher must refuse a checkpoint
whose channel count is not the config's. The checkpoint itself needs torch to open,
and the login node has none, so this reads what the checkpoint was trained with --
the run directory's ``config.json`` (``<run>/training_checkpoints/<ckpt>.tar``) --
and compares it with the arm's config yaml:

  - ``N_out_channels`` == len(yaml ``channel_names``)   (99 for F, 77 for G)
  - ``channel_names``  == yaml ``channel_names``, name for name, in order
  - the pack: the parent of the base's ``global_means_path``. The arm must train on
    it, so that it normalizes with the stats its base learned under (F: the
    production pack, 2015-2044; G: the production pack too; H: the 2020-2044 view).
    Printed for the launcher.

A 101-channel checkpoint with a 99-channel config would otherwise fail only at the
strict restore on the compute node, after the queue wait.

    python3 finetune_base_check.py <ckpt.tar> <config.yaml> <root_key>
      -> "FINETUNE_BASE_OK n_out=<n> pack=<dir> run=<dir>"  or  "ERROR FINETUNE_BASE_..." (exit 2)

Standard library only (json + re; the yaml's channel lists are one-line flow lists by
the launcher's own rule), so it runs under the login node's system python3.
"""

import json
import os
import re
import sys


def yaml_list(text, key):
    """The one-line flow list ``    <key>: [...]`` of a makani config yaml."""
    m = re.search(r"^    " + re.escape(key) + r":\s*(\[.*\])\s*$", text, re.M)
    if m is None:
        raise ValueError("no one-line `%s:` list in the yaml" % key)
    return json.loads(m.group(1))


def run_dir_of(ckpt):
    """``<run>/training_checkpoints/<name>.tar`` -> ``<run>``."""
    d = os.path.dirname(os.path.abspath(ckpt))
    if os.path.basename(d) not in ("training_checkpoints", "checkpoints"):
        raise ValueError("checkpoint is not under <run>/training_checkpoints/: %s" % ckpt)
    return os.path.dirname(d)


def check_base(run_cfg, yaml_names):
    """Return (problems, pack). ``run_cfg`` is the base run's parsed config.json."""
    bad = []
    n_out = run_cfg.get("N_out_channels")
    if n_out != len(yaml_names):
        bad.append("base N_out_channels=%r, config has %d channels" % (n_out, len(yaml_names)))
    names = run_cfg.get("channel_names")
    if names != yaml_names:
        if not isinstance(names, list):
            bad.append("base config.json has no channel_names list")
        else:
            i = next((j for j, (a, b) in enumerate(zip(names, yaml_names)) if a != b),
                     min(len(names), len(yaml_names)))
            got = names[i] if i < len(names) else "<missing>"
            want = yaml_names[i] if i < len(yaml_names) else "<missing>"
            bad.append("base channel_names != config: len %d vs %d; first mismatch [%d]: "
                       "%s vs %s" % (len(names), len(yaml_names), i, got, want))
    gmp = run_cfg.get("global_means_path")
    pack = os.path.dirname(os.path.dirname(gmp)) if isinstance(gmp, str) else None
    if pack is None:
        bad.append("base config.json has no global_means_path")
    return bad, pack


def main(argv):
    ckpt, yaml_path, key = argv[1], argv[2], argv[3]
    try:
        if not os.path.isfile(ckpt):
            raise ValueError("checkpoint missing: %s" % ckpt)
        run = run_dir_of(ckpt)
        cfg_path = os.path.join(run, "config.json")
        if not os.path.isfile(cfg_path):
            raise ValueError("no config.json beside the checkpoint's run: %s" % cfg_path)
        text = open(yaml_path).read()
        if not re.search(r"^" + re.escape(key) + r":\s*$", text, re.M):
            raise ValueError("root key %s: not in %s" % (key, yaml_path))
        yaml_names = yaml_list(text, "channel_names")
        run_cfg = json.load(open(cfg_path))
    except (OSError, ValueError) as e:
        print("ERROR FINETUNE_BASE_UNREADABLE: %s" % e)
        return 2
    bad, pack = check_base(run_cfg, yaml_names)
    if pack is not None and not os.path.isfile(os.path.join(pack, "metadata", "data.json")):
        bad.append("base pack has no metadata/data.json: %s" % pack)
    if bad:
        print("ERROR FINETUNE_BASE_MISMATCH: " + " | ".join(bad))
        return 2
    print("FINETUNE_BASE_OK n_out=%d pack=%s run=%s" % (len(yaml_names), pack, run))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
