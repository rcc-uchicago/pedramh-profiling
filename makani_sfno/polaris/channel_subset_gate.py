"""Converter<->config gate for a config that trains on a SUBSET of the pack.

The full-width gate (polaris_sfno_alldata_{smoke,full}.pbs) asserts
`channel_names == TARGET_CHANNELS` exactly, because same-width name drift maps
channels to the wrong slots with no error. Port F
(`polaris_makani_ace2_ports_handoff.md` §6a) deliberately trains on 99 of the
pack's 101 channels, so that equality no longer holds -- but the protection
must. This gate keeps it: the config names what it drops, and everything else
must still be TARGET_CHANNELS, in order, name for name.

    python channel_subset_gate.py <rendered.yaml> <root_key>
      -> "CHANNEL_SUBSET_GATE_OK ..." or "ERROR CHANNEL_SUBSET_DRIFT ..." (exit 2)

Pure python + yaml; imports the converter only for its channel lists.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_subset(cfg: dict, target: list, n_state: int, n_diag: int) -> list[str]:
    """Return a list of problems (empty = OK) for a config with
    `dropped_channel_names`, against the converter's full-width contract."""
    dropped = list(cfg.get("dropped_channel_names") or [])
    names = list(cfg.get("channel_names") or [])
    bad = []
    unknown = [d for d in dropped if d not in target]
    if unknown:
        bad.append(f"dropped_channel_names not in TARGET_CHANNELS: {unknown}")
    if len(set(dropped)) != len(dropped):
        bad.append(f"dropped_channel_names has duplicates: {dropped}")
    diag = target[len(target) - n_diag:]
    if any(d in diag for d in dropped):
        bad.append(f"cannot drop the diagnostic channel(s) {diag}")
    want = [c for c in target if c not in set(dropped)]
    if names != want:
        i = next((j for j, (a, b) in enumerate(zip(names, want)) if a != b),
                 min(len(names), len(want)))
        got_i = names[i] if i < len(names) else "<missing>"
        want_i = want[i] if i < len(want) else "<missing>"
        bad.append(f"channel_names != TARGET_CHANNELS minus dropped: len {len(names)} vs "
                   f"{len(want)}; first mismatch [{i}]: {got_i} vs {want_i}")
    want_state = n_state - sum(1 for d in dropped if d in target[:n_state])
    if cfg.get("n_state_channels") != want_state:
        bad.append(f"n_state_channels={cfg.get('n_state_channels')!r}, expected {want_state}")
    if cfg.get("n_diagnostic_channels") != n_diag:
        bad.append(f"n_diagnostic_channels={cfg.get('n_diagnostic_channels')!r}, expected {n_diag}")
    return bad


def main(argv: list[str]) -> int:
    import yaml

    path, key = argv[1], argv[2]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_e3sm_to_makani_alldata as conv

    cfg = yaml.safe_load(open(path))[key]
    if not cfg.get("dropped_channel_names"):
        print("CHANNEL_SUBSET_GATE_SKIP: no dropped_channel_names in config")
        return 0
    bad = check_subset(cfg, list(conv.TARGET_CHANNELS), conv.N_STATE, len(conv.DIAG_CHANNELS))
    for key_, want in (("forcing_channel_names", conv.FORCING_CHANNELS),
                       ("n_forcing_channels", conv.N_FORCING)):
        if cfg.get(key_) != want:
            bad.append(f"{key_}: yaml={cfg.get(key_)!r} converter={want!r}")
    if bad:
        print("ERROR CHANNEL_SUBSET_DRIFT: " + " | ".join(bad))
        return 2
    print(f"CHANNEL_SUBSET_GATE_OK: {len(cfg['channel_names'])} of {conv.N_TARGET} channels, "
          f"dropped {cfg['dropped_channel_names']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
