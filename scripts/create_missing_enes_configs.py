"""Generate all missing en-es experiment configs by mirroring cs-de configs."""
import yaml
from pathlib import Path

configs_dir = Path("experiments/configs")

# cs-de experiments to mirror (excluding INT4, LRP, M5, M1_8_fullft)
missing = [
    'B1_int8', 'B3',
    'I2_8', 'I2_12', 'I2_16',
    'I3_8_int8', 'I3_12_int8', 'I3_16_int8',
    'I4_8_int8', 'I4_12_int8', 'I4_16_int8',
    'I5_threshold_int8',
    'M2_8', 'M2_12', 'M2_16',
    'M3_8_int8', 'M3_12_int8', 'M3_16_int8',
    'M4_8_int8', 'M4_12_int8', 'M4_16_int8',
]

for eid in missing:
    # Read the cs-de config
    csde_path = configs_dir / f"{eid}.yaml"
    if not csde_path.exists():
        print(f"WARNING: {csde_path} not found, skipping")
        continue

    with open(csde_path) as f:
        cfg = yaml.safe_load(f)

    # Modify for en-es
    new_id = f"{eid}_enes"
    cfg["experiment_id"] = new_id
    cfg["description"] = cfg["description"].replace("Czech", "English").replace("German", "Spanish") + " (English-Spanish)"
    cfg["lang_pair"] = "en-es"

    # Update IFR scores path if present
    pruning = cfg.get("pruning", {})
    if pruning.get("scores_path"):
        pruning["scores_path"] = "experiments/results/ifr_scores_enes.json"

    out_path = configs_dir / f"{new_id}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"Created {out_path.name}")

print(f"\nDone: {len(missing)} en-es configs created")
