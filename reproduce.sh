#!/usr/bin/env bash
#
# reproduce.sh — regenerate every result and figure in the report.
#
# Usage:
#   ./reproduce.sh              full run, as used for the reported numbers
#   ./reproduce.sh --quick      fast sanity check (few epochs, numbers will differ)
#   ./reproduce.sh --install    pip install -r requirements.txt first, then full run
#
# Run from the repository root. Writes into results/ and figures/ and models/.

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
QUICK=0
INSTALL=0

for arg in "$@"; do
  case "$arg" in
    --quick)   QUICK=1 ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- checks
command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: $PY not found." >&2; exit 1; }
[ -f datasets/dpa_200.csv ] || { echo "ERROR: datasets/dpa_200.csv missing. Run from the repo root." >&2; exit 1; }

if [ "$INSTALL" -eq 1 ]; then
  say "Installing dependencies"
  "$PY" -m pip install -r requirements.txt
fi

say "Environment"
"$PY" - <<'EOF'
import sys
print("python  ", ".".join(map(str, sys.version_info[:3])))
for m in ("numpy", "pandas", "scipy", "matplotlib", "torch"):
    try:
        print(f"{m:9s}", __import__(m).__version__)
    except ImportError:
        print(f"{m:9s} MISSING"); raise SystemExit(f"Install {m}: pip install -r requirements.txt")
EOF

mkdir -p results figures models

# ---------------------------------------------------------------- config
PA_DEPTH=5
DPD_DEPTH=5
HIDDEN=32
SEED=0
BLOCK=2048
LR=0.001

if [ "$QUICK" -eq 1 ]; then
  PA_EPOCHS=20; DPD_EPOCHS=20
  echo "QUICK MODE: reduced epochs — results will not match the report."
else
  PA_EPOCHS=200; DPD_EPOCHS=100
fi

CKPT="models/dpd_pa${PA_DEPTH}_dpd${DPD_DEPTH}_h${HIDDEN}_s${SEED}_pe${PA_EPOCHS}_e${DPD_EPOCHS}_sp0_g1_pk1_b${BLOCK}_lr${LR}.pt"

# ---------------------------------------------------------------- pipeline
say "1/7  Dataset inspection"
"$PY" explore_dataset.py

say "2/7  Polynomial baselines"
"$PY" baselines.py

say "3/7  Time-delay MLP PA model"
"$PY" train_mlp.py

say "4/7  Comparison figures"
"$PY" make_figures.py

say "5/7  Neural DPD through a frozen PA surrogate"
"$PY" train_dpd.py \
  --pa-depth "$PA_DEPTH" --dpd-depth "$DPD_DEPTH" --hidden "$HIDDEN" \
  --seed "$SEED" --pa-epochs "$PA_EPOCHS" --epochs "$DPD_EPOCHS" \
  --block "$BLOCK" --lr "$LR"

[ -f "$CKPT" ] || { echo "ERROR: expected checkpoint not found: $CKPT" >&2; exit 1; }

say "6/7  Physics analysis"
"$PY" analyse_pa_physics.py --checkpoint "$CKPT"

say "7/7  Topology experiments"
"$PY" topology_experiments.py --checkpoint "$CKPT"

# ---------------------------------------------------------------- summary
say "Done"
echo "Checkpoint : $CKPT"
echo "Results    : results/"
echo "Figures    : figures/"
echo
echo "Headline numbers:"
"$PY" - <<'EOF'
import pandas as pd, glob
try:
    b = pd.read_csv("results/baseline_comparison.csv")
    for _, r in b.iterrows():
        n = r["Test_NMSE_dB"]
        ns = "      -" if pd.isna(n) else format(n, "7.2f")
        print(f"  {r['Model']:<34s} NMSE {ns}  ACPR {r['Test_ACPR_dB']:7.2f}")
except Exception as e:
    print("  (baseline table unavailable:", e, ")")
for f in sorted(glob.glob("results/dpd_comparison_*.csv")):
    print(f"\n  {f}")
    for _, r in pd.read_csv(f).iterrows():
        print(f"    {r['Model']:<26s} NMSE {r['NMSE_to_ideal_dB']:7.2f}  ACPR {r['ACPR_dB']:7.2f}")
EOF

echo
echo "NOTE: DPD metrics are surrogate-model predictions, not measured hardware results."
