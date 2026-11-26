# Machine Learning-Based RF Power Amplifier Modeling and Digital Predistortion

Behavioural modelling of a measured 200 MHz Doherty RF power amplifier, and
neural digital predistortion (DPD) built on the learned model.

- **Modelling.** Classical memory polynomial baselines against a time-delay MLP,
  compared on identical held-out samples.
- **Predistortion.** A predistorter trained by back-propagating through a frozen
  PA surrogate, evaluated by NMSE and ACPR.
- **Physics analysis.** AM/AM, AM/PM, two-tone IMD3, memory sensitivity and
  virtual interventions on the learned model.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate rf-pa-ml
```

Python 3.10 or newer. No GPU is required — every result in the report was
produced on CPU. `train_dpd.py` accepts `--device cuda` if one is available.

---

## 2. Dataset

`datasets/dpa_200.csv` — the `DPA_200MHz` capture from the
[OpenDPD](https://github.com/lab-emi/OpenDPD) collection.

| Property | Value |
|---|---|
| Samples | 38,400 complex I/Q pairs |
| Sampling rate | 800 MS/s |
| Signal | 10 × 20 MHz carriers, 200 MHz, 64-QAM |
| Device | Doherty power amplifier |
| Columns | `I_in`, `Q_in`, `I_out`, `Q_out` |

**The dataset is committed to this repository.** At 1.9 MB it falls well inside
the "small enough to include" case, and committing it makes the results exactly
reproducible without an external download. No other data is used.

---

## 3. Reproducing the results

Run from the repository root, in this order. Each step writes into `results/`
and `figures/`.

```bash
# 1. Dataset sanity check and AM/AM scatter
python explore_dataset.py

# 2. Linear, memoryless-polynomial and memory-polynomial baselines
#    -> results/baseline_comparison.csv, results/memory_depth_validation.csv
python baselines.py

# 3. Time-delay MLP PA model, memory depths 1/3/5, three seeds each
#    -> results/mlp_comparison.csv, results/all_models_comparison.csv
python train_mlp.py

# 4. Comparison figures (AM/AM, AM/PM, spectrum, accuracy vs depth and size)
python make_figures.py

# 5. Neural DPD through a frozen PA surrogate
#    -> models/dpd_*.pt, results/dpd_comparison_*.csv, figures/dpd_*.png
python train_dpd.py --pa-depth 5 --dpd-depth 5 --hidden 32 --seed 0 \
                    --pa-epochs 200 --epochs 100 --block 2048 --lr 1e-3

# 6. Physics analysis on a trained checkpoint
python analyse_pa_physics.py --checkpoint models/dpd_pa5_dpd5_h32_s0_pe200_e100_sp0_g1_pk1_b2048_lr0.001.pt

# 7. Topology / frequency-response experiments on the same checkpoint
python topology_experiments.py --checkpoint models/dpd_pa5_dpd5_h32_s0_pe200_e100_sp0_g1_pk1_b2048_lr0.001.pt
```

Steps 6 and 7 require step 5 to have produced a checkpoint, or you can point
`--checkpoint` at either of the two committed ones.

Approximate CPU runtimes: steps 1–2 seconds, step 3 a few minutes, step 5
roughly 10–20 minutes at the settings above, steps 6–7 a few minutes.

### Key command-line options for `train_dpd.py`

| Flag | Default | Meaning |
|---|---|---|
| `--pa-depth` | 5 | memory depth of the PA surrogate |
| `--dpd-depth` | 5 | memory depth of the predistorter |
| `--hidden` | 32 | hidden units per layer |
| `--seed` | 0 | random seed |
| `--pa-epochs` | 200 | epochs for the PA surrogate |
| `--epochs` | 100 | epochs for the predistorter |
| `--block` | 2048 | contiguous block length |
| `--batch` | 4 | blocks per update |
| `--lr` | 1e-3 | learning rate |
| `--device` | auto | `cpu`, `cuda`, or `auto` |

Output file names encode the full configuration, so runs with different
settings never overwrite each other.

---

## 4. Directory structure

```
RF_PA_ML/
├── README.md                   this file
├── REPRODUCIBILITY.md          reproducibility declaration and artifact index
├── requirements.txt            pip environment
├── environment.yml             conda environment
├── .gitignore
│
├── datasets/
│   └── dpa_200.csv             OpenDPD DPA_200MHz capture (1.9 MB)
│
├── explore_dataset.py          dataset inspection and AM/AM plot
├── baselines.py                linear, memoryless and memory polynomial
├── train_mlp.py                time-delay MLP PA model
├── make_figures.py             comparison figures
├── train_dpd.py                neural DPD through a frozen PA surrogate
├── analyse_pa_physics.py       AM/AM-PM, two-tone IMD3, memory sensitivity
├── topology_experiments.py     frequency response and static interventions
│
├── models/                     trained DPD checkpoints (.pt)
├── results/                    metric tables (.csv) and waveforms (.npz)
│   ├── physics/                physics-analysis outputs
│   └── topology/               topology-experiment outputs
└── figures/                    all generated plots (.png)
```

Scripts sit at the repository root because each resolves its paths relative to
its own location and imports its siblings directly
(`train_dpd.py` and the two analysis scripts import `train_mlp`).

---

## 5. Evaluation protocol

The split is contiguous and fixed — never shuffled, so temporal structure is
preserved for the memory models.

| Split | Samples | Use |
|---|---|---|
| Train | 23,040 (60%) | fitting parameters |
| Validation | 7,680 (20%) | memory depth, seed and architecture choices |
| Test | 7,680 (20%) | reported numbers only |

Two rules are enforced throughout:

1. **No cross-split windows.** Past-sample windows are built inside each split,
   never across a boundary.
2. **Identical scoring samples.** Models with different memory depths cannot
   predict the first samples of a split, so the same leading samples are
   discarded for every model before scoring.

Metrics are NMSE (dB, lower is better) and ACPR (dB, read against the measured
PA output rather than "lower is better").

---

## 6. License and attribution

Dataset: `DPA_200MHz` from OpenDPD (https://github.com/lab-emi/OpenDPD).
Please cite OpenDPD if you use the data.
