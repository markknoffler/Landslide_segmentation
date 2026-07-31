# Ablation Study Plan — PS-GPLNet (ICLR / ICCV / ICML track)

## Venue positioning (short answer)

| Question | Answer |
|----------|--------|
| Is this “too application” for ICLR/ICML/ICCV? | **Risk if framed as landslide mapping only.** Frame as **physics-closed multimodal dense prediction**: LCE operator family + MPEF energy routing; Bijie/L4S are **anisotropic stress tests**, not the contribution title. |
| Must we prove **Universal Approximation Theorem**? | **No.** UAT is not a submission requirement. Local Lipschitz / compactness / MPEF energy identity (already in paper) are the right theory depth. |
| Must we train **other domains/equations**? | **Not mandatory** for a first conference submission if (1) contribution is the **operator class**, (2) mechanism ablations isolate each block, (3) two datasets + strong CV baselines. A **light second domain later** (e.g. flood / building damage with a different continuum prior) strengthens journal extension—not a blocker for ICCV/ICLR if ML framing is sharp. |
| What is still missing for top venues? | **Component ablations** (this doc), fair protocol, failure analysis, compute reporting. Theory + method already exist. |

## Template lessons from `papers/perfect_ablation_study.pdf` (HFM)

Mirror their table **roles**, not their task:

1. **Table A — Component effectiveness** (like their Table 2): progressive ✓/✗ of CHA/PO/DS → our CMB / FS-gate / MAO / TTEB / MPEF / Prithvi on **both** Bijie and L4S with Acc/Prec/Rec/**F1**/**IoU**.
2. **Table B — Mechanism swaps** (not just drop): MPEF vs mean fuse vs path-A-only; FS-gate on vs off.
3. **Table C — Capacity / protocol** (like their Table 4): compact B0 vs B4 if VRAM allows; same epochs/seed/optimizer.
4. **Split “hard vs easy”**: treat **L4S as harder** (lower F1 ceiling ~0.74), **Bijie as easier** (~0.93) — report both columns like Difficult/Easy.

## Canonical metric anchors (monitor every run)

| Dataset | Expected full-model ballpark | Source |
|---------|------------------------------|--------|
| Bijie | F1 **~0.91–0.93**, IoU **~0.88–0.91** | cluster CSV 0.933/0.907; local B4 0.912/0.886 |
| L4S | F1 **~0.70–0.74**, IoU **~0.62–0.66** | cluster 0.736/0.660 |

If a **full** compact run is far below (e.g. Bijie F1 < 0.85 after ≥40 epochs) → bug / bad flags / learning-rate / data path.

## Ablation variants (train order)

All use **GPU 1**, conda `deeplearning`, **`--compact`**, AMP, batch sized for 12 GB, fixed seed **42**, same loss/optimizer as paper.

| ID | Name | Change |
|----|------|--------|
| A0 | `full` | Reference PS-GPLNet |
| A1 | `no_fs_gate` | `mechanistic_gating=False` |
| A2 | `no_mpef` | Equal-weight blend + 1×1 (no FE softmax) |
| A3 | `no_cmb` | CNN projection only (no physics resonance gate) |
| A4 | `no_mao` | 1×1 concat fusion instead of MAO-GeoEGCA |
| A5 | `no_tteb` | Present-only mix skips (no lattice/δ attention) |
| A6 | `no_prithvi` | Zero foundation stream (spatial zeros) |
| A7 | `path_a_only` | Decoder path A only (no dual/MPEF) |

Phase 1 (now): **Bijie** A0→A7, **40 epochs** each (extend winners to 100 if time).  
Phase 2: **L4S** same matrix.  
Phase 3: write tables into `paper_tpami_v3.pdf`.

## Hardware

- **GPU 1**: RTX 3060 **12 GB** (empty).
- Do **not** use GPU 0 (occupied).
- Compact: EfficientNet-B0, `C=32`, LoRA r≤4.

## Outputs

`from_first_steps/outputs_ablation/<dataset>/<variant>/results/epoch_metrics.csv`  
Master summary: `outputs_ablation/ABLATION_SUMMARY.csv` + `ABLATION_STATUS.md`
