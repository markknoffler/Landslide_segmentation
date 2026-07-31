# Ablation tables (also in `paper_tpami_v3.pdf` §5.6)

**Open this PDF (not the older `paper_tpami.pdf` from Jul 23):**
`codebase/project_report/gplnet_paper/paper_tpami_v3.pdf`

## Table A — Main ablation (Acc / Prec / Rec / F1 / IoU)

Full row = **primary-server** PS-GPLNet (not compact ablation Full).

| Variant | L4S Acc | Prec | Rec | F1 | IoU | Bijie Acc | Prec | Rec | F1 | IoU |
|---------|---------|------|-----|-----|-----|-----------|------|-----|-----|-----|
| **Full (primary)** | **0.986** | 0.802 | **0.788** | **0.736** | **0.660** | **0.991** | **0.956** | 0.946 | **0.933** | **0.907** |
| w/o Prithvi | 0.981 | 0.616 | 0.750 | 0.530 | 0.458 | 0.975 | 0.961 | 0.754 | 0.728 | 0.717 |
| Path-A only | 0.985 | 0.719 | 0.842 | 0.704 | 0.627 | 0.982 | 0.834 | 0.867 | 0.753 | 0.716 |
| w/o CMB | 0.985 | 0.755 | 0.809 | 0.715 | 0.638 | 0.987 | 0.801 | 0.960 | 0.795 | 0.767 |
| w/o MPEF | 0.983 | 0.691 | 0.853 | 0.689 | 0.611 | 0.987 | 0.860 | 0.918 | 0.812 | 0.782 |
| w/o MAO | 0.986 | 0.756 | 0.805 | 0.717 | 0.641 | 0.987 | 0.933 | 0.898 | 0.870 | 0.835 |
| w/o TTEB | 0.984 | 0.721 | 0.835 | 0.706 | 0.628 | 0.987 | 0.965 | 0.891 | 0.889 | 0.863 |
| w/o FS-gate | 0.984 | 0.736 | 0.807 | 0.703 | 0.626 | 0.988 | 0.948 | 0.916 | 0.896 | 0.869 |

## Table B — Δ vs compact Full (matched protocol)

Compact Full ref: Bijie F1/IoU **0.804/0.773**; L4S **0.693/0.617**.

| Removed | L4S ΔF1 | L4S ΔIoU | Bijie ΔF1 | Bijie ΔIoU |
|---------|---------|----------|-----------|------------|
| w/o Prithvi | −0.163 | −0.159 | −0.076 | −0.057 |
| Path-A only | +0.011 | +0.010 | −0.052 | −0.057 |
| w/o CMB | +0.022 | +0.022 | −0.009 | −0.007 |
| w/o MPEF | −0.005 | −0.006 | +0.007 | +0.009 |
| w/o MAO | +0.024 | +0.025 | +0.065 | +0.062 |
| w/o TTEB | +0.013 | +0.011 | +0.085 | +0.090 |
| w/o FS-gate | +0.010 | +0.009 | +0.092 | +0.096 |

## Table C — Mechanism swaps (compact F1)

| Setting | L4S | Bijie |
|---------|-----|-------|
| Full (compact ref.) | 0.693 | 0.804 |
| w/o FS-gate | 0.703 | 0.896 |
| w/o MPEF | 0.689 | 0.812 |
| Path-A only | 0.704 | 0.753 |

## Figures
`figures/ablation/fig_ablation_{f1_dual,delta_f1,iou_dual,component_help,mechanism_swaps,cross_rank}.png`
