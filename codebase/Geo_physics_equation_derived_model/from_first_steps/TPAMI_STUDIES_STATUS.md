# TPAMI studies status (honest ledger)

## Why a local retrain happened
Final **metrics** (0.933 Bijie / 0.736 L4S) were already on disk as CSVs under the canonical dirs in `RESULTS_AND_CHECKPOINTS.md`.  
Final **weights** were **not**. Inference studies (t-SNE hook, boundary masks, noise decay) need a loadable `.pt`.  
Legacy `prev_legacy_results/**/best.pt` is the wrong model (no physics/MPEF).

So a local B4 PS-GPLNet Bijie retrain was started **only** to produce study checkpoints — not to invent the main-table scores.

## Canonical paper numbers (unchanged)
- Bijie: `outputs_absolute_final_fully_novel_complete` (= `outputs_final_bijie`) → F1 **0.933**, IoU **0.907**, epoch **92**
- L4S: `outputs_step3_l4s` (= `outputs_final_l4s`) → F1 **0.736**, IoU **0.660**, epoch **70**

## Local study checkpoint
- `outputs_tpami_local_bijie/checkpoint/best.pt` — epoch **98**, val F1 **0.912**, IoU **0.886**
- Studies: `outputs_tpami_studies_final/` (current); interim epoch-4 run stashed at `prev_legacy_results/outputs_tpami_studies_interim/`

## Paper
- Theory (LCE, FS Lipschitz, MPEF entropy identity): `paper_tpami.tex`
- Empirical studies section + figures: updated in `paper_tpami_v2.pdf`
- Main table still cites cluster CSV finals, not the local 0.912 run
