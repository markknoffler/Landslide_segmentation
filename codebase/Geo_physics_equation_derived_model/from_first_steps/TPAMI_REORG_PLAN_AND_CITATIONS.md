# TPAMI Reorganization Plan, Theory Scaffold, and Citation Ledger

**Manuscript target:** *IEEE Transactions on Pattern Analysis and Machine Intelligence* (TPAMI)  
**Working title (proposed):** *Physics-Closed Multimodal Representation Learning via Latent Continuum Equivalency: Factor-of-Safety Cells, Equilibrium Path Fusion, and Dense Segmentation*  
**Application stress test (kept, not discarded):** binary landslide segmentation on Bijie + Landslide4Sense  
**Code root:** `codebase/Geo_physics_equation_derived_model/from_first_steps/`  
**Authors:** Samreedh Bhuyan; mentor Dr. Anil Earnest (KIIT / CSIR-4PI)  
**Hardware constraint for remaining work:** GPU0 = NVIDIA RTX 4000 Ada (20 GB) preferred; GPU1 = RTX 3060 (12 GB) often busy. Env: `conda activate deeplearning` — **install only missing packages; do not upgrade existing torch/CUDA stack.**

This document answers four questions with research-backed reasoning:

1. Is a TPAMI submission *possible* while keeping landslide segmentation as the problem statement?  
2. How must the paper be *reorganized* so reviewers read a **CV / pattern-analysis method paper**, not a remote-sensing application note?  
3. What **theorems, proofs, algorithms, and foundational citations** will the manuscript use?  
4. What **additional compute** (if any) is still required on a 20 GB card, given that main training logs already exist?

---

## 0. Executive verdict (honest)

**Yes, it is possible in principle — but only if the paper’s primary claim is methodological and theoretical, with landslide segmentation as the *validation domain*, not the *contribution*.**

TPAMI desk-rejects and early-rejects papers whose novelty sentence needs the phrase “we apply X to landslides.” Accepted TPAMI-adjacent physics/geometry papers (Gravityformer, Virtual Normal, topological PH losses, sparse-coding fusion, Lipschitz segmentation certificates) all share the same pattern:

- Name a **general CV / learning mechanism** in one sentence.  
- Give **definitions → lemmas/propositions → theorems → algorithms** that make the mechanism precise.  
- Prove or argue **stability / continuity / optimality / interpretability** properties.  
- Use **one or more hard vision tasks** as evidence that the mechanism travels.  
- Provide **ablations that isolate the mechanism**, not just leaderboard gains.

Your architecture already has the raw material for that pattern:

| Mechanism in code | TPAMI-facing name |
|-------------------|-------------------|
| Shared FS cell from inlet → decoder → MPEF (`pixel_cell.py`, `latent_cell.py`, `mechanistic_path_fusion.py`) | **Latent Continuum Equivalency (LCE)** of a physical continuum operator across representation scales |
| `positive_scale` = `exp(clip(w))` | **Bounded material reparametrization** guaranteeing finite FS and Lipschitz-friendly Jacobians |
| CMB resonance bridge | **Complementary Modality Bridge** as co-activation gated hybridization |
| MAO-GeoEGCA + TTEB | Physics-anchored multimodal attention / skip lattice |
| Dual PhysicsDecoder + PGDI | Dual mechanistic decode paths with physics-gated deep injection |
| MPEF softmax of failure energies + entropy | **Mechanistic Path Equilibrium** as energy-minimizing path routing |

**What TPAMI will not accept:** “PS-GPLNet beats DeepLabV3+ on Bijie/L4S.”  
**What TPAMI can accept:** “We introduce a closed-loop physical continuum operator that can be injected into high-dimensional multimodal networks with provable boundedness/Lipschitz properties, and we show it improves dense prediction under anisotropic topographic–spectral fusion; landslides are the stress test.”

With only 20 GB + 12 GB local VRAM and **completed epoch metrics**, you do **not** need A100-scale retraining to *write* the TPAMI draft. You *do* need a short list of **inference-only / low-batch** analyses (Section 6) and a **theory section that is currently missing** from `gplnet_paper/`.

---

## 1. What I read (papers/ + external TPAMI writing research)

### 1.1 Local `papers/` corpus (read/extracted)

| File | Role for our writing |
|------|----------------------|
| `Gravityformer_arxiv.pdf` | TPAMI-style **physics *inside* attention**; AdaGravity closed-form prior; Hadamard physics×attention; algorithm table; multi-city generalization. Closest **philosophy** to “physical law constrains fusion, not just loss.” |
| `Virtual_Normal_arxiv.pdf` | TPAMI classic: **geometric constraint as first-class loss/structure**; shows pixel metrics alone are insufficient; high-order geometry is the claim. Template for “FS gate > pixel IoU narrative.” |
| `Topological_Loss_Segmentation_arxiv.pdf` | TPAMI: **differentiable discrete prior** (persistent homology) with careful gradient story. Template for “how to claim a discrete/physical quantity is usable in backprop.” |
| `Sparse_Coding_Image_Fusion_arxiv.pdf` | TPAMI: fusion as **unfolded optimization** (\(\ell_0\) CSC); interpretability of unique/common features. Template for **MPEF as energy minimization**, not a free gate. |
| `Physics_Informed_Disentanglement_arxiv.pdf` | TPAMI: physics-informed **disentanglement** in generative nets. Template for CMB shared vs unique modality factors. |
| `Lipschitz_Segmentation_Certificates_arxiv.pdf` | Definitions/Propositions on **Lipschitz certificates for segmentation**. Direct scaffold for FS-cell Lipschitz / robustness claims. |
| `Layer_Lipschitz_Multimodal_arxiv.pdf` | Layer-wise Lipschitz modulation for multimodal robustness; theorem-led architecture design. |
| `KKT_Hardnet_Physics_Constraints_arxiv.pdf` | Hard nonlinear constraints via differentiable projection / KKT. Contrast: we use **soft but bounded** FS cells, not hard projection. |
| `PDE_Constrained_Segmentation_arxiv.pdf` | Segmentation as PDE-constrained inverse problem. Contrast: we embed continuum FS as **architectural cells**, not only residual PDE losses. |
| `Coastal_Robust_UNet_arxiv.pdf` | Lipschitz continuity of composite geometric/physics losses for coastline segmentation. Closest **task family** (geospatial boundary segmentation). |
| `dual_stream.pdf`, `BiFusion.pdf`, hierarchical landslide PDF | Application / dual-stream baselines — cite as **empirical neighbors**, not theoretical peers. |
| `2102.04306.pdf` (TransUNet), `2105.15203.pdf` (SegFormer), `2309.16653.pdf` (Prithvi-related) | Standard CV / EO foundation baselines. |
| `2204.01807v1.pdf` | Additional remote-sensing DL context. |

### 1.2 External research on *how TPAMI wants papers written*

Synthesized from TPAMI submission / readiness guides (2025–2026 editorial practice summaries):

- Contribution must be **methodologically nameable in one sentence** without “more experiments.”  
- Application-only gains on a niche dataset are **out of scope** unless the method clearly travels as pattern analysis / vision.  
- Journal version needs **theory and/or mechanism ablations** that conference pages cannot hold.  
- Ablations must **isolate the proposed mechanism** (FS cell / MPEF / CMB), not only swap backbones.  
- Reproducibility (code, protocols, seeds) is expected.  
- Typical regular article ~14 pages IEEE two-column including refs; density matters more than page count.

### 1.3 Figures already generated for theory storytelling

On **GPU0** (`CUDA_VISIBLE_DEVICES=0`), without upgrading the env, we wrote:

- `tpami_assets/figures/fig_bounded_positive_scale.png`  
- `tpami_assets/figures/fig_fs_landscape.png`  
- `tpami_assets/figures/fig_mpef_routing.png`  
- `tpami_assets/figures/fig_fs_local_lipschitz_hist.png`  

These are **cheap scientific visuals** for the theory section (not Neist qualitative panels).

---

## 2. Is TPAMI possible for *this* problem statement?

### 2.1 The real risk

Landslide segmentation is **domain-specific**. TPAMI reviewers will ask:

> “Why is this not better suited to *IEEE TGRS* / *ISPRS* / *Remote Sensing of Environment*?”

Answer that works only if true:

> “Because the contribution is a **general operator family** for injecting continuum mechanical balance laws into multimodal dense predictors, with proofs of boundedness and an equilibrium fusion principle. Landslide inventory mapping is chosen because it forces **anisotropic topographic–spectral fusion** under severe class imbalance — a harder CV fusion regime than RGB–D, where depth and appearance are strongly correlated.”

### 2.2 The reframing that makes acceptance *plausible*

Treat the paper as a **three-layer claim**:

1. **CV claim (primary):** Physics-closed continuum cells + energy-equilibrium dual-path fusion for multimodal dense prediction.  
2. **Theory claim (required for TPAMI):** Latent Continuum Equivalency + Lipschitz/boundedness of FS cells + MPEF as unique energy minimizer under simplex constraints.  
3. **Application claim (secondary but kept):** SOTA / competitive landslide segmentation on Bijie + L4S under a unified protocol.

You **do not throw away** segmentation results. You **demote** them from “the paper is about landslides” to “the stress-test suite that proves the CV mechanism.”

### 2.3 Realistic acceptance odds (not marketing)

| Venue | Fit if theory section is strong | Fit if theory is thin |
|-------|----------------------------------|------------------------|
| TPAMI | Stretch-but-possible | Unlikely |
| CVPR / ICCV / ECCV (long) | Stronger short-term | Possible as method paper |
| NeurIPS / ICML | Needs broader ML theory + more tasks | Hard |
| IEEE TGRS / ISPRS | Highest probability | High |

**Recommendation:** Write the manuscript *as if for TPAMI* (theory-first), then **submit first to a top CV conference** if GPU ablations are incomplete; use TPAMI as the **journal extension** after conference review feedback. That matches how Gravityformer / Virtual Normal matured. Still: organize now for TPAMI so you do not rewrite twice.

---

## 3. Master paper organization (TPAMI-facing outline)

Use IEEE Transactions two-column. Keep landslide results, but rename the narrative.

### Proposed section map

1. **Abstract** — method + theory + one stress-test number (already rewritten in `gplnet_paper`; keep NeurIPS/TPAMI tone, not “this report documents…”).  
2. **Introduction** — open with **multimodal dense prediction under anisotropic modalities**, not landslide death tolls. Landslides appear in paragraph 2 as the motivating stress test. End with contributions as continuous prose (no bullet laundry list).  
3. **Related Work** (deep, TPAMI style)  
   - Dense segmentation CNNs/transformers  
   - Multimodal fusion (early/late/attention; BiFusion; dual-stream)  
   - Physics-informed ML in vision (Gravityformer; Virtual Normal; PDE-constrained seg; topological losses)  
   - Lipschitz / certified robust segmentation  
   - EO foundation models (Prithvi) as *one stream*, not the novelty  
   - Landslide DL as **application literature** (short)  
4. **Problem Formulation** — multimodal anisotropic fusion; define continuum FS operator \(\mathcal{C}_{\mathrm{FS}}\).  
5. **Method** — architecture (pentastream, CMB, MAO, TTEB, dual decoder, PGDI, MPEF) with equations.  
6. **Theoretical Analysis** (**new; TPAMI heart**) — Definitions, Lemmas, Theorems, Remarks (Section 4 below).  
7. **Algorithms** — Algorithm 1 Forward; Algorithm 2 Training objective; Algorithm 3 (optional) robustness evaluation.  
8. **Experiments**  
   - Setup + baselines (keep tables)  
   - Main results (Bijie / L4S) as stress test  
   - **Mechanism ablations** (FS on/off, MPEF vs learned gate, CMB on/off) — critical  
   - Robustness / boundary / latent analysis (low-VRAM)  
9. **Discussion / Limitations**  
10. **Conclusion**

Preserve existing assets in `codebase/project_report/gplnet_paper/` — **do not destroy** current LaTeX; evolve it by adding Section 6 (Theory) and reframing Intro/Related Work.

---

## 4. Theory scaffold (what we will prove / claim)

Write in LaTeX with `amsthm` environments: `definition`, `lemma`, `proposition`, `theorem`, `corollary`, `remark`, `proof`.

### 4.1 Foundational statements we *build upon* (cite, do not reinvent)

These are the “foundation stones” for chaining proofs (as you asked):

| ID | Statement (paraphrased) | Source paper | How we use it |
|----|-------------------------|--------------|---------------|
| F1 | A map \(f\) is \(L\)-Lipschitz if \(\|f(x)-f(y)\|\le L\|x-y\|\). Composition multiplies layer Lipschitz constants. | Lipschitz segmentation certificates; Layer-Lipschitz multimodal | Bound FS gate + Conv composition |
| F2 | Softmax weights on energies are the unique maximizer of \(\mathbf{w}^\top\mathbf{e}+\mathcal{H}(\mathbf{w})\) on the simplex (entropy-regularized linear objective). | Standard convex analysis; Sparse-coding fusion papers use analogous energy arguments | Prove MPEF is an equilibrium, not an ad-hoc gate |
| F3 | Softplus / clipped-exp maps keep parameters in a compact positive set → prevent blow-up | Gravityformer Softplus non-negativity; our `positive_scale` | Material parameters stay in \([e^{-8}, e^{8}]\) |
| F4 | Physics-informed attention = learned attention \(\odot\) closed-form physical interaction | Gravityformer AdaGravity | Analogy for MAO physics anchor + MPEF FE routing |
| F5 | High-order geometric constraints beat pure pixel losses for 3D/structure fidelity | Virtual Normal | Motivate boundary/FS alignment metrics beyond IoU |
| F6 | Differentiable discrete/topological priors can supply training signal | Topological Loss (PH) | Justify differentiable FS gates as “structure priors” |
| F7 | Fusion can be an unfolded optimization, not a black-box MLP | Sparse Coding Image Fusion | Position MPEF as closed-form routing of a path energy |
| F8 | Soft PDE / physics residual constraints improve segmentation under geometry | PDE-constrained segmentation; Coastal Robust U-Net | Contrast: we use **hard architectural closure**, soft loss only as auxiliary |

### 4.2 Our definitions (to introduce formally)

**Definition 1 (Infinite-slope continuum operator).**  
For proxies \((\alpha,h,m)\) and positive materials \((c,\phi,\gamma,w_m)\),
\[
\mathrm{FS}(\alpha,h,m)=\frac{c+\phi h\cos^2\alpha}{\gamma h\sin\alpha\cos\alpha+w_m m+\varepsilon},\quad
\mathrm{FE}=\mathrm{ReLU}(1-\mathrm{FS}).
\]

**Definition 2 (Bounded material reparametrization).**  
\(c=\exp(\mathrm{clip}(w_c;-8,8))\) (likewise \(\phi,\gamma,w_m\)) as in `physics/params.py`.

**Definition 3 (Pixel mechanistic cell).**  
\(g_{\mathrm{pix}}(x;\alpha,h,m)=f(x)\odot\sigma(\psi-\mathrm{FS}(\alpha,h,m))\).

**Definition 4 (Latent Continuum Equivalency).**  
A network family \(\{T_\ell\}\) at scales \(\ell\) satisfies LCE if each \(T_\ell\) applies an operator in the same continuum class \(\mathcal{C}_{\mathrm{FS}}\) (same algebraic skeleton, possibly different parameters/resolutions), so physical failure energy remains a **commensurable** quantity across depths.  
*(This is the named novelty — analogous to Gravityformer’s consistent gravity law across layers.)*

**Definition 5 (Mechanistic Path Equilibrium Fusion).**  
Given path logits \(O_A,O_B\) and energies \(\mathrm{FE}_A,\mathrm{FE}_B\),
\[
[w_A,w_B]=\mathrm{softmax}([\mathrm{FE}_A,\mathrm{FE}_B]),\quad
O=w_A O_A+w_B O_B+\mathrm{Conv}_{1\times1}([O_A\|O_B]).
\]

### 4.3 Lemmas / Theorems to write in the paper

**Lemma 1 (Compactness of materials).**  
Under Definition 2, \(c,\phi,\gamma,w_m\in[e^{-8},e^{8}]\). *Proof:* immediate from clip+exp.

**Lemma 2 (FS is \(C^\infty\) and locally Lipschitz on the training domain).**  
On \(\alpha\in[\alpha_{\min},\alpha_{\max}]\subset(0,\pi/2)\), \(h\ge h_{\min}>0\), \(m\in[0,1]\), denominator \(\ge\varepsilon>0\), FS is smooth; gradients w.r.t. \((\alpha,h,m)\) are bounded on this compact set. *Proof:* rational function with non-vanishing denominator; continuous on compact ⇒ Lipschitz. Cite F1.

**Lemma 3 (Sigmoid FS gate is Lipschitz).**  
\(x\mapsto\sigma(\psi-\mathrm{FS}(x))\) is Lipschitz on the same domain with constant \(\le\tfrac14\mathrm{Lip}(\mathrm{FS})\) because \(\mathrm{Lip}(\sigma)\le 1/4\).

**Proposition 1 (Pixel cell does not explode under parameter learning).**  
Composition of bounded materials + Lemma 2 + 1×1 conv implies the mechanistic cell has finite local Lipschitz constant w.r.t. inputs. Empirically supported by `fig_fs_local_lipschitz_hist.png`.

**Theorem 1 (Latent Continuum Equivalency — qualitative theorem).**  
If every stage applies Definition 3 / latent FS ratios from the same algebraic class \(\mathcal{C}_{\mathrm{FS}}\), then path-wise failure energies \(\mathrm{FE}_A,\mathrm{FE}_B\) remain **comparable** (same units / same formula), enabling Definition 5.  
*Proof sketch:* commensurability of FE across paths follows from identical FS skeleton + shared positivity constraints (Lemmas 1–2). Without LCE (e.g., arbitrary learned gates), path weights lack a physical common currency.

**Theorem 2 (MPEF as entropy-regularized energy routing).**  
On the simplex \(\Delta^1=\{w_A+w_B=1,w\ge0\}\), the map
\[
\mathbf{w}^\star=\mathrm{softmax}(\mathbf{FE})
\]
is the unique maximizer of \(\mathbf{w}^\top\mathbf{FE}+\mathcal{H}(\mathbf{w})\). The training regularizer \(\mathcal{R}_{\mathrm{MPEF}}=-\mathbb{E}[\mathcal{H}(\mathbf{w})]\) therefore contracts toward decisive equilibria. Cite F2 / sparse-coding energy fusion analogy (F7).

**Corollary 1 (Contrast to DiGATe GateFuse).**  
Scalar learned gates without FE are not solutions of the above energy problem; they are unconstrained fusion parameters. This is the **architectural novelty statement**.

**Theorem 3 (Optional, if we add Lipschitz-by-design experiments).**  
Under spectral-norm constraints on \(1\times1\) maps and Lemmas 1–3, a global Lipschitz upper bound of the FS-gated branch can be stated multiplicatively (F1). Use for robustness decay plots.

### 4.4 What we will *not* falsely claim

- We will **not** claim a closed-form global optimality proof for the full nonconvex U-Net training.  
- We will **not** claim hard KKT constraint satisfaction (that is KKT-Hardnet’s claim).  
- We will **not** claim PDE residual zero for continuum mechanics — FS is a **reduced-order continuum prior**, Taylor-stabilized for backprop.

Honesty increases TPAMI trust.

---

## 5. Algorithms to include (TPAMI style)

Mirror Gravityformer’s “Table: Algorithm” and Virtual Normal’s curriculum algorithm.

**Algorithm 1 — Forward pass of PS-GPLNet (physics-closed).**  
Inputs → proxies \((\alpha,h,m)\) → five encoders → CMB → MAO/TTEB → dual PhysicsDecoder+PGDI → MPEF → logits + \(\mathcal{R}_{\mathrm{MPEF}}\).

**Algorithm 2 — Training.**  
Deeply supervised Tversky on main/aux + \(\lambda_{\mathrm{reg}}\sum_h\mathcal{R}_{\mathrm{MPEF}}^{(h)}\); Adam; clip_grad; bounded materials always on.

**Algorithm 3 — Low-VRAM robustness probe (for paper experiments).**  
For noise levels \(\sigma\in\Sigma\): perturb RGB and/or DEM → infer → record IoU/F1/BF-score.

Use `algorithmicx` / `algpseudocode` if available; else numbered paragraphs (as in current paper when `algorithm.sty` missing).

---

## 6. Experiments plan under 20 GB VRAM (what you still need)

### 6.1 Already done (reuse — do not retrain from scratch)

- Bijie + L4S epoch metrics in `outputs_absolute_final_fully_novel_complete`, `outputs_final_*`, `outputs_step3_l4s`  
- Comparative baseline tables already in `gplnet_paper/assets/comparison_metrics.json`  
- Qualitative panels already curated for the report  
- Legacy checkpoints exist under `prev_legacy_results/**/best.pt` (verify which match the **final** architecture before claiming them in TPAMI)

### 6.2 Must-have for TPAMI *without* A100s

All runnable on **GPU0** with batch size 1–4; prefer inference-only.

| ID | Experiment | Why TPAMI needs it | Compute |
|----|------------|--------------------|---------|
| E1 | **Mechanism ablation table**: full / no-FS-gate / no-MPEF (replace with 1×1 gate) / no-CMB / no-Prithvi | Isolates the claimed mechanism | If weights for each variant exist: inference only. Else: **compact** finetune 10–20 epochs on Bijie, batch 2–4 |
| E2 | Boundary F-score + Hausdorff (use `tpami_eval_utils.py`) | Virtual Normal lesson: structure > pixel | Inference on val set |
| E3 | Noise robustness decay (RGB noise, DEM downsample) | Lipschitz / stability story | Inference |
| E4 | Path-weight maps \(w_A,w_B\) visualization | Interpretable physics routing (Gravityformer-style) | Inference hooks |
| E5 | FS / FE overlay vs mask | Physical interpretability figure | Inference |
| E6 | Optional t-SNE of bottleneck features (FS-on vs FS-off) | Latent continuum visualization | Inference + CPU t-SNE |

### 6.3 Nice-to-have (only if time)

- Second domain sanity check (e.g., crack / flood / building damage segmentation) — **not required** for first TPAMI draft; huge for acceptance odds later.  
- Lipschitz-by-design spectral normalization — only if Theorem 3 is pursued seriously.

### 6.4 Checkpoint truth check (important)

Final result folders currently contain **metrics**, while `best.pt` lives under `prev_legacy_results/`. Before any TPAMI figure claiming “final model,” verify:

```bash
# inspect keys / args stored in checkpoint
CUDA_VISIBLE_DEVICES=0 conda run -n deeplearning python - <<'PY'
import torch
p='codebase/Geo_physics_equation_derived_model/from_first_steps/prev_legacy_results/outputs_bijie/checkpoint/best.pt'
ckpt=torch.load(p,map_location='cpu')
print(type(ckpt), ckpt.keys() if isinstance(ckpt,dict) else 'tensor-or-module')
PY
```

If final weights are only on the HPC, copy them locally; **do not pretend** legacy step checkpoints are the absolute-final novel run.

---

## 7. Figure plan (scientific / architectural — beyond Neist)

| Fig | Content | Source |
|-----|---------|--------|
| T1 | Bounded `positive_scale` vs unbounded exp | `tpami_assets/figures/fig_bounded_positive_scale.png` ✅ |
| T2 | FS landscape over \((\alpha,h)\) | `fig_fs_landscape.png` ✅ |
| T3 | MPEF routing curves | `fig_mpef_routing.png` ✅ |
| T4 | Local sensitivity histogram of FS gate | `fig_fs_local_lipschitz_hist.png` ✅ |
| A1 | Full architecture schematic (pentastream → MPEF) | draw in TikZ / Illustrator |
| A2 | LCE diagram: same FS skeleton at inlet/fusion/decoder/outlet | TikZ |
| E1 | Path weight heatmaps | GPU0 inference |
| E2 | Robustness decay curves | GPU0 inference |
| E3 | Existing comparative bars / PR / qualitative | already in `gplnet_paper/figures` |

---

## 8. Writing rules distilled from TPAMI peers

From Gravityformer / Virtual Normal / Topological Loss / Sparse Fusion / Lipschitz certificates:

1. **Open with the learning problem**, not the application death statistics.  
2. **Name the law** (gravity / virtual normal / Betti / FS).  
3. **Show the closed-form or operator** before the network diagram.  
4. **Prove stability or optimality on a reduced problem**, then say the full net inherits it locally.  
5. **Ablate the physics module off** — if performance barely changes, TPAMI will reject.  
6. **Interpretability figure** of the physical quantity (gravity attention / virtual normals / FE maps).  
7. Continuous academic prose; avoid AI-style bullet contribution dumps.  
8. Related work must **position against physics-in-architecture**, not only landslide CNNs.

---

## 9. Citation ledger (BibTeX keys to use / add)

### 9.1 Core theory / TPAMI-style peers (already in `papers/`)

```bibtex
@article{wang2025gravityformer,
  title={A Gravity-informed Spatiotemporal Transformer for Human Activity Intensity Prediction},
  author={Wang, Yi and others},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  year={2025},
  note={arxiv PDF: papers/Gravityformer_arxiv.pdf}
}
@article{yin2021virtualnormal,
  title={Virtual Normal: Enforcing Geometric Constraints for Accurate and Robust Depth Prediction},
  author={Yin, Wei and Liu, Yifan and Shen, Chunhua},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  year={2021},
  note={papers/Virtual_Normal_arxiv.pdf}
}
@article{clough2022topological,
  title={A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology},
  author={Clough, James R. and others},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  year={2022},
  note={papers/Topological_Loss_Segmentation_arxiv.pdf}
}
@article{panda2025sparsefusion,
  title={l0-Regularized Sparse Coding-based Interpretable Network for Multi-Modal Image Fusion},
  author={Panda, Gargi and others},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  year={2025},
  note={papers/Sparse_Coding_Image_Fusion_arxiv.pdf}
}
@article{physicsdisentangle2023,
  title={Physics-Informed Guided Disentanglement in Generative Networks},
  author={/* fill from PDF metadata */},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  year={2023},
  note={papers/Physics_Informed_Disentanglement_arxiv.pdf}
}
@article{lipschitzseg2025,
  title={Fast and Flexible Robustness Certificates for Semantic Segmentation},
  author={/* fill authors from PDF */},
  year={2025},
  note={papers/Lipschitz_Segmentation_Certificates_arxiv.pdf}
}
@article{altinses2026layerlipschitz,
  title={Layer-Specific Lipschitz Modulation for Fault-Tolerant Multimodal Representation Learning},
  author={Altinses, Diyar and Schwung, Andreas},
  year={2026},
  note={papers/Layer_Lipschitz_Multimodal_arxiv.pdf}
}
@article{kkthardnet2025,
  title={Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints},
  year={2025},
  note={papers/KKT_Hardnet_Physics_Constraints_arxiv.pdf}
}
@article{pdeconstrainedseg2026,
  title={PDE-Constrained Optimization for Neural Image Segmentation with Physics Priors},
  year={2026},
  note={papers/PDE_Constrained_Segmentation_arxiv.pdf}
}
@article{coastalrobust2025,
  title={Multi-Modal Robust Enhancement for Coastal Water Segmentation},
  year={2025},
  note={papers/Coastal_Robust_UNet_arxiv.pdf}
}
```

### 9.2 Classical geotechnical / ML foundations

```bibtex
@article{skempton1957stability,
  title={Stability of natural slopes in London Clay},
  author={Skempton, A. W. and DeLory, F. A.},
  year={1957}
}
@article{newmark1965effects,
  title={Effects of earthquakes on dams and embankments},
  author={Newmark, Nathan M.},
  journal={G{\'e}otechnique},
  year={1965}
}
@article{karniadakis2021physics,
  title={Physics-informed machine learning},
  author={Karniadakis, George Em and others},
  journal={Nature Reviews Physics},
  year={2021}
}
@article{raissi2019pinn,
  title={Physics-informed neural networks},
  author={Raissi, M. and Perdikaris, P. and Karniadakis, G. E.},
  journal={JCP},
  year={2019}
}
```

### 9.3 Segmentation / multimodal / EO (application + baselines)

```bibtex
@inproceedings{ronneberger2015unet, ...}
@inproceedings{chen2018deeplab, ...}
@article{xie2021segformer, note={papers/2105.15203.pdf}}
@article{chen2021transunet, note={papers/2102.04306.pdf}}
@article{ji2020bijie, ...}
@article{ghorbanzadeh2022landslide4sense, ...}
@article{islam2024digate, note={papers/dual_stream.pdf}}
@article{mercier2022bifusion, note={papers/BiFusion.pdf}}
@article{szink2024prithvi, note={papers/2309.16653.pdf}}
@article{salehi2017tversky, ...}
@article{houlsby2019lora, ...}
@article{tan2019efficientnet, ...}
```

*(Existing keys already live in `codebase/project_report/gplnet_paper/bib/refs.bib` — merge, do not fork forever.)*

### 9.4 Convex / softmax energy (foundational math citations to add)

```bibtex
@article{boyd2004convex,
  title={Convex Optimization},
  author={Boyd, Stephen and Vandenberghe, Lieven},
  year={2004},
  publisher={Cambridge University Press}
}
@article{martin2019softmax,
  title={On the properties of Softmax},
  note={standard reference / textbook citation for softmax as entropic projection}
}
```

---

## 10. Immediate next actions (ordered)

1. **Freeze the claim sentence** (one sentence, no “landslide” required).  
2. **Verify checkpoint ↔ final metrics correspondence** (Section 6.4).  
3. **Draft LaTeX Theory section** with Definitions 1–5, Lemmas 1–3, Theorems 1–2 into `gplnet_paper/` (new `content/theory.tex`), without destroying current floats.  
4. **Run E2–E5 on GPU0** with `tpami_eval_utils.py` once the correct `best.pt` is confirmed.  
5. **Ablation E1**: if full retrain impossible, report **module-off inference** by surgically zeroing gates / replacing MPEF in a eval fork — clearly label as analysis. Prefer short compact finetunes if fair comparison needs training.  
6. **Rewrite Intro + Related Work** to match Gravityformer’s physics-informed positioning.  
7. Decide submission path: **TPAMI direct** vs **CVPR/ICCV first → TPAMI extension**.

---

## 11. Direct answers to your questions

| Question | Answer |
|----------|--------|
| Can we organize this for TPAMI while keeping landslide segmentation? | **Yes**, if landslides are the stress test and LCE/MPEF/FS-cells are the contribution. |
| Do we need huge new GPU training? | **No** for drafting theory + robustness/interpretability figures. **Maybe light** compact ablations on GPU0. |
| Is GPU0 usable now? | **Yes** — RTX 4000 Ada ~19 GB free at planning time; GPU1 was busy. |
| Can I destroy `deeplearning` env? | **No** — only add missing deps (e.g. `pypdf`); never upgrade torch. |
| Will TPAMI accept “application novelty only”? | **No.** Theory + mechanism ablation are mandatory. |
| Did we download closest papers? | **Yes** — Gravityformer, Virtual Normal, Topological Loss, Sparse Fusion, Physics Disentanglement, Lipschitz seg certificates, Layer-Lipschitz multimodal, KKT-Hardnet, PDE-constrained seg, Coastal Robust U-Net — all under `papers/`. |

---

## 12. Status of generated assets

```
from_first_steps/
  TPAMI_REORG_PLAN_AND_CITATIONS.md   ← this file
  tpami_eval_utils.py                 ← evaluation helpers (already present)
  tpami_assets/figures/
    fig_bounded_positive_scale.png
    fig_fs_landscape.png
    fig_mpef_routing.png
    fig_fs_local_lipschitz_hist.png
```

**Next human decision needed:** confirm which `.pt` is the official final PS-GPLNet checkpoint for Bijie/L4S, then we can run the low-VRAM TPAMI analysis suite on GPU0 without touching the busy 3060.
