# GeoPhysicsLandslideNet (PS-GPLNet): Architecture Specification

**Full name:** Geo-Physics Equation-Derived Landslide Segmentation Network — **Penta-Stream** variant (**PS-GPLNet**)

**Primary implementation class:** `GeoPhysicsLandslideNet` (`model.py`)

**Authors & context:** Samreedh Bhuyan (research implementation); mentor **Dr. Anil Earnest**; **KIIT**; **4PI** landslide segmentation programme

**Task:** Binary landslide segmentation from dual raster streams (RGB + topography / auxiliary geophysics) with optional planetary foundation context

**Reference code root:** `codebase/Geo_physics_equation_derived_model/from_first_steps/`

**Default training resolution:** \(256 \times 256\) (Bijie and Landslide4Sense pipelines)

**Default unified width:** \(C = 64\) (`fusion_channels=64`); compact preset uses \(C = 32\) without structural changes

**Companion documents:** parent overview `../model_architecture.md`; incremental build log `step_by_step_implementation.md`

---

## 1. Design philosophy: closed-loop geotechnical inductive bias

PS-GPLNet is built around one principle: **the same mechanistic stability narrative must appear at every stage of the network**, not only as a post-hoc loss or auxiliary channel.

1. **Inlet (encoders):** Observed slope, elevation, and vegetation proxies are mapped to geotechnical variables \((\alpha, h, m)\). A **Taylor-stabilized infinite-slope factor of safety (FS)** gates pixel features (`PixelMechanisticCell`) and propagates through **latent FS ratios** (`LatentMechanisticCell`) in deeper encoder and decoder stages.

2. **Mid-level fusion:** Hybrid CNN–physics streams supply a **physics anchor plane** to **MAO-GeoEGCA**, which queries Prithvi foundation context under an **equilibrium gate** tied to local geomechanics. **TTEB** builds skip tensors from a tri-stream, tri-scale lattice with a **latent stability map** \(\delta\) analogous to FS imbalance in feature space.

3. **Outlet (decoder):** Two parallel **PhysicsDecoder** paths decode with path-specific \((\alpha, h, m)\) and CNN bottleneck residuals. **PGDI** injects TTEB skips through physics-gated latent cells. Full-resolution logits pass through a final **pixel mechanistic gate**. **Mechanistic Path Equilibrium Fusion (MPEF)** merges path logits by **relative failure energy**, closing the loop at the output.

CNN texture (EfficientNet) is **complemented**, not replaced: the **Complementary Modality Bridge (CMB)** increases trust in physics features only when CNN and physics **resonate** (co-activate).

This is intentionally **not** a generic dual-stream U-Net with scalar fusion gates. The deployed model has **no** `GateFuse` encoder/decoder chain in the forward path.

---

## 2. Novelty and positioning (DiGATe is baseline only)

| Aspect | DiGATe / dual-stream gated U-Net (`model_backup.py`) | **PS-GPLNet (this architecture)** |
|--------|------------------------------------------------------|-------------------------------------|
| Encoders | 2× EfficientNet (RGB, DEM) ± Prithvi in ablations | **5 parallel encoders:** EffNet RGB, Physics RGB, EffNet DEM, Physics DEM, Prithvi+LoRA |
| Physics | None (purely data-driven CNN) | **FS-gated mechanistic pyramids** + proxy mappers |
| Per-modality fusion | N/A | **CMB** (resonance-weighted CNN–physics hybrid) |
| Multi-stream fusion | Chained scalar **GateFuse** | **MAO-GeoEGCA** + **TTEB** only |
| Decoder | Paper `AdaptiveDecoder` + GateFuse on features | **Dual PhysicsDecoder** + **PGDI** + pixel FS head |
| Dual-path merge | Learned gates on feature maps | **MPEF** on logits from **instability-weighted** routing |
| Foundation model | Optional add-on in stepwise ablations | **Required** Prithvi-EO-2.0-100M-TL with LoRA |

**DiGATe** remains in `model_backup.py` **solely for comparison experiments**; it is **not** described or replicated here as the target architecture.

Differentiation from generic **physics-informed remote sensing**:

- Physics is embedded in **differentiable cells** (pixel + latent), not only in PDE residual losses.
- Foundation context is **manifold-aligned** to the hybrid physics plane (MAO), not naive concatenation.
- Dual decode paths are merged by **geotechnical failure energy**, not by a single shared decoder or feature-level scalar gate.

---

## 3. Notation

| Symbol | Meaning |
|--------|---------|
| \(B\) | Batch size |
| \(C\) | Unified channel width (`fusion_channels`; default 64) |
| \(H, W\) | Spatial height and width at a given level |
| \(i \in \{0,\ldots,4\}\) | EfficientNet pyramid index |
| \(L \in \{0,\ldots,4\}\) | Physics / Prithvi legacy level index |
| \(\alpha\) | Slope angle (radians), pixel-wise |
| \(h\) | Column height / elevation proxy (\(> 0\)) |
| \(m\) | Moisture–vegetation saturation proxy \(\in (0,1)\) |
| \(s, d, v\) | Normalized slope, DEM, NDVI/vegetation **proxies** in \([0,1]\) |
| \(\text{FS}, \text{FS}_{\text{nn}}\) | Classical / neural factor of safety |
| \(\varepsilon\) | Stabilizer (\(10^{-6}\) in code) |
| \(\odot\) | Hadamard (element-wise) product |
| \([\cdot \,\|\, \cdot]\) | Channel concatenation |
| \(\sigma(\cdot)\) | Sigmoid |
| \(\text{Softplus}(\cdot)\) | Softplus activation |
| \(\|\cdot\|_2\) | L2 norm along the feature dimension (per token in attention) |

**Stream tensors (no dataset format change):**

| Name | Shape | Content |
|------|-------|---------|
| `stream_a` / \(x_1\) | \(B \times 3 \times 256 \times 256\) | RGB |
| `stream_b` / \(x_2\) | \(B \times 3 \times 256 \times 256\) | L4S: NDVI, slope, DEM; Bijie: DEM replicated ×3 |

Proxies \((s,d,v)\) and Prithvi 6-band stacks are built **inside** `GeoPhysicsLandslideNet.forward()` (`physics_proxies_from_streams`, `observed_stack_from_streams`).

---

## 4. Inputs, proxies, and datasets

### 4.1 Proxy extraction (implementation: `model.py`)

**Landslide4Sense** (`stream_b` = NDVI, slope, DEM):

\[
s = x_2[:,1],\quad d = x_2[:,2],\quad v = x_2[:,0]
\]

**Bijie** (`stream_b` all channels equal DEM):

\[
d = x_2[:,0],\quad s = \text{Norm}(\|\nabla d\|),\quad v = \text{Norm}\!\left(\frac{G-R}{G+R+\varepsilon}\right)
\]

Per-channel batch min–max normalization to \([0,1]\) is applied to the Prithvi stack; slope/vegetation helpers normalize per sample where noted in code.

**DEM channel for EffNet-DEM:** channel 2 of `stream_b` (L4S) or channel 0 (Bijie replicate layout).

### 4.2 PhysicsProxyMapper (`physics/proxy_mapper.py`)

Separate instances `proxy_rgb` and `proxy_dem` (same architecture, independent weights):

\[
\alpha = \alpha_{\max}\,\sigma(w_\alpha s + b_\alpha), \quad \alpha_{\max} = \pi/2
\]
\[
h = \text{Softplus}(w_h d + b_h) + 10^{-4}
\]
\[
m = \sigma(w_m v + b_m)
\]

### 4.3 Prithvi observed stack (`observed_stack_from_streams`)

| Layout | 6 channels |
|--------|------------|
| L4S | \([R, G, B, \text{NDVI}, \text{slope}, \text{DEM}]\) |
| Bijie | \([R, G, B, \text{DEM}, \text{Sobel}(d), \text{vegIndex}(RGB)]\) |

Normalization in `PrithviFoundationEncoder`: \((x - 0.5)/0.25\) (`observed_rasters` mode).

### 4.4 Datasets

| Dataset | Loader | Split / notes |
|---------|--------|----------------|
| **Bijie** | `train_bijie.py`, `bijie_dataset.py` | PNG composites; typical 70/20/10 train/val/test |
| **Landslide4Sense** | `training.py`, `dataset.py` | H5 dual-stream; Train/Valid/Test or 90/10 val split |

**Reported best checkpoints (validation):**

| Run folder | Dataset | Best epoch | F1 | IoU |
|------------|---------|------------|-----|-----|
| `outputs_absolute_final_fully_novel_complete/` (successor to `outputs_final_bijie/`) | Bijie | **92** | **≈ 0.933** | **≈ 0.907** |
| `outputs_step3_l4s/` (successor to `outputs_final_l4s/`) | Landslide4Sense | **70** | **≈ 0.736** | **≈ 0.660** |

Metrics logged in `results/epoch_metrics.csv` and `results/final_metrics.csv` under each output directory.

---

## 5. Full architecture overview

### 5.1 Five encoders

| # | Module | Input | Output |
|---|--------|-------|--------|
| 1 | \(E_{\text{rgb}}\) — `encoder_rgb` | `stream_a` (3 ch) | EffNet pyramid \(\{E_{\text{rgb}}^i\}_{i=0}^4\) |
| 2 | \(P_{\text{rgb}}\) — `enc_phys_rgb` | `stream_a`, \((\alpha_r,h_r,m_r)\) | Physics pyramid \(\{P_{\text{rgb}}^L\}_{L=0}^4 \rightarrow C\) |
| 3 | \(E_{\text{dem}}\) — `encoder_dem` | DEM (1 ch) | \(\{E_{\text{dem}}^i\}_{i=0}^4\) |
| 4 | \(P_{\text{dem}}\) — `enc_phys_dem` | DEM, \((\alpha_d,h_d,m_d)\) | \(\{P_{\text{dem}}^L\}_{L=0}^4 \rightarrow C\) |
| 5 | \(T_{\text{fm}}\) — `encoder_fm` | 6-band stack | \(\{T_{\text{fm}}^L\}_{L=0}^4 \rightarrow C\) |

### 5.2 CMB → tri-stream tensors for fusion

At each EffNet index \(i\):

\[
\tilde{P}_{\text{rgb}}^i = \text{align}\big(P_{\text{rgb}}^{\text{LEGACY}[i]}, E_{\text{rgb}}^i\big),\quad
H_{\text{rgb}}^i = \text{CMB}(E_{\text{rgb}}^i, \tilde{P}_{\text{rgb}}^i)
\]

Similarly \(H_{\text{dem}}^i\) from \(E_{\text{dem}}\) and aligned \(P_{\text{dem}}\).

**Implementation note:** MAO and TTEB consume **`H_rgb`, `H_dem`** (CMB outputs at width \(C\)) and **`T_fm`** from Prithvi maps aligned to \(E_{\text{rgb}}^i\) via `fm_align` (`_fusion_pyramids`). Raw EffNet pyramids are **not** fed directly into MAO/TTEB after CMB is applied.

### 5.3 MAO-GeoEGCA (bottleneck)

Applied at **`MAO_LEVELS = (2, 3)`** → spatial grids **32×32** and **16×16** for B4@256.

Outputs: \(F^2\) (neck, with `post_fuse3` 1×1 conv) and \(F^3\) (bottleneck).

### 5.4 TTEB (skips)

Applied at **`TTEB_LEVELS = (0, 1, 2, 3)`** → skip list \(\{S^0, S^1, S^2, S^3\}\) at 128², 64², 32², 16².

### 5.5 DualPhysicsDecoder + MPEF

- Path A: \(F^3 + \text{Proj}(a_5)\), proxies \((\alpha_r,h_r,m_r)\)
- Path B: \(F^3 + \text{Proj}(b_5)\), proxies \((\alpha_d,h_d,m_d)\)
- Shared: \(F^2\), TTEB skips, same PGDI chain
- Merge: **MPEF** on **main**, **aux2**, **aux3** logits (three fusion modules)

---

## 6. Pyramid geometry and `LEGACY_LEVEL_FOR_EFFNET`

Three encoder families use **different native grids** at 256×256 input.

### 6.1 EfficientNet-B4 (\(E_{\text{rgb}}, E_{\text{dem}}\))

| Index \(i\) | Channels | Spatial |
|-------------|----------|---------|
| 0 | 24 | 128×128 |
| 1 | 32 | 64×64 |
| 2 | 56 | 32×32 |
| 3 | 160 | 16×16 |
| 4 | 448 | 8×8 |

*(B0 backbone in `--compact` mode uses timm’s B0 channel schedule; indices and alignment map unchanged.)*

### 6.2 PhysicsEncoder (\(P_{\text{rgb}}, P_{\text{dem}}\))

Internal channels: \(c_0=\max(8,C/4)\), \(c_1=\max(8,C/2)\), \(c_2=c_3=c_4=C\).

| Level \(L\) | Spatial (after downsample chain) |
|-------------|----------------------------------|
| 0 | 256×256 |
| 1 | 128×128 |
| 2 | 64×64 |
| 3 | 32×32 |
| 4 | 16×16 |

After `StreamProjector`: all levels are \(B \times C \times H_L \times W_L\).

### 6.3 Prithvi (`prithvi_encoder.py`)

- Blocks tapped: \(\{2, 5, 8, 11\}\)
- Token maps → 1×1 conv to \(C\) → resize to `LEGACY_TARGET_SIZES`: L1–L4 at 128, 64, 32, 16; L0 by upsampling L1 to 256

### 6.4 Cross-stream alignment

```python
LEGACY_LEVEL_FOR_EFFNET = (1, 2, 3, 4, 4)
```

For EffNet index \(i\), physics level \(\text{LEGACY\_LEVEL\_FOR\_EFFNET}[i]\) is bilinearly resized to match \(E_{\text{rgb}}^i\) (function `_align_legacy_level` / `match_spatial`).

| EffNet index \(i\) | Spatial (B4) | Physics level used |
|--------------------|--------------|---------------------|
| 0 | 128×128 | 1 |
| 1 | 64×64 | 2 |
| 2 | 32×32 | 3 |
| 3 | 16×16 | 4 |
| 4 | 8×8 | 4 |

Prithvi fusion maps use the **same index map** when aligning to RGB EffNet levels in `_fusion_pyramids`.

**Why MAO at indices 2–3 only:** index 4 (8×8) would leave the four upsampling stages of `PhysicsDecoder` ending at 128×128; indices 2–3 (32², 16²) allow full 256×256 main logits after four ×2 upsamples.

---

## 7. Geotechnical and neural mechanistic formulations

### 7.1 Classical infinite slope (parallel seepage)

\[
\text{FS} = \frac{c' + (\gamma_{\text{sat}} H - u)\cos^2\alpha \tan\phi'}{\gamma_{\text{sat}} H \sin\alpha \cos\alpha}, \quad u = m \gamma_w H \cos^2\alpha
\]

Failure when \(\text{FS} \le 1\).

### 7.2 Bounded positive material parameters

`physics/params.py`:

\[
\text{positive\_scale}(w) = \exp\big(\mathrm{clamp}(w, -8, 8)\big)
\]

Used for \(c, \phi, \gamma, w_m\) in pixel cells and MPEF failure energy (prevents \(\exp\) blow-up).

### 7.3 Taylor-stabilized pixel FS (`PixelMechanisticCell`)

Pore pressure enters the **driving** denominator branch:

\[
\text{FS}_{\text{nn}} = \frac{c + \phi\, h\, \cos^2\alpha}{\gamma\, h\, \sin\alpha\cos\alpha + w_m\, m + \varepsilon}
\]

with \(c,\phi,\gamma,w_m\) from `positive_scale` on learnable per-channel parameters.

\[
\text{failure\_energy} = \psi - \text{FS}_{\text{nn}}, \quad \text{gate} = \sigma(\text{failure\_energy})
\]

\[
\text{PixelMechanisticCell}(x,\alpha,h,m) = \text{Conv}_{1\times1}(x) \odot \text{gate}
\]

If `mechanistic_gating=False`, returns conv features without gating.

### 7.4 Latent mechanistic cell (`LatentMechanisticCell`)

No trigonometry in latent space:

\[
R = \text{Softplus}(W_R * H), \quad D = \text{Softplus}(W_D * H)
\]
\[
\text{LatentMechanisticCell}(H) = \text{Conv}_{3\times3}\!\left(\text{LeakyReLU}\!\left(\psi - \frac{R}{D+\varepsilon}\right)\right)
\]

### 7.5 CMB — Complementary Modality Bridge

Let \(\tilde{E} = \text{GN}(\text{ReLU}(\text{Conv}_{1\times1}(E)))\) (identity if already \(C\) channels), \(\tilde{P} = \text{GN}(P)\):

\[
g = \sigma\!\left(\text{Conv}_{1\times1}(\tilde{E} \odot \tilde{P})\right)
\]
\[
B_{\text{mix}} = \text{Conv}_{1\times1}([\tilde{E} \,\|\, \tilde{P}])
\]
\[
H = g \odot \tilde{P} + (1-g) \odot \tilde{E} + B_{\text{mix}}
\]

### 7.6 MAO-GeoEGCA (`fusion/mao_geo_egca.py`)

Given \(\hat{T} = \hat{T}_{\text{fm}}^i\), \(\hat{H}_{\text{rgb}}^i\), \(\hat{H}_{\text{dem}}^i\):

\[
X_p = \text{Conv}_{1\times1}([\hat{H}_{\text{rgb}} \,\|\, \hat{H}_{\text{dem}}])
\]
\[
G = \sigma\!\left(\text{DWSepConv}_{3\times3}(\hat{T} \odot X_p)\right)
\]

Projections \(Q = W_q X_p\), \(K = W_k \hat{T}\), \(V = W_v \hat{T}\). Manifold alignment:

\[
\hat{Q} = \frac{Q}{\|Q\|_2 + \varepsilon}, \quad K' = K \odot \hat{Q}
\]

Multi-head attention (4 heads, \(d_h = C/4\)):

\[
\text{Attn} = \mathrm{softmax}\!\left(\frac{Q K'^{\top}}{\sqrt{d_h}}\right) V
\]

\[
F = \text{Conv}_{3\times3}(\text{reshape}(\text{Attn}) \odot G) + X_p
\]

### 7.7 TTEB — Tri-Temporal Tri-Stream Bridge (`fusion/tteb.py`)

For fusion level index `level` \(L\) on lists \(\{\hat{H}_{\text{rgb}}^i\}\), \(\{\hat{H}_{\text{dem}}^i\}\), \(\{\hat{T}_{\text{fm}}^i\}\):

**Lattice:** for each stream, gather levels \(L-1, L, L+1\) (clamped), resize to present spatial size.

\[
A = \text{Conv}_{1\times1}([\text{present}_{\text{rgb}} \,\|\, \text{present}_{\text{dem}} \,\|\, \text{present}_{\text{fm}}])
\]

Context: 8 off-diagonal nodes (prev/next across streams + cross terms) → `context_mix` → \(H_{\text{ctx}}\).

**Stability map:**

\[
\delta = \left|\psi - \frac{\text{Softplus}(W_R A)}{\text{Softplus}(W_D A)+\varepsilon}\right|
\]

Spatial attention: \(Q\) from \(A\), \(K,V\) from \(H_{\text{ctx}}\) with stream-phase bias on keys; chunked attention for memory (`attn_chunk_size`, optional downsample when tokens \(> \texttt{attn\_low\_res\_max}\)).

**Temporal router:** \(\omega_\tau = \mathrm{softmax}_\tau(W_\tau \delta)\), \(\tau \in \{\text{prev}, \text{pres}, \text{next}\}\); output scaled by present-phase weight.

\[
S^L = A + \text{Conv}_{3\times3}\!\big(\text{AttnOut} \odot \sigma(\delta)\big)
\]

(with additional scaling by \(\omega_{\text{pres}}\) in code)

### 7.8 PGDI — Physics-Gated Decoder Injection

\[
\beta = \sigma\!\left(\text{Conv}_{1\times1}([D \,\|\, S'])\right), \quad S' = \text{align}(S, D)
\]
\[
D' = D + \beta \odot \text{LatentMechanisticCell}(S')
\]

### 7.9 MPEF — Mechanistic Path Equilibrium Fusion

For each path \(p \in \{A,B\}\) with logits \(O_p\) and proxies \((\alpha_p, h_p, m_p)\):

\[
\text{FE}_p = \mathrm{ReLU}\!\left(1 - \text{FS}_{\text{nn}}(\alpha_p, h_p, m_p)\right)
\]

\[
[w_A, w_B] = \mathrm{softmax}([\text{FE}_A, \text{FE}_B]) \quad \text{(along path dimension)}
\]

\[
O = w_A O_A + w_B O_B + \text{Conv}_{1\times1}([O_A \,\|\, O_B])
\]

Routing regularizer (per head):

\[
\mathcal{R}_{\text{MPEF}} = -\mathbb{E}\big[w_A \log(w_A+\varepsilon) + w_B \log(w_B+\varepsilon)\big]
\]

### 7.10 Loss (`losses.py`)

Tversky index on sigmoid logits:

\[
\mathcal{T}(p,t) = \frac{TP + s}{TP + \alpha FP + \beta FN + s}, \quad \mathcal{L}_{\text{Tversky}} = 1 - \mathcal{T}
\]

Default \(\alpha=0.3\), \(\beta=0.7\) (emphasis on recall / landslide pixels).

\[
\mathcal{L} = w_1 \mathcal{L}_{\text{Tversky}}(\text{main}) + w_2 \mathcal{L}_{\text{Tversky}}(\text{aux2}) + w_3 \mathcal{L}_{\text{Tversky}}(\text{aux3}) + \lambda_{\text{reg}} \sum_{h \in \{\text{main},\text{aux2},\text{aux3}\}} \mathcal{R}_{\text{MPEF}}^{(h)}
\]

Defaults: \(w_1=1.0\), \(w_2=0.6\), \(w_3=0.4\), \(\lambda_{\text{reg}}=10^{-3}\).

Main, aux2, aux3 are bilinearly upsampled to input resolution when spatial sizes differ before loss evaluation.

---

## 8. Algorithms (pseudocode)

### Algorithm 1 — PS-GPLNet forward (`GeoPhysicsLandslideNet.forward`)

```
Input: stream_a (RGB), stream_b (topography / aux)
──────────────────────────────────────────────────────────────
1. (s, d, v) ← physics_proxies_from_streams(stream_a, stream_b)
2. (α_r, h_r, m_r) ← proxy_rgb(s, d, v)
   (α_d, h_d, m_d) ← proxy_dem(s, d, v)

3. {E_rgb^i} ← encoder_rgb(stream_a)
   {E_dem^i} ← encoder_dem(DEM_channel(stream_b))
   {P_rgb^L} ← enc_phys_rgb(stream_a, α_r, h_r, m_r)
   {P_dem^L} ← enc_phys_dem(DEM, α_d, h_d, m_d)
   {T_fm^L}  ← encoder_fm(observed_stack_from_streams(stream_a, stream_b))

4. For i = 0..4:
       H_rgb^i ← CMB(E_rgb^i, align(P_rgb, LEGACY[i], E_rgb^i))
       H_dem^i ← CMB(E_dem^i, align(P_dem, LEGACY[i], E_dem^i))
       T̂_fm^i  ← fm_align_i(align(T_fm, LEGACY[i], E_rgb^i))

5. For L in {0,1,2,3}:
       S^L ← TTEB({H_rgb}, {H_dem}, {T̂_fm}, level=L)

6. F² ← post_fuse3( MAO(T̂_fm², H_rgb², H_dem²) )
   F³ ← MAO(T̂_fm³, H_rgb³, H_dem³)

7. F³_A ← F³ + Proj(a_5);  F³_B ← F³ + Proj(b_5)
   (main_A, aux2_A, aux3_A) ← PhysicsDecoder(F³_A, F², {S^L}, α_r, h_r, m_r)
   (main_B, aux2_B, aux3_B) ← PhysicsDecoder(F³_B, F², {S^L}, α_d, h_d, m_d)

8. main ← MPEF(main_A, main_B, ...);  likewise aux2, aux3
9. Upsample main, aux2, aux3 to 256×256 if needed
10. Return (main, aux2, aux3, (reg_main, reg_aux2, reg_aux3))
```

### Algorithm 2 — MAO_GeoEGCA(\(\hat{T}, \hat{H}_{\text{rgb}}, \hat{H}_{\text{dem}}\))

```
1. X_p ← Conv1×1([H_rgb ; H_dem])
2. G   ← Sigmoid(DWSepConv3×3(T ⊙ X_p))
3. Q ← W_q X_p;  K ← W_k T;  V ← W_v T
4. Q̂ ← normalize(Q, dim=feature);  K' ← K ⊙ Q̂
5. Context ← MultiHeadAttention(Q, K', V)   // 4 heads
6. Out ← Conv3×3(reshape(Context) ⊙ G) + X_p
7. Return Out
```

### Algorithm 3 — TTEB(\(L\))

```
1. For each stream s in {rgb, dem, fm}:
       gather F[s, L-1], F[s, L], F[s, L+1] with boundary clamp; align to size(F[s,L])
2. A ← Conv1×1([F_rgb,L ; F_dem,L ; F_fm,L])
3. H_ctx ← context_mix(concat 8 off-diagonal lattice tensors)
4. δ ← |ψ - Softplus(W_R·A) / (Softplus(W_D·A) + ε)|
5. AttnOut ← SpatialAttention(A, H_ctx) with chunked / downsampled attention if needed
6. τ ← softmax(temporal_router(δ)); scale AttnOut by present-phase weight
7. S^L ← A + Conv3×3(AttnOut ⊙ σ(δ))
8. Return S^L
```

### Algorithm 4 — PGDI(\(D, S\))

```
1. S' ← match_spatial(S, D)
2. β ← Sigmoid(Conv([D ; S']))
3. D' ← D + β ⊙ LatentMechanisticCell(S')
4. Return D'
```

### Algorithm 5 — MPEF(\(O_A, O_B, \alpha_A, h_A, m_A, \alpha_B, h_B, m_B\))

```
1. Align (α, h, m) for each path to spatial size of logits
2. FE_A ← ReLU(1 - FS_nn(α_A, h_A, m_A))
   FE_B ← ReLU(1 - FS_nn(α_B, h_B, m_B))
3. [w_A, w_B] ← softmax([FE_A, FE_B], dim=path)
4. O ← w_A ⊙ O_A + w_B ⊙ O_B + Conv1×1([O_A ; O_B])
5. R ← mean( -(w_A log(w_A+ε) + w_B log(w_B+ε)) )
6. Return O, R
```

### Algorithm 6 — CMB(\(E, P\))

```
1. P ← match_spatial(P, E)
2. Ẽ ← GN(ReLU(Conv1×1(E)))  // project CNN to C if needed
3. P̃ ← GN(P)
4. g ← Sigmoid(Conv1×1(Ẽ ⊙ P̃))
5. B ← Conv1×1([Ẽ ; P̃])
6. H ← g ⊙ P̃ + (1-g) ⊙ Ẽ + B
7. Return H
```

### Algorithm 7 — PhysicsEncoder level \(L\)

```
1. f0 ← Stem(PixelMechanisticCell(x, α, h, m))
2. f1 ← LatentMechanisticCell(Down(f0))
3. f2 ← LatentMechanisticCell(Down(f1))
4. f3 ← LatentMechanisticCell(Down(f2))
5. f4 ← LatentMechanisticCell(Down(f3))
6. Return [Proj0(f0), ..., Proj4(f4)]
```

---

## 9. Large ASCII architecture diagram

```
                         ┌─────────────────────────────────────────────────────────────┐
                         │                    PROXY & FOUNDATION INLET                  │
                         │  (s,d,v) ──► PhysicsProxyMapper_rgb ──► (α_r,h_r,m_r)       │
                         │          └──► PhysicsProxyMapper_dem ──► (α_d,h_d,m_d)       │
                         │  stream_a/b ──► observed_stack ──► Prithvi-EO-2.0 + LoRA      │
                         └─────────────────────────────────────────────────────────────┘

 stream_a (RGB) ───────► EfficientNet_rgb ──► E_rgb^0..4 ──┐
              └────────► PhysicsEnc_rgb ──► P_rgb^0..4 ─────┼──► CMB ×5 ──► H_rgb^0..4 ───┐
                                                            │                              │
 stream_b (DEM) ───────► EfficientNet_dem ──► E_dem^0..4 ──┤                              │
              └────────► PhysicsEnc_dem ──► P_dem^0..4 ─────┼──► CMB ×5 ──► H_dem^0..4 ───┼──┐
                                                            │                              │  │
 Prithvi 6-band ────────────────────────────────────────────┘                              │  │
         T_fm^0..4 ── fm_align + LEGACY align to E_rgb^i ──► T̂_fm^0..4 ──────────────────┘  │
                                                                                              │
         ┌──────────────────────────────── MAO-GeoEGCA @ i=2,3 (32², 16²) ────────────────┤
         │                         F² (neck) ────────────────┐                               │
         │                         F³ (bottleneck) ──────────┼───────────────┐               │
         │                                                     │               │               │
         └── TTEB @ L=0,1,2,3 ──► skips S⁰..S³ (128²→16²) ────┘               │               │
                                                                                 │               │
                    ┌────────────────────────────┬───────────────────────────────┘               │
                    ▼                            ▼                                               │
         F³_A = F³ + Proj(E_rgb^4=a_5)          F³_B = F³ + Proj(E_dem^4=b_5)                  │
         PhysicsDecoder path A                  PhysicsDecoder path B                            │
         (α_r,h_r,m_r)                          (α_d,h_d,m_d)                                    │
              │ LatentMech ×4 + PGDI×4 + PixelMech head                                          │
              ▼                                    ▼                                             │
         main_A, aux2_A, aux3_A              main_B, aux2_B, aux3_B                              │
              └──────────────────── MPEF (failure-energy routing) ──────────────────────────────┘
                                              │
                                              ▼
                                    main / aux2 / aux3  @ 256×256
                                    + MPEF entropy reg (×3)
```

**Data-flow summary:** five encoders → per-modality **CMB** → tri-stream **TTEB** skips + **MAO** neck/bottleneck → **dual PhysicsDecoders** → **MPEF** → Tversky multi-head loss.

---

## 10. Code mapping (`from_first_steps/`)

| Component | File |
|-----------|------|
| Full model, proxy helpers, EffNet builder | `model.py` |
| Prithvi + LoRA + pyramid | `prithvi_encoder.py` |
| PhysicsEncoder | `encoders/physics_encoder.py` |
| StreamProjector | `encoders/projector.py` |
| Pixel / latent cells, proxy mapper, params | `physics/pixel_cell.py`, `latent_cell.py`, `proxy_mapper.py`, `params.py` |
| CMB | `fusion/hybrid_stream_bridge.py` |
| MAO-GeoEGCA | `fusion/mao_geo_egca.py` |
| TTEB | `fusion/tteb.py` |
| Spatial align | `fusion/pyramid_utils.py` |
| PhysicsDecoder, PGDI | `decoder/physics_decoder.py`, `decoder/pgdi.py` |
| Dual decoder + MPEF | `decoder/dual_physics_decoder.py`, `decoder/mechanistic_path_fusion.py` |
| Loss | `losses.py` |
| Bijie training CLI | `train_bijie.py` |
| Landslide4Sense training CLI | `training.py` |
| Compact preset | `compact_config.py` |
| **DiGATe baseline (comparison only)** | `model_backup.py` |

**Backward-compatible aliases in `model.py`:** `DualStreamGateNet`, `DiGATe_Unet` → `GeoPhysicsLandslideNet` (same weights; name legacy only).

---

## 11. Training protocol and CLI

| Item | Default |
|------|---------|
| Optimizer | Adam, lr \(3\times10^{-4}\), weight decay \(10^{-4}\) |
| Epochs | 100 |
| Batch size | 32 |
| Segmentation loss | Tversky \(\alpha=0.3\), \(\beta=0.7\) on main, aux2, aux3 |
| Head weights | 1.0 / 0.6 / 0.4 |
| MPEF reg | \(\lambda_{\text{reg}}=10^{-3}\) on three routing terms |
| Metric threshold | 0.5 on main head |
| Checkpoints | every `--save_every` (default 5) + best validation F1 |
| EffNet | `--pretrained`, `--freeze_backbone` (default on) |
| Prithvi | `--prithvi_snapshot` **required** (HF cache or snapshot dir) |
| LoRA rank | 8 (default); capped at 4 in compact mode |

**Ablation:**

| Flag | Effect |
|------|--------|
| `--no_mechanistic_gating` | Pixel cells skip FS gate; decoder skips final pixel FS gate |

**Compact mode (`--compact`):** sets `tf_efficientnet_b0`, `fusion_channels=32`, `lora_rank ≤ 4`, smaller TTEB attention caps. **Architecture unchanged:** five encoders, CMB, MAO, TTEB, **dual** PhysicsDecoders, MPEF.

**Bijie example:**

```bash
python train_bijie.py \
  --dataset_root /path/to/Bijie-landslide-dataset \
  --output_dir ./outputs_absolute_final_fully_novel_complete \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --backbone tf_efficientnet_b4 --pretrained --freeze_backbone \
  --epochs 100 --batch_size 32 --lr 3e-4 \
  --tversky_alpha 0.3 --tversky_beta 0.7
```

**Landslide4Sense example:**

```bash
python training.py \
  --dataset_root /path/to/Landslide4Sense/dataset \
  --output_dir ./outputs_step3_l4s \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --bands RGB-NDVI-SLOPE-DEM \
  --backbone tf_efficientnet_b4 --pretrained --freeze_backbone \
  --epochs 100 --batch_size 32 --lr 3e-4
```

---

## 12. Parameter budget (approximate)

| Component | Parameters |
|-----------|------------|
| EfficientNet ×2 (RGB + DEM) | ~19M each, **frozen** by default |
| PhysicsEncoder ×2 | ~2M trainable (scales with \(C\)) |
| CMB ×10 (5 levels × 2 modalities) | ~0.5M |
| Prithvi ViT backbone | ~100M **frozen** |
| Prithvi LoRA (\(r=8\)) | ~0.5M trainable |
| MAO ×2 | ~0.3M |
| TTEB ×4 | ~1.5M |
| Dual PhysicsDecoder + PGDI | ~2M |
| MPEF ×3 | negligible |
| Proxy mappers ×2 | negligible |
| **Typical trainable total** | **~7–8M** (B4, \(C=64\)) |

Compact (\(C=32\), B0) reduces trainable physics/fusion/decoder mass roughly quadratically in width-sensitive blocks.

---

## 13. Shape trace (reference: \(B=2\), B4, \(C=64\), input 256×256)

| Tensor | Shape |
|--------|-------|
| `stream_a`, `stream_b` | 2×3×256×256 |
| \(E_{\text{rgb}}^0\) | 2×24×128×128 |
| \(E_{\text{rgb}}^4\) (\(a_5\)) | 2×448×8×8 |
| \(P_{\text{rgb}}^0\) (after proj) | 2×64×256×256 |
| \(H_{\text{rgb}}^2\) (post-CMB) | 2×64×32×32 |
| \(\hat{T}_{\text{fm}}^3\) | 2×64×16×16 |
| \(F^2\) (MAO neck) | 2×64×32×32 |
| \(F^3\) (MAO bottleneck) | 2×64×16×16 |
| \(S^3\) | 2×64×16×16 |
| \(S^0\) | 2×64×128×128 |
| `main` (before upsample) | 2×1×256×256 |
| `aux2` / `aux3` | 2×1×64×64 / 2×1×32×32 (upsampled for loss) |

---

## 14. Why this design is mathematically coherent

1. **Single stability grammar:** Taylor-stabilized FS ratios define gates at pixels, latent stress in encoders/decoders/PGDI, equilibrium gating in MAO, stability map \(\delta\) in TTEB, and routing weights in MPEF — one symbolic language from inlet to logits.

2. **Identifiability-friendly parameters:** `positive_scale` keeps material scalars bounded; FS appears as **ratios** (resisting/driving), improving numerical conditioning vs subtractive pore-pressure forms.

3. **Asymmetric attention with physical anchor:** MAO fixes **queries** on the hybrid physics plane and **keys/values** on foundation context, with **manifold alignment** \(K' = K \odot \mathrm{normalize}(Q)\) — a geometric constraint that standard cross-attention lacks.

4. **Dual paths with equilibrium merge:** Two decoders share context but specialize via \((\alpha,h,m)\) and \(a_5/b_5\) residuals; MPEF is a **softmax over failure energies**, a convex combination interpretable as path equilibrium rather than opaque feature gating.

5. **No shortcut to a single stream:** CMB preserves CNN when physics disagrees; MPEF preserves both paths until instability evidence selects weight — reducing collapse to a single modality.

---

## 15. Related work (brief)

- **Physics-informed neural networks (PINNs):** PDE residuals in loss; PS-GPLNet embeds mechanics **inside** forward feature maps.
- **Geotechnical limit equilibrium:** Infinite slope FS motivates pixel gates and MPEF routing.
- **Remote sensing segmentation:** EfficientNet/U-Net baselines provide texture; **Prithvi-EO** supplies seasonal/planetary priors via MAO-gated attention.
- **Multi-decoder fusion:** Unlike feature-level scalar gates (dual-stream gated papers), MPEF operates on **logits** with **explicit geotechnical energies**.
- **DiGATe / dual-stream gated U-Net:** Historical strong baseline in this repo (`model_backup.py`); **not** the PS-GPLNet architecture.

---

## 16. Reviewer-facing narrative loop

1. **Observe** RGB and topography (and derived NDVI/slope where available).
2. **Encode** with parallel CNN and FS-gated physics pyramids; **merge** by resonance (CMB).
3. **Fuse** planetary context only where local hybrid physics and Prithvi co-equilibrate (MAO); **propagate** multi-scale context to skips (TTEB).
4. **Decode** twice with shared mechanistic skips but path-specific proxies and bottlenecks.
5. **Decide** segmentation by **instability-weighted equilibrium** between paths (MPEF).

PS-GPLNet is a **standalone pentastream design**: CMB hybrid encoding, MAO/TTEB geo-equilibrium fusion, dual mechanistic decoding, and MPEF path equilibrium — unified by Taylor-stabilized factor-of-safety structure from pixels to output routing.

---

*Document version: aligned with `GeoPhysicsLandslideNet` in `model.py` (MPEF merge; no GateFuse in forward path).*
