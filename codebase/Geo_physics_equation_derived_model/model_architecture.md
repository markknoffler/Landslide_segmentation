# Geo-Physics Equation-Derived Landslide Segmentation Network

**Model name:** GeoPhysicsLandslideNet — Penta-Stream Variant (PS-GPLNet)  
**Reference implementation:** `codebase/Geo_physics_equation_derived_model/from_first_steps/`  
**Task:** Binary landslide segmentation from RGB + topography + geospatial foundation features  
**Primary class:** `GeoPhysicsLandslideNet` (`from_first_steps/model.py`)  
**Input resolution:** 256×256 (Landslide4Sense and Bijie training pipelines)  
**Unified feature width:** \(C = 64\)

---

## 1. Design philosophy

The network enforces a **closed-loop geotechnical inductive bias**: the same Taylor-stabilized infinite-slope factor-of-safety (FS) logic appears at the **encoder inlet** (pixel mechanistic cells), the **fusion core** (physics anchor in MAO-GeoEGCA), and the **decoder outlet** (latent mechanistic cells, PGDI, pixel mechanistic head).

Unlike the original three-stream GPLNet sketch (physics RGB + physics DEM + foundation model only), the deployed architecture runs **five encoders in parallel**:

| # | Stream | Role |
|---|--------|------|
| 1 | EfficientNet RGB (\(E_{\text{rgb}}\)) | ImageNet-scale texture and appearance |
| 2 | PhysicsEncoder RGB (\(P_{\text{rgb}}\)) | FS-gated mechanistic features from spectral input |
| 3 | EfficientNet DEM (\(E_{\text{dem}}\)) | Topographic texture from elevation |
| 4 | PhysicsEncoder DEM (\(P_{\text{dem}}\)) | FS-gated mechanistic features from DEM |
| 5 | Prithvi + LoRA (\(T_{\text{fm}}\)) | Planetary-scale environmental context |

CNN and physics branches per modality are merged by the **Complementary Modality Bridge (CMB)** into hybrid streams \(H_{\text{rgb}}\), \(H_{\text{dem}}\) before tri-stream fusion. The decoder is a **dual physics decoder**: two equation-based decode paths (RGB-proxy vs DEM-proxy physics variables) whose outputs are merged by **Mechanistic Path Equilibrium Fusion (MPEF)** — instability-weighted routing from each path's \((\alpha, h, m)\) proxies, not learned scalar gate fusion.

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| \(B\) | Batch size |
| \(C\) | Unified channel width (64) |
| \(H, W\) | Spatial height/width at a given level |
| \(\alpha\) | Slope angle (radians), pixel-wise |
| \(h\) | Soil-column / elevation proxy (positive) |
| \(m\) | Moisture / vegetation saturation proxy \(\in (0,1)\) |
| \(\text{FS}\) | Factor of Safety |
| \(\varepsilon\) | Numerical stabilizer (\(10^{-6}\)) |
| \(\odot\) | Element-wise (Hadamard) product |
| \([\cdot\,\|\,\cdot]\) | Channel-wise concatenation |

**Inputs (two raster streams per sample; no dataset-format changes):**

| Tensor | Shape | Content |
|--------|-------|---------|
| `stream_a` (\(x_1\)) | \(B \times 3 \times 256 \times 256\) | RGB image |
| `stream_b` (\(x_2\)) | \(B \times 3 \times 256 \times 256\) | NDVI + slope + DEM (L4S) or replicated DEM (Bijie) |

Physics proxies \((s, d, v)\) — slope, DEM, NDVI/vegetation — are derived **inside** `forward()` from \((x_1, x_2)\); no dataset changes are required.

---

## 3. Pyramid geometry and alignment

Three encoder families produce features at **different native spatial grids** when the input is 256×256:

### 3.1 EfficientNet-B4 pyramid (\(E_{\text{rgb}}, E_{\text{dem}}\))

| Index \(i\) | Channels | Spatial size |
|-------------|----------|--------------|
| 0 | 24 | 128×128 |
| 1 | 32 | 64×64 |
| 2 | 56 | 32×32 |
| 3 | 160 | 16×16 |
| 4 | 448 | 8×8 |

### 3.2 PhysicsEncoder pyramid (\(P_{\text{rgb}}, P_{\text{dem}}\))

Built by repeated stride-2 downsampling from full resolution; after `StreamProjector` all levels are \(C=64\):

| Level \(L\) | Spatial size |
|-------------|--------------|
| 0 | 256×256 |
| 1 | 128×128 |
| 2 | 64×64 |
| 3 | 32×32 |
| 4 | 16×16 |

### 3.3 Prithvi foundation pyramid (\(T_{\text{fm}}\))

Prithvi-EO-2.0-100M-TL with LoRA on attention projections; spatial maps are resized to legacy targets:

| Level \(L\) | Spatial size |
|-------------|--------------|
| 0 | 256×256 |
| 1 | 128×128 |
| 2 | 64×64 |
| 3 | 32×32 |
| 4 | 16×16 |

### 3.4 Cross-stream alignment maps

**Physics → EfficientNet (CMB):**  
\(\text{LEGACY\_LEVEL\_FOR\_EFFNET} = (1, 2, 3, 4, 4)\)  
At EfficientNet index \(i\), the physics tensor is taken from physics level \(\text{LEGACY\_LEVEL\_FOR\_EFFNET}[i]\) and bilinearly resized to match \(E^i\) spatially.

**Prithvi → EfficientNet (fusion projection):**  
The same index map is used when building the unified 64-channel fusion pyramids for MAO/TTEB.

**Spatial safety:** wherever decoder state and skip tensors differ in \(H \times W\), bilinear `match_spatial` resizes the auxiliary tensor to the reference before concatenation or gating.

---

## 4. Geotechnical foundation

### 4.1 Classical infinite slope model (parallel seepage)

\[
\text{FS} = \frac{c' + (\gamma_{\text{sat}} H - u)\cos^2\alpha \tan\phi'}{\gamma_{\text{sat}} H \sin\alpha \cos\alpha}, \quad u = m \gamma_w H \cos^2\alpha
\]

Failure when \(\text{FS} \le 1\).

### 4.2 Taylor-stabilized neural form (pixel mechanistic cell)

Pore-water pressure is moved from a subtractive resisting term to an additive driving term for stable backpropagation:

\[
\text{FS}_{\text{nn}} = \frac{e^{w_c} + e^{w_\phi}\, h\, \cos^2\alpha}{e^{w_\gamma}\, h\, \sin\alpha\cos\alpha + e^{w_m}\, m + \varepsilon}
\]

\[
\text{failure\_energy} = \psi - \text{FS}_{\text{nn}}, \quad \text{gate} = \sigma(\text{failure\_energy})
\]

\[
\text{PixelMechanisticCell}(x,\alpha,h,m) = \text{Conv}_{1\times1}(x) \odot \text{gate}
\]

Learnable per-channel parameters: \(w_c, w_\phi, w_\gamma, w_m, \psi\).

When \(\alpha, h, m\) spatial size differs from feature map \(x\), they are bilinearly interpolated to \((H,W)\) before the FS computation.

### 4.3 Latent mechanistic cell (encoder middle layers and decoder)

Angles are not applied in latent space. Hidden tensor \(H\) is split into resisting/driving manifolds:

\[
R = \text{Softplus}(W_R * H), \quad D = \text{Softplus}(W_D * H)
\]

\[
\text{LatentMechanisticCell}(H) = \text{Conv}_{3\times3}\left(\text{LeakyReLU}\left(\psi - \frac{R}{D+\varepsilon}\right)\right)
\]

### 4.4 Physics proxy mapper

Dataset channels are min-max normalized to \([0,1]\). Separate learnable mappers for the RGB-decoder path and DEM-decoder path produce equation variables:

\[
\alpha = \alpha_{\max}\cdot\sigma(w_\alpha s + b_\alpha), \quad \alpha_{\max}=\pi/2
\]
\[
h = \text{Softplus}(w_h d + b_h) + \varepsilon
\]
\[
m = \sigma(w_m v + b_m)
\]

where \(s\) = slope proxy, \(d\) = DEM proxy, \(v\) = NDVI/vegetation proxy.

**Proxy extraction from streams (no dataset change):**

| Layout | slope \(s\) | DEM \(d\) | vegetation \(v\) |
|--------|-------------|-----------|------------------|
| L4S (`stream_b` = NDVI, slope, DEM) | channel 1 | channel 2 | channel 0 |
| Bijie (`stream_b` = DEM ×3) | Sobel norm of DEM | channel 0 | green-red index from RGB |

---

## 5. Stage A — Penta-stream encoders

### 5.1 EfficientNet encoders (\(E_{\text{rgb}}, E_{\text{dem}}\))

- **Backbone:** `tf_efficientnet_b4` via `timm`, `features_only=True`, `out_indices=(0,1,2,3,4)`
- **RGB path:** 3-channel input from `stream_a`
- **DEM path:** 1-channel input (DEM channel extracted from `stream_b`)
- **Pretraining:** ImageNet weights; backbone frozen by default (`--freeze_backbone`)
- **Output:** native pyramids \(\{E_{\text{rgb}}^i\}_{i=0}^4\), \(\{E_{\text{dem}}^i\}_{i=0}^4\) with channel widths \([24, 32, 56, 160, 448]\)

### 5.2 Physics encoders (\(P_{\text{rgb}}, P_{\text{dem}}\))

Shared architecture; different input channels and separate `PhysicsProxyMapper` instances.

| Stage | Operation | Output (RGB path example) |
|-------|-----------|---------------------------|
| Input | RGB: 3 ch / DEM: 1 ch | \(B \times \{3,1\} \times 256 \times 256\) |
| L0 | PixelMechanisticCell + stem Conv3×3 + GN + ReLU | \(B \times 16 \times 256 \times 256\) |
| ↓ | Conv3×3, stride 2 | \(B \times 16 \times 128 \times 128\) |
| L1 | LatentMechanisticCell | \(B \times 32 \times 128 \times 128\) |
| ↓ | stride 2 | \(B \times 32 \times 64 \times 64\) |
| L2 | LatentMechanisticCell | \(B \times 64 \times 64 \times 64\) |
| ↓ | stride 2 | \(B \times 64 \times 32 \times 32\) |
| L3 | LatentMechanisticCell | \(B \times 64 \times 32 \times 32\) |
| ↓ | stride 2 | \(B \times 64 \times 16 \times 16\) |
| L4 | LatentMechanisticCell | \(B \times 64 \times 16 \times 16\) |
| Proj | StreamProjector per level | \(B \times C \times H_L \times W_L\) |

Outputs: \(\{P_{\text{rgb}}^L\}_{L=0}^4\), \(\{P_{\text{dem}}^L\}_{L=0}^4\), all at \(C=64\) after projection.

### 5.3 Prithvi foundation encoder (\(T_{\text{fm}}\))

- **Checkpoint:** `ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL`
- **Input:** 6-channel observed stack built inside `forward()`:
  - L4S: \([R, G, B, \text{NDVI}, \text{slope}, \text{DEM}]\)
  - Bijie: \([R, G, B, \text{DEM}, \text{Sobel(DEM)}, \text{vegIndex}(RGB)]\)
- **Normalization:** per-band \((x - 0.5) / 0.25\) (observed-rasters mode)
- **Backbone:** ViT encoder frozen; **LoRA** rank \(r=8\) on `attn.qkv` and `attn.proj`
- **Feature taps:** transformer blocks \(\{2, 5, 8, 11\}\)
- **Spatial maps:** token maps → 1×1 conv → bilinear resize to \(\{128, 64, 32, 16\}\) for L1–L4; L0 from upsampling L1 to 256×256
- **Projector:** StreamProjector → \(\{T_{\text{fm}}^L\}_{L=0}^4\) at \(C=64\)

### 5.4 StreamProjector

\[
\text{StreamProjector}(F) = \text{GN}(\text{ReLU}(\text{Conv}_{1\times1}(F;\, C_{\text{in}} \rightarrow C)))
\]

---

## 6. Stage B — Complementary Modality Bridge (CMB)

Before tri-stream fusion, each modality fuses its CNN pyramid with its physics pyramid **per EfficientNet index** \(i \in \{0,1,2,3,4\}\):

\[
H_{\text{rgb}}^i = \text{CMB}(E_{\text{rgb}}^i,\; \tilde{P}_{\text{rgb}}^i), \qquad
H_{\text{dem}}^i = \text{CMB}(E_{\text{dem}}^i,\; \tilde{P}_{\text{dem}}^i)
\]

where \(\tilde{P}^i\) is the physics level aligned to \(E^i\) via `LEGACY_LEVEL_FOR_EFFNET`.

**CMB equations:**

1. CNN projection (if needed): \(\tilde{E} = \text{GN}(\text{ReLU}(\text{Conv}_{1\times1}(E)))\) to \(C\) channels  
2. Physics calibration: \(\tilde{P} = \text{GN}(P)\)  
3. **Resonance gate:** \(g = \sigma\!\left(\text{Conv}_{1\times1}(\tilde{E} \odot \tilde{P})\right) \in [0,1]^{B \times 1 \times H \times W}\)  
4. **Blend path:** \(B_{\text{mix}} = \text{Conv}_{1\times1}([\tilde{E}\, \|\, \tilde{P}])\)  
5. **Hybrid output:**

\[
H = g \odot \tilde{P} + (1 - g) \odot \tilde{E} + B_{\text{mix}}
\]

**Interpretation:** when CNN texture and physics features co-activate (high \(g\)), mechanistic features dominate; when they disagree, CNN features are preserved and the blend path supplies a learned correction. This avoids discarding the proven ImageNet priors while injecting geotechnical structure.

**Unified fusion pyramids:** before MAO/TTEB, all three streams are projected to \(C=64\) at each EfficientNet index:

\[
\hat{H}_{\text{rgb}}^i = \text{rgb\_to\_fusion}_i(H_{\text{rgb}}^i), \quad
\hat{H}_{\text{dem}}^i = \text{dem\_to\_fusion}_i(H_{\text{dem}}^i), \quad
\hat{T}_{\text{fm}}^i = \text{fm\_align}_i(\text{align}(T_{\text{fm}}^{\text{LEGACY}[i]}, E_{\text{rgb}}^i))
\]

---

## 7. Stage C — Tri-stream fusion (MAO-GeoEGCA + TTEB)

The hybrid streams \(\hat{H}_{\text{rgb}}\), \(\hat{H}_{\text{dem}}\), and \(\hat{T}_{\text{fm}}\) (all \(C=64\)) enter the **Geo-Equilibrium Gated Cross-Attention** core. This is the sole encoder/skip fusion path in `GeoPhysicsLandslideNet` — there is no alternate gated CNN fusion branch in the main model.

### 7.1 MAO-GeoEGCA (bottleneck levels)

Applied at **EfficientNet pyramid indices 2 and 3** (spatially 32×32 and 16×16 for B4@256). These indices are chosen so the four-stage physics decoder upsampling chain reaches full 256×256 resolution (index 4 at 8×8 would end at 128×128).

For each bottleneck index \(i \in \{2, 3\}\):

**Step 1 — Physics anchor plane:**

\[
X_p = \text{Conv}_{1\times1}([\hat{H}_{\text{rgb}}^i \,\|\, \hat{H}_{\text{dem}}^i])
\]

**Step 2 — Physical equilibrium gate:**

\[
G = \sigma\!\left(\text{DWSepConv}_{3\times3}(\hat{T}_{\text{fm}}^i \odot X_p)\right) \in \mathbb{R}^{B \times 1 \times H \times W}
\]

**Step 3 — Asymmetric projections:**

\[
Q = W_q X_p, \quad K = W_k \hat{T}_{\text{fm}}^i, \quad V = W_v \hat{T}_{\text{fm}}^i
\]

**Step 4 — Manifold alignment:**

\[
\hat{Q} = \frac{Q}{\|Q\|_2}, \quad K' = K \odot \hat{Q}
\]

**Step 5 — Multi-head attention** (4 heads, \(d_h = C/4\)):

\[
\text{Attn} = \text{softmax}\!\left(\frac{Q K'^{\top}}{\sqrt{d_h}}\right) V
\]

**Step 6 — Gated residual output:**

\[
F^i = \text{Conv}_{3\times3}(\text{reshape}(\text{Attn}) \odot G) + X_p
\]

- \(F^2\) (32×32) is the **L3 neck** feature passed to the decoder (after optional 1×1 post-conv).  
- \(F^3\) (16×16) is the **L4 bottleneck** feature.

### 7.2 TTEB — Tri-Temporal Tri-Stream Bridge (skip levels)

Applied at EfficientNet indices \(\{0, 1, 2, 3\}\) (spatially 128², 64², 32², 16²). One `TriTemporalTriStreamBridge` module per level.

**Lattice:** for each stream \(s \in \{\text{rgb}, \text{dem}, \text{fm}\}\), gather nodes at scales \(L-1, L, L+1\) (boundary replication at pyramid ends).

**Steps:**

1. **Present anchor:**  
   \(A = \text{Conv}_{1\times1}([\hat{H}_{\text{rgb}}^L \,\|\, \hat{H}_{\text{dem}}^L \,\|\, \hat{T}_{\text{fm}}^L])\)

2. **Context stack:** concatenate 8 off-diagonal lattice nodes (3 streams × 3 temporal scales minus present diagonal) → \(B \times 8C \times H \times W\)

3. **Context mix:** depthwise Conv3×3 + pointwise Conv1×1 → \(H_{\text{ctx}}\)

4. **Stability map:**  
   \(\delta = \left|\psi - \dfrac{\text{Softplus}(W_R A)}{\text{Softplus}(W_D A)+\varepsilon}\right|\)

5. **Spatial attention:** \(Q\) from \(A\); \(K, V\) from \(H_{\text{ctx}}\); keys receive additive stream-phase embedding; chunked attention for memory efficiency

6. **Temporal routing:** \(\omega_\tau = \text{softmax}_\tau(W_\tau \delta)\) for \(\tau \in \{\text{prev}, \text{pres}, \text{next}\}\); output scaled by present-phase weight

7. **Skip output:**  
   \(S^L = A + \text{Conv}_{3\times3}(\text{AttnOut} \odot \sigma(\delta))\)

**Skip tensor list:** \(\{S^0, S^1, S^2, S^3\}\) at 128², 64², 32², 16² respectively. PGDI in the decoder resizes skips to the current decoder state when spatial sizes differ.

---

## 8. Stage D — Dual physics decoder

`DualPhysicsDecoder` runs two structurally identical `PhysicsDecoder` instances. Both share fused MAO/TTEB context \((F^3, F^2, \{S^L\})\) but use **stream-specific physics variables** \((\alpha, h, m)\) and **stream-specific CNN bottleneck residuals** \((a_5, b_5)\).

### 8.1 Bottleneck injection

EfficientNet deepest features \(a_5\) (RGB) and \(b_5\) (DEM) are projected to \(C=64\) and added to the fused bottleneck:

\[
F^3_A = F^3 + \text{Proj}_{64}(a_5), \qquad F^3_B = F^3 + \text{Proj}_{64}(b_5)
\]

(spatial alignment via bilinear resize when \(a_5, b_5\) are 8×8 and \(F^3\) is 16×16)

### 8.2 Single physics decoder path

Each path receives \((F^3_{(\cdot)}, F^2, \{S^L\}, \alpha, h, m)\) from its respective `PhysicsProxyMapper`.

| Step | Spatial (from 16×16 bottleneck) | Operation |
|------|----------------------------------|-----------|
| D4 | 16×16 | \(D_4 = \text{LatentMechanisticCell}(F^3)\) |
| ↑ | 32×32 | bilinear ×2 + Conv3×3 |
| Neck | 32×32 | \(D_3 \leftarrow \text{LatentMechanisticCell}(D_3) + F^2\) |
| PGDI@L3 | 32×32 | see §8.3 with \(S^3\) |
| Aux3 | 32×32 | \(\text{Conv}_{1\times1} \rightarrow\) logits |
| ↑ | 64×64 | bilinear ×2 + Conv |
| PGDI@L2 | 64×64 | with \(S^2\) |
| Aux2 | 64×64 | \(\text{Conv}_{1\times1} \rightarrow\) logits |
| ↑ | 128×128 | bilinear ×2 + Conv |
| PGDI@L1 | 128×128 | with \(S^1\) |
| ↑ | 256×256 | bilinear ×2 + Conv |
| PGDI@L0 | 256×256 | with \(S^0\) |
| Head | 256×256 | PixelMechanisticCell\((D_0, \alpha, h, m)\) + \(\text{Conv}_{1\times1} \rightarrow B \times 1 \times 256 \times 256\) |

Deep supervision: **aux3** at 32×32, **aux2** at 64×64; all heads upsampled to 256×256 before loss if needed.

### 8.3 PGDI — Physics-Gated Decoder Injection

\[
\beta = \sigma\!\left(\text{Conv}([D\, \|\, S])\right), \qquad
D' = D + \beta \odot \text{LatentMechanisticCell}(S)
\]

\(S\) is bilinearly resized to match \(D\) when shapes differ.

### 8.4 Dual-path output fusion (MPEF)

Path A (RGB proxies + \(a_5\) bias) and Path B (DEM proxies + \(b_5\) bias) each produce \((\text{main}, \text{aux2}, \text{aux3})\). **Mechanistic Path Equilibrium Fusion** merges logits:

1. Per-path failure energy from each path's \((\alpha, h, m)\):
   \[
   \text{FE}_p = \text{relu}\!\left(1 - \text{FS}_{\text{nn}}(\alpha_p, h_p, m_p)\right)
   \]
2. Instability routing weights: \([w_A, w_B] = \text{softmax}([\text{FE}_A, \text{FE}_B])\)  
3. Merged output:
   \[
   O = w_A O_A + w_B O_B + \text{Conv}_{1\times1}([O_A \, \| \, O_B])
   \]
4. Routing regularizer (per head):
   \[
   \mathcal{R}_{\text{MPEF}} = -\mathbb{E}\!\left[w_A \log(w_A+\varepsilon) + w_B \log(w_B+\varepsilon)\right]
   \]
   Lower entropy → more decisive path routing. Applied independently to **main**, **aux2**, and **aux3** (three `MechanisticPathEquilibriumFusion` modules).

**Design note:** MPEF replaces learned scalar gate fusion between decode paths. Routing is driven by **relative geotechnical instability** on each path's proxy variables, keeping the dual-decoder structure while closing the physics loop at the output.

---

## 9. End-to-end forward pass

### Algorithm 1 — PS-GPLNet forward (`GeoPhysicsLandslideNet`)

```
Input: stream_a (RGB), stream_b (topography)
──────────────────────────────────────────────────────────────
A. PROXY EXTRACTION
   (s, d, v) ← physics_proxies_from_streams(stream_a, stream_b)
   (α_r, h_r, m_r) ← PhysicsProxyMapper_rgb(s, d, v)
   (α_d, h_d, m_d) ← PhysicsProxyMapper_dem(s, d, v)

B. PENTA-STREAM ENCODING
   {E_rgb^i} ← EfficientNet_rgb(stream_a)          for i = 0..4
   {E_dem^i} ← EfficientNet_dem(dem_channel)       for i = 0..4
   {P_rgb^L} ← PhysicsEncoder_rgb(stream_a, α_r, h_r, m_r)
   {P_dem^L} ← PhysicsEncoder_dem(dem, α_d, h_d, m_d)
   {T_fm^L}  ← PrithviEncoder(observed_stack(stream_a, stream_b))

C. COMPLEMENTARY MODALITY BRIDGE
   for i = 0..4:
       H_rgb^i ← CMB(E_rgb^i, align(P_rgb, i))
       H_dem^i ← CMB(E_dem^i, align(P_dem, i))
       Ĥ_rgb^i ← rgb_to_fusion_i(H_rgb^i)
       Ĥ_dem^i ← dem_to_fusion_i(H_dem^i)
       T̂_fm^i  ← fm_align_i(align(T_fm, i))

D. TRI-STREAM FUSION
   F² ← MAO_GeoEGCA(T̂_fm², Ĥ_rgb², Ĥ_dem²)     // 32×32 neck
   F³ ← MAO_GeoEGCA(T̂_fm³, Ĥ_rgb³, Ĥ_dem³)     // 16×16 bottleneck
   for L in {0,1,2,3}:
       S^L ← TTEB({Ĥ_rgb}, {Ĥ_dem}, {T̂_fm}, level=L)

E. DUAL PHYSICS DECODERS + MPEF
   F³_A ← F³ + Proj(a_5)
   F³_B ← F³ + Proj(b_5)
   (main_A, aux2_A, aux3_A) ← PhysicsDecoder(F³_A, F², {S^L}, α_r, h_r, m_r)
   (main_B, aux2_B, aux3_B) ← PhysicsDecoder(F³_B, F², {S^L}, α_d, h_d, m_d)
   main  ← MPEF(main_A,  main_B,  α_r, h_r, m_r, α_d, h_d, m_d)
   aux2  ← MPEF(aux2_A,  aux2_B,  ...)
   aux3  ← MPEF(aux3_A,  aux3_B,  ...)

F. OUTPUT
   Return (main, aux2, aux3, reg_tuple)
```

### Algorithm 2 — MAO_GeoEGCA\((T, H_{\text{rgb}}, H_{\text{dem}})\)

```
1. X_p ← Conv1×1([H_rgb ; H_dem])
2. G   ← Sigmoid(DWSepConv3×3(T ⊙ X_p))
3. Q,K,V ← linear projections of X_p and T
4. Q̂ ← normalize(Q);  K' ← K ⊙ Q̂
5. Context ← MultiHeadAttention(Q, K', V)     // 4 heads
6. Out ← Conv3×3(reshape(Context) ⊙ G) + X_p
7. Return Out
```

### Algorithm 3 — TTEB\((L)\)

```
1. Gather F[s,τ] for s ∈ {rgb,dem,fm}, τ ∈ {L-1, L, L+1}  (replicate boundaries)
2. A ← Conv1×1([F_rgb,L ; F_dem,L ; F_fm,L])
3. H_ctx ← Mix(concat 8 off-diagonal nodes)
4. δ ← |ψ - Softplus(W_R·A) / (Softplus(W_D·A) + ε)|
5. AttnOut ← SpatialAttention(A, H_ctx) with stream-phase key bias
6. S^L ← A + Conv3×3(AttnOut ⊙ σ(δ))
7. Return S^L
```

### Algorithm 4 — PGDI\((D, S)\)

```
1. S' ← match_spatial(S, D)
2. β ← Sigmoid(Conv([D ; S']))
3. D' ← D + β ⊙ LatentMechanisticCell(S')
4. Return D'
```

### Algorithm 5 — MPEF\((O_A, O_B, \alpha_A, h_A, m_A, \alpha_B, h_B, m_B)\)

```
1. Resize (α, h, m) for each path to match logits spatial size
2. FE_A ← relu(1 - FS_nn(α_A, h_A, m_A))
3. FE_B ← relu(1 - FS_nn(α_B, h_B, m_B))
4. [w_A, w_B] ← softmax([FE_A, FE_B]) along path dimension
5. O ← w_A ⊙ O_A + w_B ⊙ O_B + Conv1×1([O_A ; O_B])
6. R ← -(w_A log(w_A+ε) + w_B log(w_B+ε)).mean()
7. Return O, R
```

---

## 10. Architecture diagram

```
stream_a ──► E_rgb (EffNet)  ──┐
         └──► P_rgb (Physics) ─┼──► CMB ──► Ĥ_rgb (64ch × 5 levels) ──┐
                                                                 │
stream_b ──► E_dem (EffNet)  ──┐                                 │
         └──► P_dem (Physics) ─┼──► CMB ──► Ĥ_dem (64ch × 5 levels) ─┼──┐
                                                                 │  │
         observed 6-band stack ──► Prithvi+LoRA ──► T̂_fm (64ch) ──┘  │
                                                                        │
              ┌──────────────── MAO @ idx 2,3 (32², 16²) ───────────────┤
              │                                                        │
              └── TTEB @ idx 0..3 (128², 64², 32², 16²) ───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         PhysicsDecoder_A (+a₅, α_r,h_r,m_r)   PhysicsDecoder_B (+b₅, α_d,h_d,m_d)
                    └──────────── MPEF (main, aux2, aux3) ─────────────┘
                                       │
                              segmentation logits
```

---

## 11. Shape trace (B=2, C=64, input 256×256, EfficientNet-B4)

| Tensor | Shape |
|--------|-------|
| stream_a | 2×3×256×256 |
| stream_b | 2×3×256×256 |
| \(E_{\text{rgb}}^0\) | 2×24×128×128 |
| \(E_{\text{rgb}}^4\) (\(a_5\)) | 2×448×8×8 |
| \(P_{\text{rgb}}^0\) | 2×64×256×256 |
| \(P_{\text{rgb}}^4\) | 2×64×16×16 |
| \(\hat{T}_{\text{fm}}^3\) | 2×64×16×16 |
| \(F^2\) (MAO neck) | 2×64×32×32 |
| \(F^3\) (MAO bottleneck) | 2×64×16×16 |
| \(S^3\) (TTEB skip) | 2×64×16×16 |
| \(S^0\) (TTEB skip) | 2×64×128×128 |
| main logits | 2×1×256×256 |
| aux2 logits | 2×1×64×64 (upsampled to 256 for loss) |
| aux3 logits | 2×1×32×32 (upsampled to 256 for loss) |

---

## 12. Loss, training, and outputs

| Item | Value |
|------|-------|
| Optimizer | Adam, lr = 3×10⁻⁴, weight decay = 10⁻⁴ |
| Segmentation loss | Tversky \(\alpha=0.3, \beta=0.7\) on main, aux2, aux3 |
| Head weights | \(w_1=1.0,\; w_2=0.6,\; w_3=0.4\) |
| MPEF routing regularization | \(\lambda_{\text{reg}} = 10^{-3}\) on \(\mathcal{R}_{\text{MPEF}}\) (3 terms: main, aux2, aux3) |
| Epochs | 100 |
| Batch size | 32 |
| Metric threshold | 0.5 |
| Metrics | acc, precision, recall, f1, iou (main head at 0.5 threshold) |
| Checkpoints | every 5 epochs + best on validation F1 |

\[
\mathcal{L} = w_1 \mathcal{L}_{\text{Tversky}}(\text{main}) + w_2 \mathcal{L}_{\text{Tversky}}(\text{aux2}) + w_3 \mathcal{L}_{\text{Tversky}}(\text{aux3}) + \lambda_{\text{reg}} \sum_{h \in \{\text{main},\text{aux2},\text{aux3}\}} \mathcal{R}_{\text{MPEF}}^{(h)}
\]

---

## 13. CLI configuration

| Flag | Default | Effect |
|------|---------|--------|
| `--no_mechanistic_gating` | | Physics cells run without FS gate (conv features only) |
| `--prithvi_snapshot` | required on cluster | Path to Prithvi HF cache or snapshot dir |
| `--freeze_backbone` | yes | Freeze EfficientNet weights |
| `--tteb_attn_chunk` | 1024 | TTEB attention chunk size |
| `--tteb_attn_low_res_max` | 4096 | TTEB downsample threshold for attention |

**Recommended Bijie command:**

```bash
python train_bijie.py \
  --dataset_root /path/to/Bijie-landslide-dataset \
  --output_dir ./outputs_gplnet_bijie \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --backbone tf_efficientnet_b4 --pretrained --freeze_backbone \
  --epochs 100 --batch_size 32 --lr 3e-4
```

**Recommended Landslide4Sense command:**

```bash
python training.py \
  --dataset_root /path/to/Landslide4Sense/dataset \
  --output_dir ./outputs_gplnet_l4s \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --backbone tf_efficientnet_b4 --pretrained --freeze_backbone \
  --epochs 100 --batch_size 32 --lr 3e-4
```

**External baseline (not part of this architecture):** DiGATe dual-stream gated U-Net remains in `from_first_steps/model_backup.py` for comparison experiments only.

---

## 14. Parameter budget (approximate)

| Component | Trainable params |
|-----------|------------------|
| EfficientNet encoders ×2 (frozen) | ~38M frozen |
| Physics encoders ×2 | ~2M |
| CMB ×10 (5 levels × 2 modalities) | ~0.5M |
| Prithvi LoRA (r=8) | ~0.5M |
| Prithvi backbone | ~100M frozen |
| MAO ×2 | ~0.3M |
| TTEB ×4 | ~1.5M |
| Dual physics decoder + PGDI | ~2M |
| MPEF ×3 (main, aux2, aux3) | negligible |
| Proxy mappers ×2 | negligible |
| **Trainable total (typical)** | **~7–8M** |

---

## 15. Narrative loop (reviewer-facing summary)

1. **Inlet (encoders):** spectral and topographic pixels are transformed by Taylor-stabilized FS gates into mechanistic feature pyramids, complemented — not replaced — by EfficientNet texture encoders merged through CMB.

2. **Traffic control (fusion):** MAO-GeoEGCA uses the combined hybrid physics plane as the query anchor and Prithvi as contextual key/value, modulated by an equilibrium gate that suppresses irrelevant planetary context where local geomechanics indicate stability. TTEB propagates tri-stream, tri-temporal context into skip connections governed by a latent stability map \(\delta\).

3. **Outlet (decoder):** two physics decoders upsample through latent mechanistic cells and PGDI-injected skips, apply FS gating at full resolution via pixel mechanistic cells, and merge through **MPEF** instability routing — closing the equation loop from data to segmentation mask.

PS-GPLNet is a **standalone pentastream architecture**: CMB hybrid encoding, MAO/TTEB fusion, dual mechanistic decoding, and MPEF path equilibrium — unified by the same Taylor-stabilized FS formalism at inlet, fusion anchor, skip stability, decode cells, and output routing.

---

## 16. Implementation file map

| Module | File |
|--------|------|
| Full model (`GeoPhysicsLandslideNet`) | `from_first_steps/model.py` |
| Pixel / latent cells, proxy mapper | `from_first_steps/physics/` |
| Physics encoder | `from_first_steps/encoders/physics_encoder.py` |
| Prithvi encoder | `from_first_steps/prithvi_encoder.py` |
| CMB | `from_first_steps/fusion/hybrid_stream_bridge.py` |
| MAO-GeoEGCA | `from_first_steps/fusion/mao_geo_egca.py` |
| TTEB | `from_first_steps/fusion/tteb.py` |
| Physics decoder, PGDI | `from_first_steps/decoder/physics_decoder.py`, `pgdi.py` |
| Dual decoder + MPEF | `from_first_steps/decoder/dual_physics_decoder.py`, `mechanistic_path_fusion.py` |
| DiGATe baseline (comparison only) | `from_first_steps/model_backup.py` |
| Implementation log | `from_first_steps/step_by_step_implementation.md` |
