# Geo-Physics Equation-Derived Landslide Segmentation Network

**Model name:** GeoPhysicsLandslideNet (GPLNet)  
**Task:** Binary landslide segmentation from RGB + topography + geospatial foundation features  
**Input resolution:** 256×256 (training/evaluation aligned with ablation study)  
**Unified feature width:** C = 64  

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| C | Unified channel width (64) |
| H, W | Spatial height/width at a given level |
| α | Slope angle (radians), pixel-wise |
| h | Soil-column / elevation proxy (positive) |
| m | Moisture / saturation proxy ∈ (0,1) |
| FS | Factor of Safety |
| ε | Numerical stabilizer (1e-6) |

**Encoder scales (L = 0…4):**

| Level L | Resolution |
|---------|------------|
| 0 | 256×256 |
| 1 | 128×128 |
| 2 | 64×64 |
| 3 | 32×32 |
| 4 | 16×16 |

Streams: **rgb** (physics RGB encoder), **dem** (physics DEM encoder), **fm** (EfficientNet-B4 via `timm` **or** Prithvi-EO-2.0 + LoRA; select with `--fm_backbone`).

---

## 2. Geotechnical foundation

### 2.1 Classical infinite slope model (parallel seepage)

\[
\text{FS} = \frac{c' + (\gamma_{\text{sat}} H - u)\cos^2\alpha \tan\phi'}{\gamma_{\text{sat}} H \sin\alpha \cos\alpha}, \quad u = m \gamma_w H \cos^2\alpha
\]

Failure when FS ≤ 1.

### 2.2 Taylor-stabilized neural form (pixel cells)

Pore-water is moved from a subtractive resisting term to an additive driving term (first-order expansion) for stable backpropagation:

\[
\text{FS}_{\text{nn}} = \frac{e^{w_c} + e^{w_\phi}\, h\, \cos^2\alpha}{e^{w_\gamma}\, h\, \sin\alpha\cos\alpha + e^{w_m}\, m + \varepsilon}
\]

\[
\text{failure\_energy} = \psi - \text{FS}_{\text{nn}}, \quad \text{gate} = \sigma(\text{failure\_energy})
\]

\[
\text{PixelMechanisticCell}(x,\alpha,h,m) = \text{Conv}_{1\times1}(x) \odot \text{gate}
\]

Learnable per-channel: \(w_c, w_\phi, w_\gamma, w_\mu, \psi\).

### 2.3 Latent continuum cell (middle layers)

Angles are not applied in latent space. Hidden tensor H is split into resisting/driving manifolds:

\[
R = \text{Softplus}(W_R * H), \quad D = \text{Softplus}(W_D * H)
\]

\[
\text{LatentMechanisticCell}(H) = \text{Conv}_{3\times3}\left(\text{LeakyReLU}\left(\psi - \frac{R}{D+\varepsilon}\right)\right)
\]

### 2.4 Physics proxy mapper

Dataset channels are min-max normalized to [0,1]. Learnable maps produce equation variables:

\[
\alpha = \alpha_{\max}\cdot\sigma(w_\alpha s + b_\alpha), \quad \alpha_{\max}=\pi/2
\]
\[
h = \text{Softplus}(w_h d + b_h) + \varepsilon
\]
\[
m = \sigma(w_m v + b_m)
\]

Separate mappers for RGB and DEM streams (`PhysicsProxyMapper`).

---

## 3. Module specifications

### 3.1 Physics encoder (RGB and DEM)

Shared architecture; different input channels and proxy mappers.

| Stage | Operation | Output shape |
|-------|-----------|--------------|
| Input | RGB: 3ch or DEM: 1ch | B×3×256×256 or B×1×256×256 |
| L0 | PixelMechanisticCell + Conv3×3×2 + GN + ReLU | B×32×256×256 |
| ↓ | Conv3×3, stride 2 | B×32×128×128 |
| L1 | LatentMechanisticCell | B×48×128×128 |
| ↓ | Conv3×3, stride 2 | B×48×64×64 |
| L2 | LatentMechanisticCell | B×64×64×64 |
| ↓ | Conv3×3, stride 2 | B×64×32×32 |
| L3 | LatentMechanisticCell | B×64×32×32 |
| ↓ | Conv3×3, stride 2 | B×64×16×16 |
| L4 | LatentMechanisticCell | B×64×16×16 |
| Proj | StreamProjector 1×1 Conv per level | B×C×H_L×W_L |

Outputs: \(\{P_{\text{rgb}}^L\}_{L=0}^4\), \(\{P_{\text{dem}}^L\}_{L=0}^4\).

### 3.2 Prithvi foundation encoder

- **Checkpoint:** `ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL` (stored under `/4PI/prithvi_weights/`)
- **Input:** B×6×T×H×W with T=1, H=W=256, bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
- **Normalization:** per-band `(x - mean) / std` from config
- **Backbone:** `PrithviViT.forward_features` at blocks {2, 5, 8, 11}
- **LoRA:** rank r=8 on `Block.attn.qkv` and `Block.attn.proj` (encoder frozen except LoRA)
- **Spatial maps:** `prepare_features_for_image_model` → 1×1 conv + bilinear resize to {128, 64, 32, 16} for L1–L4; L0 from bilinear upsample of L1 to 256×256
- **Projector:** StreamProjector → \(T_{\text{fm}}^L\)

### 3.3 StreamProjector

\[
\text{StreamProjector}(F) = \text{GN}(\text{ReLU}(\text{Conv}_{1\times1}(F; C_{\text{in}}\rightarrow C)))
\]

### 3.4 Fusion strategies (levels L0–L4)

The model now supports **two fusion modes**, selectable at run time:

- **`--fusion balanced`** (default): symmetric tri-stream fusion with intra-stream refinement.  
- **`--fusion concat`**: simple baseline — concat RGB-physics, DEM-physics, FM at each level, 1×1 conv to C.  
- **`--fusion mao`**: legacy MAO + TTEB design (FM-heavy cross-attention).

#### 3.4.1 Concat tri-stream fusion (baseline, all levels)

`ConcatTriStreamLevel(C)` — diagnostic / ablation baseline:

\[
F^L_{\text{concat}} = \text{ReLU}(\text{GN}(\text{Conv}_{1\times1}([P_{\text{rgb}}^L; P_{\text{dem}}^L; T_{\text{fm}}^L])))
\]

Applied at L3/L4 (bottleneck) and L0–L3 (skips). No cross-stream attention or gating. Used with `--decoder conv` to test whether poor F1 is fusion-related or pipeline/data-related.

#### 3.4.2 Balanced tri-stream fusion (default, levels L3, L4)

Modules:

- `IntraStreamBlock(C)` — per-stream refinement (RGB-physics, DEM-physics, FM).  
  - Depthwise 3×3 + 1×1 conv + GN + GELU  
  - Optional spatial multi-head self-attention when \(H\cdot W \le 4096\)  
  - 1×1 Conv FFN (expansion=2) + residual
- `BalancedTriStreamFusion(C)` — symmetric bottleneck fusion at L3/L4.  
- `BalancedTriStreamSkip(C)` — symmetric skip fusion at L0–L3.

Let \(R^L = P_{\text{rgb}}^L\), \(D^L = P_{\text{dem}}^L\), \(F^L = T_{\text{fm}}^L\).

**Intra-stream refinement (per stream, per level):**

1. \(R' = \text{IntraStreamBlock}(R^L)\)  
2. \(D' = \text{IntraStreamBlock}(D^L)\)  
3. \(F' = \text{IntraStreamBlock}(F^L)\)

Each block is applied **independently** to its stream (no cross-encoder attention here).

**Calibration and symmetric gating:**

4. \(\tilde{R} = \text{GN}(R')\), \(\tilde{D} = \text{GN}(D')\), \(\tilde{F} = \text{GN}(F')\)  
5. Concatenate along channels and predict logits:
   \[
   H_{\text{gate}} = \text{Conv}_{1\times1}([\tilde{R};\tilde{D};\tilde{F}])
   \]
   \[
   G = \text{Conv}_{1\times1}(H_{\text{gate}}) \in \mathbb{R}^{B\times 3\times H\times W}
   \]
6. Stream probabilities (per-pixel softmax):
   \[
   [g_R, g_D, g_F] = \text{softmax}(G,\ \text{dim}=1)
   \]
7. Symmetric mixed representation:
   \[
   M = g_R \odot \tilde{R} + g_D \odot \tilde{D} + g_F \odot \tilde{F}
   \]

**Symmetric cross-attention:**

8. Keys/values use the **equal-weight average** of all refined streams:
   \[
   K_{\text{src}} = V_{\text{src}} = (\tilde{R} + \tilde{D} + \tilde{F}) / 3
   \]
9. Linear projections:
   \[
   Q = W_q M,\quad K = W_k K_{\text{src}},\quad V = W_v V_{\text{src}}
   \]
10. Reshape to heads (H×W tokens), scale, and attend:
    \[
    \text{Attn} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_h}}\right)V
    \]
11. Reshape back to feature map and project:
    \[
    F_{\text{ctx}} = W_o(\text{Attn})
    \]
12. Output bottleneck feature:
    \[
    F^L_{\text{balanced}} = \text{Conv}_{3\times3}(\text{GN}(M + F_{\text{ctx}}))
    \]

Here **no single stream is structurally privileged**: queries, keys, and values all come from symmetric combinations of \(R^L, D^L, F^L\).

#### 3.4.3 Balanced tri-stream skips (levels L0–L3)

`BalancedTriStreamSkip(C)` mirrors the intra-stream + gating part of `BalancedTriStreamFusion` but **omits cross-attention** for efficiency.

For each level L:

1. \(R' = \text{IntraStreamBlock}(P_{\text{rgb}}^L)\), \(D' = \text{IntraStreamBlock}(P_{\text{dem}}^L)\), \(F' = \text{IntraStreamBlock}(T_{\text{fm}}^L)\)  
2. \(\tilde{R}, \tilde{D}, \tilde{F}\) via GN  
3. Gating logits and softmax as above → \(g_R, g_D, g_F\)  
4. Mixed skip:
   \[
   S_{\text{balanced}}^L = \text{Conv}_{3\times3}\left(g_R \odot \tilde{R} + g_D \odot \tilde{D} + g_F \odot \tilde{F}\right)
   \]

These skips are used directly by the decoder in place of `TTEB` outputs when `--fusion balanced`.

#### 3.4.4 Legacy MAO-GeoEGCA + TTEB (fusion=`mao`)

When `--fusion mao` is selected, the original FM-heavy design is used.

**MAO-GeoEGCA (levels L3, L4):** applied independently at each bottleneck scale.

1. **Physics anchor:** \(X_p = \text{Conv}_{1\times1}([P_{\text{rgb}}; P_{\text{dem}}])\)  
2. **Equilibrium gate:** \(G = \sigma(\text{DWSepConv}_{3\times3}(T_{\text{fm}} \odot X_p))\) → B×1×H×W  
3. **Projections:** \(Q = W_q X_p\), \(K = W_k T_{\text{fm}}\), \(V = W_v T_{\text{fm}}\) (flattened to tokens N=H·W)  
4. **Manifold alignment:** \(\hat{Q} = Q/\|Q\|_2\), \(K' = K \odot \hat{Q}\)  
5. **Multi-head attention** (4 heads): \(\text{Attn} = \text{softmax}(Q K'^T / \sqrt{d_h}) V\)  
6. **Output:** \(F = \text{Conv}_{3\times3}(\text{reshape}(\text{Attn}) \odot G) + X_p\)

In this mode, skips are produced by **TTEB** (Tri-Temporal Tri-Stream Bridge) as described below.

### 3.5 TTEB — Tri-Temporal Tri-Stream Bridge (levels L0–L3)  *(fusion=`mao` only)*

**Lattice:** For each level L and stream s ∈ {rgb, dem, fm}, nodes at scales L−1, L, L+1 (replicate boundaries).

**Steps:**

1. **Present anchor:** \(A = \text{Conv}_{1\times1}([P_{\text{rgb}}^L; P_{\text{dem}}^L; T_{\text{fm}}^L])\)
2. **Context stack:** Concatenate 8 off-diagonal nodes (3 streams × 3 scales minus present diagonal) → B×(8C)×H×W
3. **Mix:** Depthwise 3×3 + Pointwise 1×1 → H_ctx
4. **Stability map:** \(\delta = \left|\psi - \frac{\text{Softplus}(W_R A)}{\text{Softplus}(W_D A)+\varepsilon}\right|\)
5. **Temporal routing:** For each stream, \(\omega_{s,\tau} = \text{softmax}_\tau(W_\tau \delta)\) for τ ∈ {prev, pres, next}
6. **Stream phase:** Keys receive additive embedding \(e_s \in \mathbb{R}^C\)
7. **Attention:** Q from A; K,V from mixed context; output scaled by δ
8. **Skip:** \(S^L = A + \text{Conv}_{3\times3}(\text{AttnOut})\)

### 3.6 Decoders: physics vs conv

The network exposes a **decoder switch** via `--decoder`:

- `--decoder physics` (default): original physics-aware decoder with PGDI.  
- `--decoder conv`: standard UNet-style convolutional decoder (ablation).

#### 3.6.1 Physics decoder + PGDI

**Top:** Fused \(F^{4}\) from the chosen fusion module at L4.

| Step | Operation |
|------|-----------|
| D4 | LatentMechanisticCell(F4) at 16×16 |
| ↑ | Bilinear ×2 + Conv |
| PGDI@L3 | \(\beta=\sigma(\text{Conv}([D;S^3]))\), \(D \leftarrow D + \beta \odot \text{LatentCell}(S^3)\) |
| D3 | LatentMechanisticCell |
| ↑ | ×2 |
| PGDI@L2 | same with S² |
| Aux head | Conv1×1 → logits (deep supervision) |
| ↑ | ×2 |
| PGDI@L1 | same with S¹ |
| ↑ | ×2 |
| PGDI@L0 | same with S⁰ |
| Head | PixelMechanisticCell + Conv1×1 → B×1×256×256 logits |

#### 3.6.2 ConvDecoder (standard convolutional decoder)

`ConvDecoder` uses a UNet-style upsampling path without explicit physics gates.

- Inputs: bottleneck features \(F^{4}\), \(F^{3}\), and skips \(\{S^L\}_{L=0}^3\) from the chosen fusion mode.  
- Operations:

| Step | Operation |
|------|-----------|
| Stem4 | DoubleConv(F4) at 16×16 |
| ↑ | Bilinear ×2 + Conv |
| Fuse@L3 | Concatenate with S³, DoubleConv → D3, Aux3 head: Conv1×1(D3) |
| ↑ | ×2 |
| Fuse@L2 | Concatenate with S², DoubleConv → D2, Aux2 head: Conv1×1(D2) |
| ↑ | ×2 |
| Fuse@L1 | Concatenate with S¹, DoubleConv → D1 |
| ↑ | ×2 |
| Fuse@L0 | Concatenate with S⁰, DoubleConv → D0 |
| Head | Conv1×1(D0) → B×1×256×256 logits |

The interface matches the physics decoder (three outputs: main, aux2, aux3), so loss and metrics are unchanged; only the internal decoding strategy differs.

**Auxiliary outputs:** aux2 at 64×64, aux3 at 32×32 (upsampled to full res for loss).

---

## Algorithm 1: End-to-end forward pass

```
Input: stream_a (RGB), stream_b (topo), prithvi_6band, proxies (slope, dem, ndvi)
1. (α_rgb, h_rgb, m_rgb) ← ProxyMapper_rgb(slope, dem, ndvi)
2. (α_dem, h_dem, m_dem) ← ProxyMapper_dem(slope, dem, ndvi)
3. {P_rgb^L} ← PhysicsEncoder_rgb(stream_a, α_rgb, h_rgb, m_rgb)
4. {P_dem^L} ← PhysicsEncoder_dem(dem_ch, α_dem, h_dem, m_dem)
5. {T_fm^L} ← PrithviEncoder(prithvi_6band)
6. If `fusion=concat` or `fusion=balanced`:
      - For L in {3,4}: F^L ← ConcatTriStreamLevel or BalancedTriStreamFusion(...)
      - For L in {0,1,2,3}: S^L ← ConcatTriStreamSkip or BalancedTriStreamSkip(...)
   Else (`fusion=mao`):
      - For L in {3,4}: F^L ← MAO_GeoEGCA(P_rgb^L, P_dem^L, T_fm^L)
      - For L in {0,1,2,3}: S^L ← TTEB({P_rgb}, {P_dem}, {T_fm}, L)
7. If `decoder=physics`: (main, aux2, aux3) ← PhysicsDecoder(F^4, F^3, {S^L}, α, h, m at full res)  
   Else (`decoder=conv`): (main, aux2, aux3) ← ConvDecoder(F^4, F^3, {S^L}, α, h, m)  (α,h,m unused)
9. Return (main, aux2, aux3)
```

---

## Algorithm 2: TTEB(L)

```
Input: pyramid features per stream, level L
1. Gather F[s,τ] for s∈{rgb,dem,fm}, τ∈{L-1,L,L+1} with boundary replicate
2. A ← Conv([F[rgb,L], F[dem,L], F[fm,L]])
3. H_ctx ← Mix(concat off-diagonal 8 tensors)
4. δ ← |ψ - Softplus(W_R*A)/(Softplus(W_D*A)+ε)|
5. AttnOut ← TriStreamTemporalAttention(A, H_ctx, δ, stream_phase_emb)
6. S^L ← A + Conv(AttnOut)
7. Return S^L
```

---

## Algorithm 3: MAO_GeoEGCA

```
Input: P_rgb, P_dem, T_fm at scale L
1. X_p ← Conv1x1([P_rgb, P_dem])
2. G ← Sigmoid(DWSepConv3x3(T_fm ⊙ X_p))
3. Q,K,V ← linear projections of X_p and T_fm
4. Q̂ ← normalize(Q); K' ← K ⊙ Q̂
5. Context ← MultiHeadAttention(Q, K', V)
6. Out ← Conv3x3(Context ⊙ G) + X_p
7. Return Out
```

---

## Algorithm 4: PGDI(D, S, α, h, m optional)

```
1. β ← Sigmoid(Conv([D, S]))
2. S' ← LatentMechanisticCell(S)
3. D' ← D + β ⊙ S'
4. Return D'
```

---

## Shape trace (B=2, C=64)

| Tensor | Shape |
|--------|-------|
| stream_a | 2×3×256×256 |
| stream_b | 2×3×256×256 |
| fm_input (efficientnet) | 2×3×256×256 RGB |
| fm_input (prithvi) | 2×6×256×256 observed stack |
| P_rgb^0, P_dem^0 | 2×64×256×256 |
| P_rgb^4, T_fm^4 | 2×64×16×16 |
| F^4 (fused) | 2×64×16×16 |
| S^3 | 2×64×32×32 |
| main logits | 2×1×256×256 |

---

## Training protocol (comparability)

Aligned with `codebase/ablation_study/baseline_models/common/trainer.py` and dual-stream paper:

| Item | Value |
|------|-------|
| Optimizer | Adam, lr=3e-4, wd=1e-4 |
| Loss | Tversky α=0.3, β=0.7; weights 1.0, 0.6, 0.4 |
| Epochs | 100 |
| Batch | 32 |
| Threshold | 0.5 |
| Metrics | acc, precision, recall, f1, iou, auroc, auprc, image_best_f1 |
| Checkpoints | every 5 epochs + best on val F1 |
| Bijie split | 70/20/10 |
| L4S split | 90/10 from TrainData only |

---

## Parameter budget (approximate)

| Component | Params |
|-----------|--------|
| Physics encoders ×2 | ~2M |
| Prithvi encoder (frozen) | ~100M |
| LoRA (r=8) | ~0.5M |
| MAO ×2 scales | ~0.3M |
| TTEB ×4 levels | ~1.5M |
| Decoder + PGDI | ~1M |
| **Trainable total** | **~5–6M** (excluding frozen backbone) |
