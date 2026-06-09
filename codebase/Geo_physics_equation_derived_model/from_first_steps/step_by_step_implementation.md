# Step-by-Step Implementation Log (`from_first_steps/`)

Living document for the incremental graft of **GeoPhysicsLandslideNet** ideas onto the working **DiGATe / dual_stream_gated** baseline. Training, datasets, losses, and metrics stay unchanged unless noted.

**Target architecture docs:** `model_architecture.md`, `architecture_chain_of_thought.md`

**Working baseline reference:** `model_backup.py` (same family as `codebase/ablation_study/dual_stream_gated/`)

---

## Overall strategy

The full geo-physics model under-performed vs the paper-aligned dual-stream gated U-Net. Instead of editing `Geo_physics_equation_derived_model/model.py` directly, we evolve `from_first_steps/` in isolated steps:

| Step | Focus | Status |
|------|--------|--------|
| 1 | 3 encoders (EffNet RGB + EffNet DEM + Prithvi); keep `GateFuse` + paper decoder | Done |
| 2 | Document fusion (MAO-GeoEGCA + TTEB); keep EffNet encoders + paper decoder | Done |
| 3 | **5 encoders** (EffNet×2 + Physics×2 + Prithvi); CMB hybrid bridge; dual physics decoder | **Done** |
| 4 | Optional ablations (balanced/concat fusion, mechanistic gating off) | Planned |

---

# Step 1 — Tri-Stream Encoders (GateFuse fusion)

## Your prompt (summary)

- Add 3 encoders from the architecture docs (Prithvi + two stand-ins for physics RGB/DEM).
- Use **EfficientNet** for RGB/DEM, **not** physics cells yet.
- **Do not** implement document fusion yet — keep working `GateFuse`.
- Keep paper `AdaptiveDecoder` and dual-stream decoder `GateFuse`.
- Do not change training / data / metrics.
- Document in this file.

## What was implemented

| File | Change |
|------|--------|
| `model.py` | Tri-stream encoders on top of `model_backup.py` |
| `prithvi_encoder.py` | Prithvi-EO-2.0 + LoRA + 5-level pyramid |
| `step_by_step_implementation.md` | This log (was `STEP1_IMPLEMENTATION.md`) |

### Encoders (Step 1)

- **RGB:** EfficientNet on `stream_a` (3 ch)
- **DEM:** EfficientNet on DEM extracted from `stream_b` (1 ch)
- **Prithvi:** 6 ch observed stack built inside `forward()`; LoRA on `qkv`/`proj`

### Fusion (Step 1)

Chained `GateFuse` at encoder c3/c4:

```python
fused_ab = GateFuse(rgb, dem)
fused    = GateFuse(fused_ab, prithvi_projected)
```

### Decoder (Step 1 — unchanged from paper)

- Dual `AdaptiveDecoder` paths
- `GateFuse` on decoder features x3/x4
- `main`, `aux2`, `aux3` + gate regularization

### Step 1 diagram

```
RGB EffNet ──► {a1..a5}     ─┐
DEM EffNet ──► {b1..b5}     ─┼─ GateFuse chain @ c3,c4
Prithvi    ──► {fm0..fm4}   ─┘
         ▼
  decoderA / decoderB (paper) ── GateFuse x3,x4 ── heads
```

### Step 1 ablation flag (still available)

`--fusion gate` restores Step-1 encoder fusion.

---

# Step 2 — Document Fusion (MAO-GeoEGCA + TTEB)

## Your prompt (summary)

- Implement the **fusion strategy from your architecture docs** in `from_first_steps/`.
- Keep **EfficientNet RGB + DEM** and **Prithvi** encoders (no physics encoders yet).
- Keep the **paper decoder** (`AdaptiveDecoder` + dual-path + decoder `GateFuse`) — same as `dual_stream_gated`.
- Implement encoder fusion, skip fusion, and multimodal handling per your idea.
- Update this MD file with full rationale.

## What your documents specify (fusion=mao)

From `model_architecture.md` §3.4.4 and `architecture_chain_of_thought.md`:

### MAO-GeoEGCA (bottleneck L3/L4)

1. **Physics anchor:** `X_physics = Conv1x1([P_rgb; P_dem])`
2. **Gate:** `G = Sigmoid(DWConv3x3(T_fm ⊙ X_physics))`
3. **Q** from physics anchor; **K, V** from Prithvi `T_fm`
4. **Manifold alignment:** `K' = K ⊙ normalize(Q)`
5. Multi-head attention + gated residual: `Out = Conv3x3(Attn ⊙ G) + X_physics`

### TTEB (skip levels L0–L3)

Tri-Temporal Tri-Stream Bridge: lattice over rgb/dem/fm at scales L−1, L, L+1; stability routing; spatial attention; skip output `S^L`.

### What we deliberately did NOT implement in Step 2

| Component | Status |
|-----------|--------|
| Physics encoders (`PixelMechanisticCell`, `LatentMechanisticCell`) | Deferred → Step 3 |
| Physics decoder + PGDI | Deferred — paper `AdaptiveDecoder` kept |
| Balanced / concat fusion modes | Deferred — `mao` only for Step 2 |
| `PhysicsProxyMapper` | Not needed without physics encoders |

## Design decisions for Step 2

### 1. Unified C=64 fusion space

Your docs assume all three streams share width **C=64** at each pyramid level. Native EfficientNet-B4 uses **different** channel widths per level (`[24, 32, 56, 160, 448]`). TTEB concatenates features from levels L−1, L, L+1 — if channel widths differ across levels, concatenation breaks (e.g. 216 ch instead of 192 at level 0).

**Solution:** Before MAO/TTEB:

- `rgb_to_fusion[i]`: EffNet level → 64 ch  
- `dem_to_fusion[i]`: EffNet level → 64 ch  
- `fm_align[i]`: Prithvi 64 ch → 64 ch (spatial align to RGB)

After fusion, **`fusion_to_decoder[i]`** projects back to EffNet widths so the **paper decoder** receives correctly shaped skips.

### 2. Bridging TTEB skips to the paper decoder

Geo-physics `PhysicsDecoder` consumes TTEB skips via PGDI. The paper decoder expects:

```python
decoderA(f1, f2, f3, f4, f5)  # AdaptiveDecoder
```

**Step 2 mapping:**

| Decoder input | Source |
|---------------|--------|
| f1 | `fusion_to_decoder[0](TTEB level 0)` |
| f2 | `fusion_to_decoder[1](TTEB level 1)` |
| f3 | `fusion_to_decoder[2](MAO @ level 2)` |
| f4 | `fusion_to_decoder[3](MAO @ level 3)` |
| f5 | **Stream-specific bottleneck** `a5` (decoderA) or `b5` (decoderB) |

Both decoder paths share **tri-stream fused** early/mid features but keep **separate deepest bottlenecks** — preserving dual-stream structure at the coarsest scale.

### 3. Decoder fusion unchanged (paper)

`fuse_x3` and `fuse_x4` (`GateFuse` between decoderA and decoderB outputs) are **unchanged** from `dual_stream_gated`.

### 4. Regularization

MAO/TTEB do not emit gate regularization. With `--fusion mao`, `reg` has **2 terms** (decoder `GateFuse` only). Step 1 `gate` fusion still returns 6 terms.

## Files added/changed (Step 2)

| File | Change |
|------|--------|
| `fusion/mao_geo_egca.py` | MAO-GeoEGCA module |
| `fusion/tteb.py` | TriTemporalTriStreamBridge |
| `fusion/pyramid_utils.py` | `match_spatial` |
| `fusion/__init__.py` | Exports |
| `model.py` | `fusion_type='mao'` (default), unified 64ch fusion pyramids, decoder bridge |
| `train_bijie.py`, `training.py` | `--fusion`, `--tteb_attn_chunk`, `--tteb_attn_low_res_max` |

## Step 2 architecture diagram

```
stream_a ──► RGB EffNet ──► rgb_to_fusion ──► P_rgb (64ch pyramid) ─┐
stream_b ──► DEM EffNet ──► dem_to_fusion ──► P_dem (64ch pyramid) ─┼─┐
         └──► 6ch stack ──► Prithvi+LoRA ──► fm_align ──► T_fm ─────┘ │
                                                                       │
                    ┌──────────────── MAO @ L2,L3 (c3,c4) ────────────┤
                    │                                                  │
                    └── TTEB @ L0,L1 (c1,c2 skips) ──────────────────────┘
                                       │
                         fusion_to_decoder (64 → EffNet ch)
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
     AdaptiveDecoder(..., a5)                          AdaptiveDecoder(..., b5)
              └──────────── GateFuse x3,x4 (paper) ─────────────┘
                                       │
                              main + aux2 + aux3
```

## Model API (Step 2)

```python
model = DualStreamGateNet(
    fusion_type="mao",           # default; use "gate" for Step 1 ablation
    enable_prithvi=True,
    prithvi_snapshot="/path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL",
    tteb_attn_chunk=1024,
    tteb_attn_low_res_max=4096,
)
```

## Train command (Bijie, Step 2 default)

```bash
python train_bijie.py \
  --dataset_root /path/to/Bijie-landslide-dataset \
  --output_dir ./outputs_step2_bijie \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --fusion mao \
  --tversky_alpha 0.3 --tversky_beta 0.7 \
  --backbone tf_efficientnet_b4 --pretrained --freeze_backbone \
  --epochs 100 --batch_size 32 --lr 3e-4
```

Do **not** pass `--pretrained_path` unless loading a local **EfficientNet** checkpoint (not Prithvi).

## Verification (Step 2)

```
mao fusion: main/aux2/aux3 = [B,1,256,256], reg length = 2
gate fusion: reg length = 6 (Step 1 behavior)
```

## Expected outcome

Step 2 answers: **"Does your MAO+TTEB fusion improve over Step 1 GateFuse, while keeping the paper decoder?"**

Compare `outputs_step1_*` vs `outputs_step2_*` on the same splits/hyperparameters.

---

# Step 3 — Penta-Stream Encoders + Dual Physics Decoder

## Your prompt (summary)

- Expand from **3 to 5 encoders**: keep EffNet RGB, EffNet DEM, Prithvi **and add** physics encoders for RGB and DEM.
- Design a **principled fusion strategy** for 5 encoders (not a hack).
- Replace decoder with **physics decoder + PGDI** from the architecture docs, but preserve the **dual-stream decoder strategy** (two paths + `GateFuse` on outputs).
- Update `model_architecture.md` for the 5-encoder design.
- Document here as Step 3.

## Design rationale

### Why not replace EfficientNet with physics?

Step 2 showed strong metrics with CNN texture + Prithvi + MAO/TTEB. Physics encoders encode **geotechnical stability** (FS gates) but lack ImageNet-scale texture priors. Removing CNNs would discard proven features.

**Solution:** run both in parallel and merge per level with the **Complementary Modality Bridge (CMB)**:

- High CNN–physics resonance → trust mechanistic features.
- Low resonance → keep CNN, add learned blend correction.

### Five-encoder → three-stream fusion

MAO/TTEB were designed for three streams at C=64. Rather than inventing a 5-way fusion block (which would be harder to train and interpret), we **collapse to hybrid bimodal streams** first:

| Modality | Encoders | CMB output |
|----------|----------|------------|
| RGB | EffNet RGB + PhysicsEncoder RGB | \(H_{\text{rgb}}^L\) |
| DEM | EffNet DEM + PhysicsEncoder DEM | \(H_{\text{dem}}^L\) |
| FM | Prithvi (unchanged) | \(T_{\text{fm}}^L\) |

Then Step 2 fusion applies unchanged on \((H_{\text{rgb}}, H_{\text{dem}}, T_{\text{fm}})\).

### Physics proxies (no dataset changes)

`slope`, `dem`, `ndvi` are derived inside `forward()` via `physics_proxies_from_streams(x1, x2)` — same logic as the parent GPLNet pipeline. Separate `PhysicsProxyMapper` instances for RGB-path vs DEM-path decoders.

### Dual physics decoder (paper strategy preserved)

| Component | Step 2 (paper) | Step 3 (physics) |
|-----------|----------------|------------------|
| Paths | `AdaptiveDecoder` A / B | `PhysicsDecoder` A / B |
| Bottleneck | `a5` / `b5` (EffNet) | `a5` / `b5` projected to C=64 and added to \(F^4\) |
| Shared context | MAO/TTEB skips (projected to EffNet ch) | MAO/TTEB skips at C=64 directly |
| Physics vars | — | RGB proxies → path A; DEM proxies → path B |
| Output fusion | `GateFuse` on x3/x4 features | `GateFuse` on main/aux2/aux3 **logits** |

## Files added/changed (Step 3)

| File | Change |
|------|--------|
| `physics/pixel_cell.py` | `PixelMechanisticCell` (FS gate at L0) |
| `physics/latent_cell.py` | `LatentMechanisticCell` (latent FS ratio) |
| `physics/proxy_mapper.py` | `PhysicsProxyMapper` (slope/dem/ndvi → α,h,m) |
| `physics/__init__.py` | Exports |
| `encoders/physics_encoder.py` | 5-level physics pyramid + projectors |
| `encoders/projector.py` | `StreamProjector` |
| `encoders/__init__.py` | Exports |
| `fusion/hybrid_stream_bridge.py` | `ComplementaryModalityBridge` (CMB) |
| `decoder/pgdi.py` | `PhysicsGatedDecoderInjection` |
| `decoder/physics_decoder.py` | Upsampling + PGDI + aux heads |
| `decoder/dual_physics_decoder.py` | Dual paths + `GateFuse` on logits |
| `decoder/__init__.py` | Exports |
| `model.py` | 5 encoders, CMB, physics decoder path, ablation flags |
| `train_bijie.py`, `training.py` | `--decoder`, `--no_physics_encoders`, `--no_mechanistic_gating` |
| `model_architecture.md` | §3.1.1 CMB, §3.6.3 dual physics decoder, Algorithm 1B |

## Step 3 architecture diagram

```
stream_a ──► EffNet RGB ────────┐
         └──► PhysicsEnc RGB ──┼──► CMB ──► H_rgb (64ch pyramid) ─┐
stream_b ──► EffNet DEM ────────┐                                  │
         └──► PhysicsEnc DEM ──┼──► CMB ──► H_dem (64ch pyramid) ─┼─┐
         └──► 6ch stack ──► Prithvi ──► T_fm (64ch pyramid) ──────┘ │
                                                                      │
                         MAO @ L2,L3 + TTEB @ L0–L3 ──────────────────┘
                                       │
                         DualPhysicsGatedDecoder
                         (PhysicsDecoder A + B, GateFuse logits)
                                       │
                              main + aux2 + aux3 + reg (3 terms)
```

## Model API (Step 3 defaults)

```python
model = DualStreamGateNet(
    fusion_type="mao",              # required for physics decoder
    decoder_type="physics",         # default Step 3
    enable_physics_encoders=True,   # default
    enable_prithvi=True,
    mechanistic_gating=True,
    prithvi_snapshot="/path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL",
)
```

## Ablation flags

| Flag | Effect |
|------|--------|
| `--decoder paper` | Step 2 `AdaptiveDecoder` + `fusion_to_decoder` bridge |
| `--no_physics_encoders` | Step 2 tri-stream (EffNet only); requires `--decoder paper` |
| `--fusion gate` | Step 1 encoder `GateFuse`; requires `--decoder paper` |
| `--no_mechanistic_gating` | Physics cells run without FS gate (features only) |
| `--no_prithvi` | Disable Prithvi (not compatible with `--decoder physics`) |

## Train command (Bijie, Step 3 default)

```bash
python train_bijie.py \
  --dataset_root /path/to/Bijie-landslide-dataset \
  --output_dir ./outputs_step3_bijie \
  --prithvi_snapshot /path/to/models--ibm-nasa-geospatial--Prithvi-EO-2.0-100M-TL \
  --fusion mao \
  --decoder physics \
  --tversky_alpha 0.3 --tversky_beta 0.7 \
  --backbone tf_efficientnet_b4 --pretrained --freeze_backbone \
  --epochs 100 --batch_size 32 --lr 3e-4
```

Reproduce Step 2 exactly:

```bash
python train_bijie.py ... --decoder paper --no_physics_encoders --fusion mao
```

## Verification (Step 3)

```
physics + mao + prithvi: main/aux2/aux3 = [B,1,256,256], reg length = 3
paper + mao (Step 2):    reg length = 2
gate (Step 1):           reg length = 4–6 depending on Prithvi
```

Local smoke test (no Prithvi weights): `--no_prithvi --decoder paper --fusion gate` runs forward.

## Expected outcome

Step 3 answers: **"Does adding physics encoders + dual physics decoder on top of Step 2 fusion improve landslide segmentation?"**

Compare `outputs_step2_*` vs `outputs_step3_*` on identical splits/hyperparameters.

---

## Planned Step 4+

1. Balanced / concat fusion modes from parent `model_architecture.md`.
2. Further ablations: CMB off (concat CNN+physics), single physics decoder path.

---

## Requirement → implementation matrix

| Requirement | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| Only `from_first_steps/` | Yes | Yes | Yes |
| No training/data/metrics changes | Yes | Yes (CLI flags) | Yes (CLI flags) |
| Encoders | 3 (EffNet×2 + Prithvi) | Same | **5** (EffNet×2 + Physics×2 + Prithvi) |
| Physics encoders | No | No | **Yes** (parallel to EffNet, not replacement) |
| Document fusion | `GateFuse` | MAO + TTEB | MAO + TTEB on **hybrid** streams |
| Decoder | Paper | Paper | **Dual physics** (default); paper via flag |
| Prithvi path via CLI | Yes | Yes | Yes |
