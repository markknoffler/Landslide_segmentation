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
| 2 | Document fusion (MAO-GeoEGCA + TTEB); keep EffNet encoders + paper decoder | **Done** |
| 3 | Replace EffNet RGB/DEM with physics encoders | Planned |
| 4 | Physics decoder + PGDI (optional; only if paper decoder is swapped) | Planned |

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

## Planned Step 3+

1. **Physics encoders** — swap EffNet RGB/DEM for `PhysicsEncoder` + `PhysicsProxyMapper`; keep MAO/TTEB + paper decoder.
2. **Optional:** balanced/concat fusion modes as ablations.
3. **Optional later:** physics decoder + PGDI instead of `AdaptiveDecoder`.

---

## Requirement → implementation matrix

| Requirement | Step 1 | Step 2 |
|-------------|--------|--------|
| Only `from_first_steps/` | Yes | Yes |
| No training/data/metrics changes | Yes | Yes (added CLI flags only) |
| 3 encoders | EffNet + EffNet + Prithvi | Same |
| Physics encoders | No | No |
| Document fusion | No (`GateFuse`) | Yes (MAO + TTEB) |
| Paper decoder | Yes | Yes |
| Prithvi path via CLI | Added after Step 1 | `--prithvi_snapshot` |
