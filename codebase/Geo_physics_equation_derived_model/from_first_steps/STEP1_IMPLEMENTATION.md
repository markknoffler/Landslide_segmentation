## Step 1 — Tri-Stream Encoder Integration (from_first_steps)

## Your goal

You have a **working** landslide segmentation baseline in `from_first_steps/` (the DiGATe-style dual-stream gated U-Net) that trains well and produces good metrics. Separately, you designed a much more ambitious **geo-physics equation-derived** architecture in `Geo_physics_equation_derived_model/` (documented in `model_architecture.md` and `architecture_chain_of_thought.md`) with:

1. **RGB physics encoder** — custom infinite-slope mechanistic cells  
2. **DEM physics encoder** — same equation on topography  
3. **Prithvi foundation encoder** — IBM-NASA Prithvi-EO-2.0 + LoRA  
4. **Fusion core** — MAO-GeoEGCA / balanced tri-stream / concat fusion  
5. **Physics decoder** — mechanistic cells + PGDI skip injection  

That full geo-physics model did **not** train as well as the working baseline. Your strategy is therefore:

> **Do not edit the main geo-physics codebase directly.**  
> Instead, evolve `from_first_steps/` **step by step**, grafting geo-physics ideas onto the model that already works — changing **only model architecture** per step, leaving training, data preprocessing, and metrics untouched.

---

## Your exact prompt (Step 1)

> Replace the 2 encoders with the **3 encoders** defined in the architecture documents, but:
> - the two geo-physics encoders should **not** be physics-based yet — use standard EfficientNet encoders for RGB and DEM  
> - **add the Prithvi encoder** as defined in the documents  
> - **do not** implement the document fusion strategy yet (MAO / balanced / TTEB)  
> - keep the **existing fusion strategy and decoder** from the working `from_first_steps` model  
> - do **not** change training, data preprocessing, or metrics  
> - write an MD file in `from_first_steps/` explaining what was done and why  

---

## What the architecture documents specify (target end-state)

| Stream | Intended role | Step-1 status |
|--------|---------------|---------------|
| RGB encoder | Physics equation cells (`PixelMechanisticCell` + `LatentMechanisticCell`) | **Deferred** — standard EfficientNet on `stream_a` |
| DEM encoder | Same physics stack on 1-channel DEM | **Deferred** — standard EfficientNet on extracted DEM |
| FM encoder | Prithvi-EO-2.0-100M-TL, blocks {2,5,8,11}, LoRA on `qkv`/`proj`, 6-band input, `StreamProjector` → C=64 pyramid | **Implemented** |
| Fusion | MAO-GeoEGCA / balanced / concat tri-stream | **Deferred** — kept working `GateFuse` |
| Decoder | Physics decoder + PGDI | **Deferred** — kept `AdaptiveDecoder` |

The documents define a **5-level pyramid** (L0–L4 at 256→16 px) with unified width C=64. The working baseline uses **native EfficientNet-B4** channel widths `[24, 32, 56, 160, 448]`. Step 1 bridges Prithvi's 64-ch maps into the EfficientNet channel space at the two fusion levels (c3, c4) only.

---

## What was implemented

### Files changed / added

| File | Action |
|------|--------|
| `model.py` | Rebuilt from `model_backup.py` (working baseline). Extended to tri-stream encoders. |
| `prithvi_encoder.py` | **New** — Prithvi-EO-2.0 + LoRA + multi-scale pyramid (adapted from parent `encoders/prithvi_lora.py`). |
| `STEP1_IMPLEMENTATION.md` | **New** — this document. |

### Unchanged (by design)

- `training.py`, `train_bijie.py`  
- `dataset.py`, `bijie_dataset.py`, `data_processing.py`  
- `losses.py`, `metrics.py`  

Training still calls `DualStreamGateNet(x1, x2)` and expects `(main, aux2, aux3, reg)`.

---

## Architecture diagram (Step 1)

```
stream_a (RGB, 3ch) ──► EfficientNet RGB encoder ──► {a1..a5}
                                                          │
stream_b ──► extract DEM (1ch) ──► EfficientNet DEM enc ──► {b1..b5}
                                                          │
stream_a + stream_b ──► 6ch observed stack ──► Prithvi+LoRA ──► {fm0..fm4}
                                                          │         │
                                                          │    project fm3→c3, fm4→c4
                                                          ▼         ▼
                                              GateFuse chain at c3, c4
                                              (RGB↔DEM, then ↔ Prithvi)
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                       ▼
                    AdaptiveDecoder (RGB skips)                          AdaptiveDecoder (DEM skips)
                              │                                                       │
                              └──────────── GateFuse x3, x4 ──────────────────────────┘
                                                          │
                                              main + aux2 + aux3 heads
```

### Encoder details

**RGB encoder** — `tf_efficientnet_b4` (or `--backbone`), 3 input channels, frozen by default.

**DEM encoder** — same backbone, **1 input channel**. DEM is extracted inside `forward()`:
- Landslide4Sense (`stream_b` = NDVI, slope, DEM): `dem = stream_b[:, 2:3]`
- Bijie (`stream_b` = DEM × 3): `dem = stream_b[:, 0:1]`

**Prithvi encoder** — `PrithviFoundationEncoder`:
- Checkpoint: `/4PI/prithvi_weights/.../Prithvi-EO-2.0-100M-TL`
- LoRA rank 8 on attention `qkv` and `proj`; backbone frozen
- 6-channel input built in-model via `observed_stack_from_streams()`:
  - **L4S**: `[R, G, B, NDVI, slope, DEM]`
  - **Bijie**: `[R, G, B, DEM, sobel_slope(DEM), vegetation_index(RGB)]`
- Normalization: `observed_rasters` mode `(x - 0.5) / 0.25`
- Output pyramid: 5 levels at 64 channels, sizes `[256, 128, 64, 32, 16]`
- Levels fm3 (32×32) and fm4 (16×16) are projected to EfficientNet c3/c4 widths before fusion

### Fusion (unchanged philosophy)

The working model fused RGB and DEM at encoder levels **c3** and **c4** with `GateFuse`, then ran **two** `AdaptiveDecoder` paths and fused decoder features at x3/x4.

Step 1 extends this with a **chained** `GateFuse`:

```python
fused_ab, reg1 = GateFuse(rgb, dem)
fused,    reg2 = GateFuse(fused_ab, prithvi_projected)
```

No MAO-GeoEGCA, no balanced tri-stream attention, no TTEB — those are reserved for later steps.

### Decoder (unchanged)

- Dual `AdaptiveDecoder` (transposed cross-attention upsampling + attention gates)  
- `SubPixelUp` + `OutConv` head  
- Auxiliary heads at x3, x4 with bilinear upsample to full resolution  

---

## Why these design choices

| Choice | Reason |
|--------|--------|
| Build 6-channel Prithvi input inside `forward()` | Training/datasets unchanged; compatible with both L4S and Bijie loaders |
| Project Prithvi only at c3/c4 | Matches where the working model already fuses streams; avoids reshaping the whole EfficientNet pyramid |
| Keep `GateFuse` instead of document fusion | Isolates the effect of adding Prithvi; if metrics change, we know it's from the new encoder stream |
| Standard EfficientNet for RGB/DEM | Step 1 validates the 3-encoder **wiring** before introducing physics cells |
| Separate `prithvi_encoder.py` | Self-contained `from_first_steps/` module; no dependency on the main geo-physics package layout |
| `DualStreamGateNet` name preserved | `training.py` / `train_bijie.py` import unchanged |

---

## Model API

```python
model = DualStreamGateNet(
    backbone="tf_efficientnet_b4",
    pretrained=True,
    freeze_backbone=True,
    enable_prithvi=True,          # set False to ablate Prithvi
    lora_rank=8,
    prithvi_snapshot=None,        # defaults to /4PI/prithvi_weights/...
)
main, aux2, aux3, reg = model(stream_a, stream_b)
```

`share_backbone` is accepted for API compatibility but ignored (three encoders are always separate).

---

## Verification

Forward-pass smoke test (CPU, `pretrained=False`):

```
L4S shapes:  main/aux2/aux3 = [B, 1, 256, 256], reg tuple length = 6
Bijie shapes: main/aux2/aux3 = [B, 1, 256, 256], reg tuple length = 6
```

Regularization tuple grew from 4 terms (dual-stream) to 6 terms (extra GateFuse at c3/c4 for Prithvi). `DualStreamLoss` averages all reg terms, so training remains valid.

---

## Expected result after training

This step answers: **"Does adding a Prithvi foundation stream improve (or at least not harm) the working baseline?"**

Compare against `model_backup.py` / prior checkpoints on the same splits and hyperparameters. You should watch:

- val F1 / IoU vs previous dual-stream runs  
- training stability (Prithvi LoRA params are trainable even when EfficientNet is frozen)  
- GPU memory (Prithvi adds ~100M frozen + LoRA overhead)

No training run was executed as part of this implementation step.

---

## Suggested next steps (Step 2+)

These are **not** implemented yet — listed here so the next prompt can pick up cleanly:

1. **Replace RGB/DEM EfficientNet encoders** with `PhysicsEncoder` (pixel + latent mechanistic cells) while keeping Prithvi and the current GateFuse/decoder.  
2. **Swap fusion** from chained `GateFuse` to document fusion (`concat` baseline first, then `balanced` or `mao`).  
3. **Swap decoder** from `AdaptiveDecoder` to `PhysicsDecoder` + PGDI.  
4. **Align pyramids** to unified C=64 (`legacy` EfficientNet pyramid mode) for full compatibility with the main geo-physics fusion modules.

---

## Quick reference — your requirement → implementation mapping

| Your requirement | Implementation |
|------------------|----------------|
| Only edit `from_first_steps/` | Yes — `model.py` + `prithvi_encoder.py` only |
| Don't touch training / data / metrics | Yes — unchanged |
| 3 encoders as in documents | RGB EffNet + DEM EffNet + Prithvi LoRA |
| Not physics encoders yet | EfficientNet stand-ins for RGB/DEM |
| Add Prithvi as in documents | `PrithviFoundationEncoder` with observed-raster 6ch stack |
| Don't use document fusion yet | `GateFuse` retained |
| Keep decoder same | `AdaptiveDecoder` retained |
| Document everything in MD | This file |
