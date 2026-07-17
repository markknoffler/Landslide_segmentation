# Model Architecture: GeoPhysicsLandslideNet

## 1. Overview
The **GeoPhysicsLandslideNet** architecture is a novel multi-modal, physics-informed deep learning framework for landslide segmentation. It expands upon dual-stream baselines by integrating structural geological knowledge (mechanistic gating) alongside optical and topographical foundation features. 

The architecture consists of three main stages:
1. **Penta-Stream Encoders**: Five distinct encoders process RGB, DEM, and their respective physics proxies, as well as multispectral representations via the Prithvi-EO-2.0 foundation model.
2. **Complementary Modality Bridge (CMB) & Trimodal Fusion**: The encoders are merged into three hybrid streams, which are subsequently fused using Multi-scale Attention-based Operator (MAO-GeoEGCA) and Tri-Temporal Tri-Stream Bridge (TTEB) mechanisms.
3. **Dual Physics Decoder**: A dual-path decoder leveraging Physics Gated Decoder Injection (PGDI) to refine the feature maps and generate final landslide segmentation probabilities.

## 2. Penta-Stream Encoders
To capture both visual texture and geotechnical stability constraints, the model utilizes five parallel encoders:
- **RGB Encoder (EfficientNet-B4)**: Extracts high-level visual and textural patterns from optical imagery.
- **DEM Encoder (EfficientNet-B4)**: Captures multi-scale topographical structures from elevation data.
- **Physics Encoder (RGB Proxies)**: Employs `PixelMechanisticCell` and `LatentMechanisticCell` layers to interpret Factor of Safety (FS) stability metrics using RGB-derived proxies (e.g., NDVI).
- **Physics Encoder (DEM Proxies)**: Processes topographical variables (slope, elevation) through mechanistic gates to determine localized structural vulnerabilities.
- **Foundation Model Encoder (Prithvi-EO-2.0)**: Uses a LoRA-adapted 6-band input (RGB, NDVI, slope, DEM) to generate a robust multispectral context pyramid.

## 3. Fusion Strategies
### 3.1 Complementary Modality Bridge (CMB)
Instead of a complex 5-way fusion, the architecture first collapses the optical/topographical and physics streams into hybrid bimodal streams:
- `H_rgb`: Fusion of EffNet RGB and Physics Encoder RGB via CMB.
- `H_dem`: Fusion of EffNet DEM and Physics Encoder DEM via CMB.
- `T_fm`: The Prithvi feature pyramid remains unchanged.

### 3.2 MAO-GeoEGCA & TTEB
The three hybrid streams (`H_rgb`, `H_dem`, `T_fm`) undergo cross-modal fusion:
- **MAO-GeoEGCA**: Applied at the deepest bottleneck layers (L2, L3) for global manifold alignment and context modulation.
- **TTEB (Tri-Temporal Tri-Stream Bridge)**: Applied at the skip connections (L0-L3) to maintain high-resolution spatial details across modalities.

## 4. Dual Physics Decoder
The fused feature pyramids are projected back to standard spatial channels and fed into a **DualPhysicsGatedDecoder**:
- Consists of two symmetrical paths (`PhysicsDecoder A` and `PhysicsDecoder B`).
- Uses **PGDI (Physics Gated Decoder Injection)** to embed physical proxy data (slope, DEM, NDVI) directly into the upsampling layers.
- The final outputs of the two decoder paths are fused using a `GateFuse` module operating on the logits (main, aux2, aux3) to produce the final segmentation map.

## 5. Loss and Regularization
The network is optimized using a weighted combination of Tversky Loss (to handle class imbalance between landslide and background pixels) and a specific Geomorphological Alignment Loss, which penalizes boundary formations that contradict the underlying slope and elevation gradients.
