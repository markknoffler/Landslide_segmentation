"""
TPAMI Evaluation Utilities for PS-GPLNet.
Provides low-compute, high-impact analysis tools:
1. t-SNE / PCA Latent Manifold Projection
2. Boundary Complexity Metrics (Hausdorff Distance & Boundary F-Score)
3. Input Pertribution & Robustness Decay Analysis
Runs comfortably on a single local GPU (RTX A4000 20GB or RTX 3060 12GB).
"""

from __future__ import annotations
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# 1. BOUNDARY COMPLEXITY METRICS

def compute_hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute symmetric Hausdorff Distance between prediction and ground-truth boundary points."""
    # Find contours
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(pred_contours) == 0 or len(gt_contours) == 0:
        if len(pred_contours) == 0 and len(gt_contours) == 0:
            return 0.0
        return float(max(pred_mask.shape)) # Penalty for empty boundaries
        
    pts_pred = np.vstack([c[:, 0, :] for c in pred_contours])
    pts_gt = np.vstack([c[:, 0, :] for c in gt_contours])
    
    d1 = directed_hausdorff(pts_pred, pts_gt)[0]
    d2 = directed_hausdorff(pts_gt, pts_pred)[0]
    return float(max(d1, d2))


def compute_boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, theta: float = 2.0) -> float:
    """
    Calculate the Boundary F-score (BF-score) matching contour segments.
    Ref: Csurka et al., "What is a good evaluation measure for semantic segmentation?"
    """
    pred_edge = cv2.Canny(pred_mask.astype(np.uint8) * 255, 50, 150) > 0
    gt_edge = cv2.Canny(gt_mask.astype(np.uint8) * 255, 50, 150) > 0
    
    if not np.any(pred_edge) and not np.any(gt_edge):
        return 1.0
    if not np.any(pred_edge) or not np.any(gt_edge):
        return 0.0
        
    # Dilate edges to construct match regions within tolerance theta
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(2 * theta + 1), int(2 * theta + 1)))
    dilated_pred = cv2.dilate(pred_edge.astype(np.uint8), kernel) > 0
    dilated_gt = cv2.dilate(gt_edge.astype(np.uint8), kernel) > 0
    
    # Precision: fraction of pred edges close to gt edges
    prec = np.sum(pred_edge & dilated_gt) / (np.sum(pred_edge) + 1e-8)
    # Recall: fraction of gt edges close to pred edges
    rec = np.sum(gt_edge & dilated_pred) / (np.sum(gt_edge) + 1e-8)
    
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# 2. LATENT MANIFOLD VISUALIZER

def extract_and_plot_tsne(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device, 
    num_samples: int = 1000,
    save_path: str = "latent_manifold_tsne.png"
):
    """
    Pass samples through the network, collect intermediate bottleneck features,
    and save a t-SNE plot comparing feature clustering.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    print("[TPAMI Eval] Extracting features for t-SNE visualization...")
    collected = 0
    
    # Register hook to capture bottleneck/decoder features
    bottleneck_feats = []
    def hook_fn(module, input, output):
        # Flatten features over spatial dimensions for representation mapping
        if isinstance(output, tuple):
            output = output[0]
        # output shape: [B, C, H, W] -> Pool to [B, C]
        pooled = torch.mean(output, dim=(2, 3))
        bottleneck_feats.append(pooled.detach().cpu().numpy())

    # We attach to the bottleneck or decoder output (adjust model target module accordingly)
    # Assumes model has an encoder bottleneck or we hook into the decoder path
    if hasattr(model, 'mpef'):
        hook = model.mpef.register_forward_hook(hook_fn)
    else:
        # Fallback to general model output hooks
        hook = model.register_forward_hook(hook_fn)
        
    try:
        with torch.no_grad():
            for batch in dataloader:
                # Load streams depending on dataset layout
                if isinstance(batch, dict):
                    x_rgb = batch["image"].to(device)
                    # Use matching DEM bands if present
                    x_dem = batch.get("dem", x_rgb).to(device)
                    y = batch["mask"]
                else:
                    x_rgb, x_dem, y = batch[0].to(device), batch[1].to(device), batch[2]
                
                # Forward pass trigger hook
                _ = model(x_rgb, x_dem)
                
                # Retrieve hooked features
                features = bottleneck_feats.pop(0)
                # Assign labels by pixel percentage or primary class
                flat_y = y.view(y.size(0), -1).cpu().numpy()
                labels = (np.sum(flat_y, axis=1) > (flat_y.shape[1] * 0.05)).astype(int) # 5% threshold
                
                all_features.append(features)
                all_labels.append(labels)
                collected += x_rgb.size(0)
                if collected >= num_samples:
                    break
    finally:
        hook.remove()
        
    X = np.concatenate(all_features, axis=0)[:num_samples]
    labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    # Reduce dimension via t-SNE
    print(f"[TPAMI Eval] Running t-SNE on {X.shape[0]} feature maps...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_embedded = tsne.fit_transform(X)
    
    # Plotting
    plt.figure(figsize=(8, 6), dpi=150)
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="coolwarm", alpha=0.7, edgecolors='w', linewidths=0.5)
    plt.colorbar(scatter, label="Landslide Presence (>5% area)")
    plt.title("t-SNE Projection of PS-GPLNet Latent Space", fontsize=12)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[TPAMI Eval] Saved t-SNE scatter plot to {save_path}")


# 3. PERTURBATION & ROBUSTNESS TEST

def run_noise_robustness_test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_levels: list[float] = [0.0, 0.05, 0.1, 0.2, 0.3],
    save_plot_path: str = "robustness_decay.png"
):
    """
    Run evaluation metrics while systematically injecting Gaussian noise to the input streams.
    Saves a line graph demonstrating metric decay rate.
    """
    model.eval()
    iou_scores = []
    f1_scores = []
    
    print("[TPAMI Eval] Commencing noise robustness test...")
    for noise in noise_levels:
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x_rgb = batch["image"].to(device)
                    x_dem = batch.get("dem", x_rgb).to(device)
                    y = batch["mask"].to(device)
                else:
                    x_rgb, x_dem, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                
                # Apply additive Gaussian noise
                if noise > 0:
                    x_rgb = x_rgb + torch.randn_like(x_rgb) * noise
                    x_dem = x_dem + torch.randn_like(x_dem) * noise
                    
                logits = model(x_rgb, x_dem)
                # Parse logits (assumes sigmoid threshold 0.5)
                pred = (torch.sigmoid(logits) >= 0.5).int()
                
                tp += int((pred * y).sum().item())
                fp += int((pred * (1 - y)).sum().item())
                fn += int(((1 - pred) * y).sum().item())
                
        # Calculate dataset-wide metrics for this noise level
        eps = 1e-8
        iou_val = tp / (tp + fp + fn + eps)
        f1_val = (2 * tp) / (2 * tp + fp + fn + eps)
        
        iou_scores.append(iou_val)
        f1_scores.append(f1_val)
        print(f"  Noise={noise:.2f} | IoU={iou_val:.4f} | F1={f1_val:.4f}")
        
    # Plotting Decay Curve
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(noise_levels, f1_scores, marker='o', label="F1-Score", color="crimson")
    plt.plot(noise_levels, iou_scores, marker='s', label="IoU", color="darkblue")
    plt.xlabel("Gaussian Noise $\sigma$", fontsize=10)
    plt.ylabel("Performance Metric", fontsize=10)
    plt.title("PS-GPLNet Performance Decay Under Input Perturbation", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_plot_path, bbox_inches='tight')
    plt.close()
    print(f"[TPAMI Eval] Robustness curve saved to {save_plot_path}")
