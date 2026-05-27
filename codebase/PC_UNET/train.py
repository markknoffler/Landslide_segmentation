import torch
import torch.optim as optim
from model import PC_UNet
from loss import PC_SegmentationLoss

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_running_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass returns the mask AND the PC residuals
        output, residuals = model(images)

        # Calculate the hybrid loss
        loss, seg_l, pc_l = criterion(output, masks, residuals)

        loss.backward()
        optimizer.step()

        total_running_loss += loss.item()

    return total_running_loss / len(dataloader)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PC_UNet(num_classes=5, in_channels=14).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = PC_SegmentationLoss(lambda_pc=0.1)

# Usage in your main loop:
# avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
