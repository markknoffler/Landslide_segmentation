import torch
import torch.nn as nn

class PC_SegmentationLoss(nn.Module):
    def __init__(self, lambda_pc=0.1):
        super(PC_SegmentationLoss, self).__init__()
        self.seg_criterion = nn.CrossEntropyLoss()
        self.lambda_pc = lambda_pc

    def forward(self, output, target, error_tensors):
        # 1. Standard Segmentation Accuracy
        seg_loss = self.seg_criterion(output, target)

        # 2. Predictive Coding Loss (MSE of residuals)
        # We want the 'normal' mountain features to have zero error
        pc_loss = 0
        for err in error_tensors:
            pc_loss += torch.mean(err**2)

        # Total Loss
        total_loss = seg_loss + (self.lambda_pc * pc_loss)
        return total_loss, seg_loss, pc_loss
