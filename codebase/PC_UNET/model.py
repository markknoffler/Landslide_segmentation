import torch
import torch.nn as nn
import torch.nn.functional as F

class PCBlock(nn.Module):
    """Calculates the residual error between Encoder and Decoder"""
    def __init__(self, channels):
        super(PCBlock, self).__init__()
        # 1x1 conv to ensure features are in the same mathematical space
        self.align = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, encoder_features, decoder_prediction):
        # The core PC logic: Error = Evidence - Prediction
        prediction = self.align(decoder_prediction)
        error = encoder_features - prediction
        return error

class PC_UNet(nn.Module):
    def __init__(self, num_classes=5, in_channels=14): # Set to 14 for your bands
        super(PC_UNet, self).__init__()

        # Encoder (Standard Bottom-Up Evidence)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        # PC Comparison Units (The Novelty)
        self.pc4 = PCBlock(512)
        self.pc3 = PCBlock(256)
        self.pc2 = PCBlock(128)
        self.pc1 = PCBlock(64)

        # Decoder (Top-Down Prediction)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 512) # Processes the error only

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 64)

        self.output = nn.Conv2d(64, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder Pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        # Decoder Pass with PC Handshake
        # Level 4
        pred4 = self.upconv4(bn)
        err4 = self.pc4(e4, pred4)
        d4 = self.dec4(err4)

        # Level 3
        pred3 = self.upconv3(d4)
        err3 = self.pc3(e3, pred3)
        d3 = self.dec3(err3)

        # Level 2
        pred2 = self.upconv2(d3)
        err2 = self.pc2(e2, pred2)
        d2 = self.dec2(err2)

        # Level 1
        pred1 = self.upconv1(d2)
        err1 = self.pc1(e1, pred1)
        d1 = self.dec1(err1)

        final_out = self.output(d1)
        
        # We return the errors so the loss function can penalize them
        return final_out, [err4, err3, err2, err1]
