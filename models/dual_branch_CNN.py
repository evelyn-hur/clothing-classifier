import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchCNN(nn.Module):
    def __init__(self, in_channels, feature_dim=256):
        super(DualBranchCNN, self).__init__()
        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels, 64,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.conv_layers(x)


class DualBranchCNNClassifier(nn.Module):
    def __init__(self, num_classes=13, shape_weight=1.0, texture_weight=1.0, feature_dim=256):
        super(DualBranchCNNClassifier, self).__init__()
        self.shape_weight = shape_weight
        self.texture_weight = texture_weight

        # No Color
        self.shape_branch = DualBranchCNN(
            in_channels=1, feature_dim=feature_dim)
        # Color
        self.texture_branch = DualBranchCNN(
            in_channels=3, feature_dim=feature_dim)

        self.fc1 = nn.Linear(feature_dim * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        shape_feats = self.shape_branch(x_gray)
        texture_feats = self.texture_branch(x)

        shape_feats = shape_feats.view(x.size(0), -1)
        texture_feats = texture_feats.view(x.size(0), -1)

        shape_feats = shape_feats * self.shape_weight
        texture_feats = texture_feats * self.texture_weight
        combined = torch.cat([shape_feats, texture_feats], dim=1)

        x = F.relu(self.fc1(combined))
        out = self.fc2(x)
        return out
