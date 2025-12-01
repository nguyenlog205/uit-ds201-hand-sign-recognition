import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class RGB_based_model(nn.Module):
    def __init__(
        self,
        # Hyperparameters for ResNet50
        resnet_dropout_rate = 0.5,

        # Hyperparameters for LSTM
        lstm_hidden_size = 512,
        lstm_num_layers = 1,
        lstm_dropout_rate = 0.5,
        num_classes = 2,
    ):
        super(RGB_based_model, self).__init__()
        
        # --- ResNet50
        self.resnet50 = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.resnet50_dropout = nn.Dropout(p = resnet_dropout_rate)

        # --- LSTM ---
        self.feature_dim = 2048
        self.lstm = nn.LSTM(
            input_size = self.feature_dim,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
        )
        self.lstm_dropout = nn.Dropout(p = lstm_dropout_rate)

        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        N, C, D, H, W = x.shape
        x = x.transpose(1, 2).contiguous()

        # --- ResNet50 ---
        x = x.view(N * D, C, H, W)
        x = self.resnet50(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(N * D, self.feature_dim) 
        x = self.resnet50_dropout(x)

        # --- LSTM ---
        x = x.view(N, D, self.feature_dim)
        x_sequence, _ = self.lstm(x) 
        x_sequence = self.lstm_dropout(x_sequence)

        # --- Classifier ---
        x = x_sequence[:, -1, :] 
        x = self.classifier(x)

        return x