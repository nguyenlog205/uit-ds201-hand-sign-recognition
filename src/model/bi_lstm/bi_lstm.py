import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(
        self,
        num_joints,
        num_dims,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_dropout_rate,
        num_classes,
    ):
        super(BiLSTM, self).__init__()

        self.feature_dim = num_joints * num_dims
        self.hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(
            input_size = self.feature_dim,
            hidden_size = self.hidden_size,
            num_layers = lstm_num_layers,
            bidirectional = True,
            batch_first = True,
        )

        self.dropout = nn.Dropout(p = lstm_dropout_rate)

        self.classifier = nn.Linear(
            self.hidden_size * 2,
            num_classes,
        )

    def forward(self, x):
        x_sequence, _ = self.lstm(x)
        x_sequence = self.dropout(x_sequence)
        x = x_sequence[:, -1, :]

        x = self.classifier(x)

        return x

    