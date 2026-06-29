import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # hidden_dim * 2 because the input comes from a BiLSTM
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x : (batch, seq_len, hidden_dim*2)
        weights = torch.softmax(self.attention(x), dim=1)   # (batch, seq_len, 1)
        output  = torch.sum(x * weights, dim=1)             # (batch, hidden_dim*2)
        return output


class StrongMCQModel(nn.Module):


    def __init__(self, vocab_size: int):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, 256, padding_idx=0)

        # CNN
        self.conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)

        # BiLSTM  (num_layers=1; dropout is a no-op but kept for future layers)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        # Attention
        self.attention = Attention(128)

        # Classifier head
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(256, 128)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)               # (B, L, 256)

        x = x.permute(0, 2, 1)             # (B, 256, L)
        x = torch.relu(self.conv1(x))      # (B, 128, L)
        x = x.permute(0, 2, 1)             # (B, L, 128)

        x, _ = self.lstm(x)                # (B, L, 256)

        x = self.attention(x)              # (B, 256)
        x = self.dropout(x)

        x = self.relu(self.fc1(x))         # (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)                    # (B, 1)
        return x.squeeze()


def build_model(vocab_size: int) -> StrongMCQModel:
    return StrongMCQModel(vocab_size)
