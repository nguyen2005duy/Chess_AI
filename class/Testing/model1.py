import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, num_legal_moves):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # 64 for board squares
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_legal_moves)  # Output scores for each legal move

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output a score for each legal move
        return x
