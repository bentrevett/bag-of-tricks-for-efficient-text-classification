import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.embedding(x)
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1)
        x = self.fc(x)
        return x
